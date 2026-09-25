export * as SessionCompaction from "./compaction.js"

import {
  AIError,
  InvalidProviderOutputError,
  UnknownProviderError,
  isContextOverflowFailure,
  LLMClient,
  LLMEvent,
  LLMRequest,
  Message,
  type ContentPart,
  type ToolResultPart,
  type Usage,
} from "@opencode/ai"
import type { StreamOptions } from "@opencode/ai/route"
import type { SessionCompactionResult } from "@opencode/plugin/effect/session"
import { SessionError } from "@opencode/schema/session-error"
import { Context, Effect, Layer, Ref, Stream } from "effect"
import { Bus } from "../bus.js"
import { Database } from "../database/database.js"
import { makeLocationNode } from "@opencode/util/effect/app-node"
import { llmClient } from "../effect/app-node-platform.js"
import { SessionEvent } from "./event.js"
import type { SessionContext } from "./context.js"
import { SessionHistory } from "./history.js"
import type { SessionMessage } from "./message.js"
import { SessionModelRequest } from "./model-request.js"
import { SessionProviderContext } from "./provider-context.js"
import type { SessionRunnerModel } from "./runner/model.js"
import { SessionRunnerRetry } from "./runner/retry.js"
import { SessionSchema } from "./schema.js"
import { toSessionError } from "./to-session-error.js"
import { Token } from "../util/token.js"
import { SessionUsage } from "./usage.js"
import { State } from "../state.js"
import { toLLMMessages } from "./runner/to-llm-message.js"
import type { AgentNotFoundError } from "./error.js"
import type { Instructions } from "../instructions/index.js"

const DEFAULT_BUFFER = 20_000
const DEFAULT_KEEP_TOKENS = 15_000
const OUTPUT_TOKEN_MAX = 32_000
const TOOL_OUTPUT_MAX_CHARS = 2_000
const FALLBACK_TOOL_CHARS = 1_000
const FALLBACK_OVERFLOW_RETRIES = 3
const IMAGE_TOKEN_ESTIMATE = 1_500
const PDF_TOKEN_ESTIMATE = 2_000

export type Settings = {
  auto: boolean
  buffer: number
  tokens: number
}

export type NativeInput = {
  readonly request: LLMRequest
  readonly options: StreamOptions
  /** Whole, real user messages within the retained-token allowance, for checkpoint-only mechanisms. */
  readonly retained: Effect.Effect<ReadonlyArray<Message>>
}

export type NativeResult = {
  readonly replacement: ReadonlyArray<Message>
  readonly usage?: Usage
}

/** Returns the provider's replacement window, or `undefined` when this strategy has no mechanism for the route. */
export type NativeStrategy = (input: NativeInput) => Effect.Effect<NativeResult, AIError> | undefined

export type Editor = {
  configure: (settings: Partial<Settings>) => void
  /** Later registrations take precedence. */
  native: (strategy: NativeStrategy) => void
}

export type AutoInput = {
  readonly context: SessionContext.Loaded
  readonly prepare: SessionModelRequest.Interface["compaction"]
  /** Known overflow must recover from durable history, not submit the overflowing native window again. */
  readonly overflow?: boolean
}

type RequiredInput = {
  readonly messages: readonly SessionMessage.Info[]
  readonly context: SessionContext.Loaded
}

export type ManualInput = {
  readonly session: SessionSchema.Info
  readonly messages: readonly SessionMessage.Info[]
  readonly inputID: SessionMessage.ID
  readonly started?: boolean
  /** Empty compaction controls do not preflight model or instruction availability. */
  readonly resolveContext: (
    session: SessionSchema.Info,
  ) => Effect.Effect<
    SessionContext.Loaded & { readonly instructionUpdate: string },
    SessionRunnerModel.Error | AgentNotFoundError | Instructions.InitializationBlocked
  >
  readonly prepare: SessionModelRequest.Interface["compaction"]
}

type ExecuteInput = AutoInput & {
  readonly reason: SessionMessage.Compaction["reason"]
  readonly inputID?: SessionMessage.ID
  readonly started?: boolean
  readonly instructionUpdate?: string
}

export type Outcome =
  | (Pick<SessionMessage.CompactionCompleted, "status"> & {
      /** Consumes the logical step's one overflow rebuild even when the native attempt overflowed first. */
      readonly recoveredOverflow?: boolean
    })
  | Pick<SessionMessage.CompactionFailed, "status" | "error">

export interface Interface extends State.Transformable<Editor> {
  readonly enabled: () => boolean
  readonly required: (input: RequiredInput) => boolean
  readonly compact: (input: AutoInput) => Effect.Effect<Outcome>
  readonly compactManual: (input: ManualInput) => Effect.Effect<Outcome>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/SessionCompaction") {}

// Summary prompt

const SUMMARY_TEMPLATE = `You MUST use this format for your response (you may omit sections that aren't applicable). Do not include the <template> tags in your response.
<template>
## Objective
- [one or two brief sentences describing what the user is trying to accomplish]

## Requirements
- [constraints, preferences, requirements, and scope boundaries stated by the user, or "(none)"]

## Decisions
- [decisions already made and why, or "(none)"]

## Work State
Break the objective into smaller goals and report which are completed, which are being worked on, and which are blocked.
### Completed
- [goals that have been completed; otherwise "(none)"]

### Active
- [goals currently being worked on; otherwise "(none)"]

### Blocked
- [anything blocking progress, and why; otherwise "(none)"]

## Next Move
1. [ordered list of next actions, or "(none)"]

## Relevant Files
List the files and directories, other than the current working directory, that another agent would need to open to continue this work. Include at most 15, most important first. Do not list every file that was read or changed. Include paths outside the current working directory when relevant. If none, write "(none)".
- \`[file or directory path]\`: [brief reason it matters]

## Important Context
- [facts the next agent cannot continue without and cannot easily find on its own; or "(none)"]
</template>`

const SUMMARY_RULES = `Rules:
- Keep each section concise. Use terse, single-line bullets, not prose paragraphs or nested lists.
- Prefer short references over detailed restatement. It is fine to leave out information the next agent can recover from the code or the files listed above.
- Preserve exact file paths, symbols, commands, error strings, URLs, and identifiers.
- Carry forward only user questions or requests that remain unanswered or require further action. Do not repeat ones that newer history has answered or resolved. Preserve exact wording when carrying one forward.
- Preserve consequential workflow state, including whether changes are uncommitted, committed, pushed, under review, or merged.
- Do not mention the summary process or that context was compacted.`

export const buildPrompt = (update: boolean, legacy = false) => {
  const shared = [
    "Summarize only what the user and the assistant said and did. Leave out instructions and setup the assistant was given rather than told by the user: repository conventions, instruction files such as AGENTS.md, and environment details like the session ID. The next agent receives current versions of all of these separately.",
    SUMMARY_TEMPLATE,
    SUMMARY_RULES,
    "Do not continue the task or call tools.",
    "Return only the structured summary in the requested format. Do not include a preamble, explanation, or other commentary.",
  ]
  if (!update) {
    return [
      "You MUST summarize the conversation above into a structured summary that will be given to another agent to resume the work.",
      ...shared,
    ].join("\n\n")
  }
  return [
    "Update the existing checkpoint in the conversation above into one consolidated summary.",
    ...(legacy
      ? [
          "The existing checkpoint was written with an earlier format that recorded far more detail than this one asks for. Rewrite it at the level of detail described below rather than carrying its detail forward. Keep its requirements, decisions, and open questions; they came from earlier conversation with the user.",
        ]
      : []),
    "Newer history always takes precedence over the existing checkpoint. Preserve previous information unless newer history clearly contradicts, supersedes, resolves, or makes it stale. If something is no longer relevant to continuing the work, you may remove it.",
    "Incorporate newer requirements, decisions, progress, and context. Reconcile Work State and Next Move: move completed work out of Active, remove resolved blockers and answered questions, and preserve unresolved or pending work.",
    "Return only the updated Markdown sections. Do not reproduce the `<conversation-checkpoint>`, `<summary>`, or `<recent-context>` wrapper tags from the previous checkpoint.",
    ...shared,
  ].join("\n\n")
}

const NUDGE =
  "The previous response did not fill in the required summary template. Do not call tools. Return the summary as text using the exact section headings from the template."

/** Checkpoints written with the previous template carry this catch-all heading. */
const LEGACY_HEADING = "## Additional Context"

const SUMMARY_HEADINGS = SUMMARY_TEMPLATE.split("\n").filter((line) => line.startsWith("##"))

const hasSummarySection = (text: string) => text.split("\n").some((line) => SUMMARY_HEADINGS.includes(line.trim()))

// Message text

/** The text form of one message in the retained tail. Empty for messages the tail does not show. */
const serializeRecentMessage = (message: SessionMessage.Info): string => {
  switch (message.type) {
    // Checkpoints and instruction updates are handled outside the serialized tail.
    case "compaction":
    case "system":
      return ""

    case "user": {
      const skills =
        message.skills?.flatMap((skill) =>
          skill.text === undefined ? [] : [`[Skill activated: ${skill.name}]\n${skill.text}`],
        ) ?? []
      const files =
        message.files?.map((file) => {
          const name = file.name ?? (file.source.type === "uri" ? file.source.uri : "inline attachment")
          return `[Attached ${file.mime}: ${name}]`
        }) ?? []
      return [...skills, `[User]: ${message.text}`, ...files].join("\n")
    }

    case "location-switched":
      return `[User]: The working directory has been changed to ${message.location.directory}.`

    case "assistant":
      return message.content
        .flatMap((part) => {
          if (part.type === "text") return [`[Assistant]: ${part.text}`]
          if (part.type === "reasoning") return part.text ? [`[Assistant reasoning]: ${part.text}`] : []

          const input = typeof part.state.input === "string" ? part.state.input : JSON.stringify(part.state.input)
          const call = `[Assistant tool call]: ${part.name}(${input})`
          if (part.state.status === "completed") {
            return [call, `[Tool result]: ${truncateToolOutput(serializeToolContent(part.state.content))}`]
          }
          if (part.state.status === "error") return [call, `[Tool error]: ${part.state.error.message}`]
          return [call]
        })
        .join("\n")

    case "synthetic":
      return `[Synthetic context]: ${message.text}`

    case "skill":
      return `[Skill activated: ${message.name}]\n${message.text}`

    case "shell":
      if (message.metadata?.background === true) return ""
      return `[Shell]: ${message.command}\n${truncateToolOutput(message.output?.output ?? "")}`

    default:
      return ""
  }
}

export const truncateToolOutput = (value: string, maxChars = TOOL_OUTPUT_MAX_CHARS) => {
  if (value.length <= maxChars) return value

  // Count code points so a surrogate pair is never split.
  let end = 0
  let kept = 0
  for (const char of value) {
    if (kept === maxChars) break
    end += char.length
    kept++
  }
  if (end === value.length) return value
  return `${value.slice(0, end)}\n[truncated]`
}

export const serializeToolContent = (content: ReadonlyArray<SessionMessage.ToolStateCompleted["content"][number]>) =>
  content
    .map((item) => {
      if (item.type === "text") return item.text
      return `[Attached ${item.mime}${item.name === undefined ? "" : `: ${item.name}`}]`
    })
    .join("\n")

const toolResultText = (result: ToolResultPart["result"]) => {
  if (result.type === "content") return serializeToolContent(result.value)
  if (typeof result.value === "string") return result.value
  return JSON.stringify(result.value) ?? ""
}

// Prompt size

/** The base request without any conversation, as the runner would send it. */
const transcript = (context: SessionContext.Loaded, messages: readonly SessionMessage.Info[]) =>
  SessionModelRequest.baseTranscript({
    agent: context.agent.info,
    model: context.model,
    tools: context.tools,
    initial: context.initial,
    messages,
  })

/** The token size of the whole prompt these messages would produce. */
export const estimateTokens = (input: RequiredInput) => {
  const model = input.context.model
  const anchorIndex = input.messages.findLastIndex(hasInputUsage)
  const anchor = input.messages[anchorIndex]

  // Everything from the anchor onward is unmeasured, apart from the anchor's own provider-bound parts.
  // Its local tool results are kept: the provider never saw them.
  const unmeasured = SessionModelRequest.unsupportedParts(
    toLLMMessages(input.messages.slice(Math.max(0, anchorIndex)), model.ref),
    model.capabilities,
  ).filter((message) => message.role !== "assistant" || message.id !== anchor?.id)
  const added = unmeasured.reduce((sum, message) => sum + estimateMessage(message), 0)

  if (anchor?.type === "assistant" && anchor.tokens) {
    const tokens = anchor.tokens
    return added + tokens.input + tokens.cache.read + tokens.cache.write + tokens.output + tokens.reasoning
  }

  // Without a measured response, only the fixed prompt is known to be there.
  return added + estimateFixed(transcript(input.context, []).system, input.context.tools)
}

/** The newest such message anchors the estimate: its provider-reported usage covers everything before it. */
const hasInputUsage = (message: SessionMessage.Info) =>
  message.type === "assistant" &&
  !message.error &&
  message.tokens !== undefined &&
  message.tokens.input + message.tokens.cache.read + message.tokens.cache.write > 0

/** The prompt size that starts automatic compaction and bounds a reduced summary request. */
const promptCeiling = (limit: SessionRunnerModel.Resolved["limit"], buffer: number) => {
  const outputReserve = Math.min(limit.output, OUTPUT_TOKEN_MAX)
  const contextCeiling = limit.context - Math.max(outputReserve, buffer)
  if (limit.input === undefined) return contextCeiling
  return Math.min(contextCeiling, limit.input - buffer)
}

/** System prompt and tool definitions: sent with every request but outside the message history. */
const estimateFixed = (system: ReadonlyArray<{ readonly text: string }>, tools: SessionContext.Loaded["tools"]) => {
  const systemTokens = system.reduce((sum, part) => sum + Token.estimate(part.text), 0)
  const toolTokens = tools.definitions.reduce(
    (sum, tool) => sum + Token.estimate(tool.name + tool.description + JSON.stringify(tool.inputSchema)),
    0,
  )
  return systemTokens + toolTokens
}

const estimateMessage = (message: Message) => message.content.reduce((sum, part) => sum + estimatePart(part), 0)

const estimatePart = (part: ContentPart): number => {
  // Encrypted checkpoints have no locally measurable token size.
  if (part.type === "compaction") return Token.estimate(part.text ?? "")
  if (part.type === "effort") return 0
  if (part.type === "text" || part.type === "reasoning") return Token.estimate(part.text)
  if (part.type === "media") return estimateMedia(part.media.mediaType)
  if (part.type === "tool-call") return Token.estimate(part.name + (JSON.stringify(part.input) ?? ""))

  // Tool results: media inside a result counts as media, not as its placeholder text.
  if (part.result.type === "content") {
    return part.result.value.reduce((sum, content) => {
      if (content.type === "text") return sum + Token.estimate(content.text)
      return sum + estimateMedia(content.mime)
    }, 0)
  }
  return Token.estimate(toolResultText(part.result))
}

const estimateMedia = (mime: string) => {
  const type = mime.toLowerCase()
  if (type.startsWith("image/")) return IMAGE_TOKEN_ESTIMATE
  if (type === "application/pdf") return PDF_TOKEN_ESTIMATE
  return 0
}

// History selection

/**
 * Split the history into the older prefix to summarize and the newest exchanges kept as text beside the summary.
 * Undefined when there is no conversation to compact.
 */
const splitHistory = (messages: readonly SessionMessage.Info[], keepTokens: number) => {
  const entries = messages.flatMap((message, index) => {
    const text = serializeRecentMessage(message)
    return text ? [{ message, text, index }] : []
  })
  if (entries.length === 0) return undefined

  const tail = entries.slice(findTailStart(entries, keepTokens, lastCheckpoint(messages)))
  const tailStart = tail[0]?.index ?? messages.length
  return {
    messages: messages.slice(0, tailStart),
    recent: tail.map((entry) => entry.text).join("\n\n"),
  }
}

/** Index into `entries` where the retained tail begins. */
const findTailStart = (
  entries: ReadonlyArray<{ readonly message: SessionMessage.Info; readonly text: string }>,
  keepTokens: number,
  previous: SessionMessage.CompactionCompleted | undefined,
) => {
  // Keep at least the newest entry, even if it exceeds the allowance.
  const fitted = Math.min(
    fitNewest(entries, (entry) => Token.estimate(entry.text), keepTokens),
    entries.length - 1,
  )

  // Start at a user boundary so an assistant's tool calls and results stay together.
  const boundary = entries.findLastIndex((entry, index) => index <= fitted && entry.message.type === "user")
  if (boundary > 0) return boundary

  // If everything fits, retain only the latest exchange to leave an older prefix to summarize.
  const latestUser = entries.findLastIndex((entry) => entry.message.type === "user")
  if (latestUser > 0) return latestUser

  // Without an older retained tail to summarize, summarize everything and retain nothing.
  return previous?.recent ? 0 : entries.length
}

/** Index of the oldest item in the newest run whose sizes total at most `budget`. */
const fitNewest = <T>(items: readonly T[], size: (item: T) => number, budget: number) => {
  let total = 0
  let start = items.length
  while (start > 0) {
    const next = total + size(items[start - 1])
    if (next > budget) break
    total = next
    start--
  }
  return start
}

const lastCheckpoint = (messages: readonly SessionMessage.Info[]) =>
  messages.findLast(
    (message): message is SessionMessage.CompactionCompleted =>
      message.type === "compaction" && message.status === "completed",
  )

/** Keep whole, real user messages, never synthetic guidance or half an attachment/tool exchange. */
export const retainUsers = (
  messages: readonly SessionMessage.Info[],
  model: Pick<SessionRunnerModel.Resolved, "ref" | "capabilities">,
  keepTokens: number,
) => {
  const users = messages
    .filter((message) => message.type === "user")
    .map((message) => ({ ...message, skills: undefined }))
  const bound = SessionModelRequest.boundImages(
    SessionModelRequest.unsupportedParts(toLLMMessages(users, model.ref), model.capabilities),
  )
  return bound.slice(fitNewest(bound, estimateMessage, keepTokens))
}

// Overflow fallback

/** Flatten provider-bound history into text so a reduced summary request carries no tool pairs or reasoning signatures. */
const serializeFallback = (messages: readonly Message[]) =>
  messages.flatMap((message) => {
    const parts = message.content.flatMap((part) => {
      if (part.type === "text" || part.type === "reasoning") return part.text ? [part.text] : []
      if (part.type === "compaction") return part.text ? [part.text] : []
      if (part.type === "media") {
        const name = part.filename ? `: ${part.filename}` : ""
        return [`[Attached ${part.media.mediaType}${name}; content omitted]`]
      }
      if (part.type === "tool-call") return [`[Tool call ${part.name}(${JSON.stringify(part.input)})]`]
      if (part.type === "tool-result") {
        const output = truncateToolOutput(toolResultText(part.result), FALLBACK_TOOL_CHARS)
        return [`[Tool result ${part.name}]: ${output}`]
      }
      return []
    })
    if (parts.length === 0) return []
    return [{ role: message.role, text: `[${message.role}]: ${parts.join("\n")}` }]
  })

/** Retain a prior checkpoint and the newest complete exchanges that fit the summary input budget. */
const fitFallback = (entries: ReturnType<typeof serializeFallback>, budget: number) => {
  // A prior checkpoint leads the history and is always kept.
  const checkpoint = entries[0]?.text.includes("<conversation-checkpoint>") ? entries[0].text : undefined
  const header = checkpoint ? `${checkpoint}\n\n` : ""

  // Each user message and the work that followed it form one exchange, dropped whole or not at all.
  const exchanges = entries.slice(checkpoint ? 1 : 0).reduce<string[]>((groups, entry) => {
    if (entry.role === "user" || groups.length === 0) groups.push(entry.text)
    else groups[groups.length - 1] += `\n\n${entry.text}`
    return groups
  }, [])

  const allowance = budget - Token.estimate(header)
  if (allowance <= 0) return undefined
  const omitted = fitNewest(exchanges, Token.estimate, allowance)
  if (exchanges.length > 0 && omitted === exchanges.length) return undefined

  const note = omitted ? `[${omitted} older exchanges omitted from this summary input]\n\n` : ""
  return { text: header + note + exchanges.slice(omitted).join("\n\n"), omitted }
}

// Service

type Envelope = Pick<SessionEvent.Compaction.Failed["data"], "sessionID" | "reason" | "inputID">

/** One summary attempt: text so far plus the first failure, if any, and whether it was a context overflow. */
type Summary = {
  readonly text: string
  readonly overflow: boolean
  readonly failure?: SessionError.Error
  readonly providerState?: SessionMessage.ProviderState
}

export const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const bus = yield* Bus.Service
    const llm = yield* LLMClient.Service
    const db = (yield* Database.Service).db

    const state = State.create<Settings & { readonly native: NativeStrategy[] }, Editor>({
      name: "session-compaction",
      initial: () => ({ auto: true, buffer: DEFAULT_BUFFER, tokens: DEFAULT_KEEP_TOKENS, native: [] }),
      editor: (editor) => ({
        configure: (settings) => {
          if (settings.auto !== undefined) editor.auto = settings.auto
          if (settings.buffer !== undefined) editor.buffer = settings.buffer
          if (settings.tokens !== undefined) editor.tokens = settings.tokens
        },
        native: (strategy) => {
          editor.native.push(strategy)
        },
      }),
    })

    // Events

    const envelope = (input: ExecuteInput): Envelope => ({
      sessionID: input.context.session.id,
      reason: input.reason,
      inputID: input.inputID,
    })

    const recordUsage = (sessionID: SessionSchema.ID, usage: SessionUsage.Recorded | undefined) => {
      if (!usage) return Effect.void
      return bus.publish(SessionEvent.UsageRecorded, { sessionID, source: "compaction", ...usage })
    }

    const failed = Effect.fnUntraced(function* (
      target: Envelope,
      error: SessionError.Error,
      usage?: SessionUsage.Recorded,
    ) {
      yield* recordUsage(target.sessionID, usage)
      yield* bus.publish(SessionEvent.Compaction.Failed, { ...target, error, ...usage })
      return { status: "failed" as const, error }
    })

    const ended = Effect.fnUntraced(function* (
      input: ExecuteInput,
      result: {
        readonly text: string
        readonly recent: string
        readonly providerState?: SessionMessage.ProviderState
        readonly providerContext?: SessionProviderContext.Info
        readonly usage?: SessionUsage.Recorded
        readonly metadata?: Record<string, unknown>
      },
    ) {
      const context = input.context
      yield* recordUsage(context.session.id, result.usage)
      yield* bus.publish(
        SessionEvent.Compaction.Ended,
        {
          sessionID: context.session.id,
          reason: input.reason,
          model: context.model.ref,
          providerState: result.providerState,
          providerContext: result.providerContext,
          text: result.text,
          recent: result.recent,
          ...result.usage,
        },
        { metadata: result.metadata },
      )
      return { status: "completed" as const }
    })

    const started = (input: ExecuteInput, recent: string) => {
      if (input.started) return Effect.void
      return bus.publish(SessionEvent.Compaction.Started, { ...envelope(input), recent })
    }

    /** A hook answered the request itself. */
    const supplied = (input: ExecuteInput, result: SessionCompactionResult, recent: string) =>
      ended(input, {
        text: result.summary,
        recent,
        providerState: result.providerState,
        usage: result.tokens && {
          tokens: result.tokens,
          cost: SessionUsage.calculateCost(input.context.model.cost, result.tokens),
        },
        metadata: result.metadata,
      })

    // Manual controls settle through the inbox; only automatic work needs a durable interruption record.
    const interrupted = Effect.fnUntraced(function* (input: ExecuteInput, usage?: SessionUsage.Recorded) {
      yield* recordUsage(input.context.session.id, usage)
      if (input.reason !== "auto") return
      yield* failed(envelope(input), { type: "compaction.interrupted", message: "Compaction was interrupted" })
    })

    // Requests

    const prepare = (
      input: ExecuteInput,
      request: Pick<SessionModelRequest.Input, "system" | "messages">,
      webSocket?: "session",
    ) =>
      input.prepare({
        session: input.context.session,
        agent: input.context.agent.id,
        model: input.context.model,
        tools: input.context.tools,
        system: request.system,
        messages: request.messages,
        webSocket,
      })

    /** The conversation as the runner would send it, plus any pending instruction update. */
    const compactionRequest = (
      input: ExecuteInput,
      messages: readonly SessionMessage.Info[],
      webSocket?: "session",
    ) => {
      const base = transcript(input.context, messages)
      const update = input.instructionUpdate ? [Message.system(input.instructionUpdate)] : []
      return prepare(input, { system: base.system, messages: [...base.messages, ...update] }, webSocket)
    }

    const retry = Effect.fnUntraced(function* (input: ExecuteInput, hook: SessionModelRequest.Prepared["retry"]) {
      const policy = yield* SessionRunnerRetry.policy(input.context.session.id)
      return SessionRunnerRetry.transient(policy, {
        agent: input.context.agent.id,
        model: input.context.model.ref,
        hook,
      })
    })

    // Local summary

    const summarize = Effect.fn("SessionCompaction.summarize")(function* (input: ExecuteInput) {
      const context = input.context
      const history = splitHistory(context.messages, state.get().tokens)
      if (!history) {
        return yield* failed(envelope(input), { type: "compaction.unavailable", message: "Nothing to compact yet" })
      }
      yield* started(input, history.recent)

      const previous = lastCheckpoint(history.messages)
      const prompt = buildPrompt(previous !== undefined, previous?.summary.includes(LEGACY_HEADING) ?? false)
      const prepared = yield* compactionRequest(input, history.messages)
      if (prepared.event.result) return yield* supplied(input, prepared.event.result, history.recent)

      // Every attempt shares one retry allowance and one usage total, so an interruption can still account for it.
      const transient = yield* retry(input, prepared.retry)
      const usage = yield* Ref.make<SessionUsage.Recorded | undefined>(undefined)

      // Hooks see the transcript alone; the summary prompt is appended after they run.
      const generate = Effect.fnUntraced(function* (request: LLMRequest, options: StreamOptions) {
        const prompted = LLMRequest.update(request, { messages: [...request.messages, Message.user(prompt)] })
        const first = yield* stream(input, prompted, options, transient, usage)
        if (first.failure || hasSummarySection(first.text)) return first

        const nudged = LLMRequest.update(prompted, { messages: [...prompted.messages, Message.user(NUDGE)] })
        return yield* stream(input, nudged, options, transient, usage)
      })

      const finish = Effect.fnUntraced(function* (result: Summary, omitted: number) {
        const total = yield* Ref.get(usage)
        if (result.failure || !hasSummarySection(result.text)) {
          const message = result.text.trim()
            ? "Compaction summary did not match the required template"
            : "Compaction produced no summary"
          return yield* failed(envelope(input), result.failure ?? { type: "compaction.failed", message }, total)
        }

        const note = omitted
          ? `\n\n[${omitted} older exchanges were omitted from the summary input; original session history is retained.]`
          : ""
        return yield* ended(input, {
          text: result.text + note,
          recent: history.recent,
          providerState: result.providerState,
          usage: total,
        })
      })

      // The room the conversation has once the fixed prompt and summary instructions are counted.
      const limit = context.model.limit
      const fixed = estimateFixed(prepared.request.system, context.tools) + Token.estimate(prompt)
      const budget = limit.context > 0 ? promptCeiling(limit, state.get().buffer) - fixed : undefined

      // A prefix that cannot fit the window at all skips the normal request and starts from the reduced text.
      const window = Math.min(limit.context, limit.input ?? Number.POSITIVE_INFINITY)
      const size = estimateTokens({ messages: history.messages, context }) + Token.estimate(prompt)
      const oversized = budget !== undefined && budget > 0 && size > window

      const normal = oversized ? undefined : yield* generate(prepared.request, prepared.options)
      if (normal && !normal.overflow) return yield* finish(normal, 0)

      // Flatten the history to text and drop the oldest exchanges until the provider accepts the request.
      const entries = serializeFallback(prepared.request.messages)
      const system = transcript(context, []).system
      let allowance = budget ?? Token.estimate(entries.map((entry) => entry.text).join("\n\n"))
      let previousText: string | undefined
      for (let attempt = 1; ; attempt++) {
        const fitted = fitFallback(entries, allowance)
        if (!fitted || fitted.text === previousText) {
          return yield* failed(
            envelope(input),
            {
              type: "compaction.failed",
              message: "The summary input cannot be reduced further without losing the latest exchange or checkpoint",
            },
            yield* Ref.get(usage),
          )
        }

        const reduced = yield* prepare(input, { system, messages: [Message.user(fitted.text)] })
        if (reduced.event.result) {
          yield* recordUsage(context.session.id, yield* Ref.get(usage))
          return yield* supplied(input, reduced.event.result, history.recent)
        }

        const result = yield* generate(reduced.request, reduced.options)
        if (!result.overflow || attempt === FALLBACK_OVERFLOW_RETRIES) return yield* finish(result, fitted.omitted)
        allowance = Math.floor(allowance / 2)
        previousText = fitted.text
      }
    })

    /** One summary request, folded into the text it produced. Usage accumulates in `usage` across attempts. */
    const stream = (
      input: ExecuteInput,
      request: LLMRequest,
      options: StreamOptions,
      transient: ReturnType<typeof SessionRunnerRetry.transient>,
      usage: Ref.Ref<SessionUsage.Recorded | undefined>,
    ) => {
      const context = input.context
      const metadataKey = context.model.model.route.providerMetadataKey ?? context.model.model.provider

      const step = (summary: Summary, event: LLMEvent): Effect.Effect<Summary, AIError> => {
        if (LLMEvent.is.providerError(event)) {
          const overflow = event.classification === "context-overflow"
          const type = overflow ? "provider.invalid-request" : "provider.error"
          return Effect.succeed({ ...summary, overflow, failure: { type, message: event.message } })
        }

        if (LLMEvent.is.textDelta(event)) {
          return bus
            .publish(SessionEvent.Compaction.Delta, { sessionID: context.session.id, text: event.text })
            .pipe(Effect.as({ ...summary, text: summary.text + event.text }))
        }

        if (LLMEvent.is.stepFinish(event)) {
          return Ref.update(usage, (total) => {
            const recorded = SessionUsage.record(event.usage, context.model.cost)
            return total ? SessionUsage.add(total, recorded) : recorded
          }).pipe(Effect.as({ ...summary, providerState: event.providerMetadata?.[metadataKey] }))
        }

        if (!LLMEvent.is.finish(event)) return Effect.succeed(summary)
        switch (event.reason.normalized) {
          case "unknown":
            return Effect.fail(
              new AIError({
                reason: new InvalidProviderOutputError({
                  message: "The provider response ended with an unknown finish reason.",
                  classification: "incomplete-stream",
                }),
              }),
            )
          case "error":
            return Effect.fail(
              new AIError({ reason: new UnknownProviderError({ message: "Compaction generation failed" }) }),
            )
          case "length":
            return Effect.succeed({
              ...summary,
              failure: { type: "compaction.failed", message: "Compaction summary reached the output token limit" },
            })
          case "content-filter":
            return Effect.succeed({
              ...summary,
              failure: { type: "provider.content-filter", message: "Compaction summary was blocked by the provider" },
            })
          default:
            return Effect.succeed(summary)
        }
      }

      return llm.stream(request, options).pipe(
        Stream.runFoldEffect((): Summary => ({ text: "", overflow: false }), step),
        transient,
        Effect.catchTag("AI.Error", (error) =>
          Effect.succeed<Summary>({
            text: "",
            overflow: isContextOverflowFailure(error),
            failure: toSessionError(error),
          }),
        ),
        Effect.onInterrupt(() => Ref.get(usage).pipe(Effect.flatMap((total) => interrupted(input, total)))),
      )
    }

    // Provider checkpoint

    const native = Effect.fn("SessionCompaction.native")(function* (input: ExecuteInput) {
      const context = input.context
      const reject = (message: string) => failed(envelope(input), { type: "provider.unsupported-operation", message })

      const prepared = yield* compactionRequest(input, context.messages, "session")
      if (prepared.event.result) {
        yield* started(input, "")
        return yield* supplied(input, prepared.event.result, "")
      }
      const request = prepared.request

      const provenance = SessionProviderContext.provenance(context.model)
      if (!provenance) return yield* reject("Provider compaction requires a stable, configured endpoint")
      // History is selected before request hooks. Until that interface can select on the final route,
      // require routing in the catalog; never install a checkpoint that the next request would skip.
      const routed = SessionProviderContext.provenance({ model: request.model, ref: context.model.ref })
      if (!SessionProviderContext.compatible(provenance, routed)) {
        return yield* reject(
          "Provider compaction requires the endpoint in provider/model settings, not a model.request rewrite",
        )
      }

      const retained = original(context.session.id).pipe(
        Effect.map((messages) => retainUsers(messages, context.model, state.get().tokens)),
      )
      const strategy = state
        .get()
        .native.toReversed()
        .map((strategy) => strategy({ request, options: prepared.options, retained }))
        .find((effect) => effect !== undefined)
      if (!strategy) {
        return yield* reject(
          `No plugin provides native compaction for ${request.model.provider}/${request.model.route.id}`,
        )
      }

      const transient = yield* retry(input, prepared.retry)
      yield* started(input, "")

      // Transient provider failures retry like any other request; only a known automatic overflow permits
      // local recovery, and nothing is installed until the provider returns a checkpoint.
      const install = (result: NativeResult) =>
        ended(input, {
          text: "",
          recent: "",
          providerContext: SessionProviderContext.encode(provenance, result.replacement),
          usage: result.usage && SessionUsage.record(result.usage, context.model.cost),
        })
      const recover = (cause: AIError): Effect.Effect<Outcome> => {
        if (input.reason !== "auto" || !isContextOverflowFailure(cause))
          return failed(envelope(input), toSessionError(cause))
        return summarizeOriginal({ ...input, started: true }).pipe(
          Effect.map((result) => (result.status === "completed" ? { ...result, recoveredOverflow: true } : result)),
        )
      }
      return yield* Effect.uninterruptibleMask((restore) =>
        restore(strategy.pipe(transient)).pipe(Effect.flatMap(install)),
      ).pipe(
        Effect.onInterrupt(() => interrupted(input)),
        Effect.catchTag("AI.Error", recover),
      )
    })

    /** The durable transcript since the last local summary, re-expanding every native window. */
    const original = (sessionID: SessionSchema.ID) => SessionHistory.load(db, sessionID, "local").pipe(Effect.orDie)

    /** Summarize from the original transcript, whatever the model's compaction setting: an overflowing window cannot be resubmitted. */
    const summarizeOriginal = (input: ExecuteInput) =>
      original(input.context.session.id).pipe(
        Effect.flatMap((messages) => summarize({ ...input, context: { ...input.context, messages } })),
      )

    // Entry points

    const run = (input: ExecuteInput) =>
      input.context.model.compaction?.type === "native" ? native(input) : summarize(input)

    const compact = Effect.fn("SessionCompaction.compact")(function* (input: AutoInput): Effect.fn.Return<Outcome> {
      const request = { ...input, reason: "auto" as const }
      return yield* input.overflow ? summarizeOriginal(request) : run(request)
    })

    const required = (input: RequiredInput) => {
      const config = state.get()
      if (!config.auto) return false

      // Run the completed checkpoint before considering another automatic compaction.
      const last = input.messages.at(-1)
      if (last?.type === "compaction" && last.status === "completed") return false

      // Native usage describes the compaction operation, not the replacement's size. Wait for
      // a primary response to anchor the new window, including after restart or new admission.
      const anchorIndex = input.messages.findLastIndex(hasInputUsage)
      const checkpointIndex = input.messages.findLastIndex(SessionProviderContext.isCheckpoint)
      if (anchorIndex < checkpointIndex) return false

      const limit = input.context.model.limit
      if (limit.context <= 0) return false
      return estimateTokens(input) >= promptCeiling(limit, config.buffer)
    }

    const compactManual = Effect.fn("SessionCompaction.compactManual")(function* (input: ManualInput) {
      const target: Envelope = { sessionID: input.session.id, reason: "manual", inputID: input.inputID }
      if (splitHistory(input.messages, state.get().tokens) === undefined) {
        return yield* failed(target, { type: "compaction.unavailable", message: "Nothing to compact yet" })
      }
      return yield* input.resolveContext(input.session).pipe(
        Effect.matchEffect({
          onFailure: (cause) => failed(target, toSessionError(cause)),
          onSuccess: (context) =>
            run({
              context,
              instructionUpdate: context.instructionUpdate,
              prepare: input.prepare,
              reason: "manual",
              inputID: input.inputID,
              started: input.started,
            }),
        }),
      )
    })

    return Service.of({
      transform: state.transform,
      reload: state.reload,
      enabled: () => state.get().auto,
      required,
      compact,
      compactManual,
    })
  }),
)

export const node = makeLocationNode({
  service: Service,
  layer,
  deps: [Bus.node, Database.node, llmClient],
})
