export * as SessionSubagent from "./subagent.js"

import { Effect } from "effect"
import { Agent } from "../agent.js"
import { Instance } from "../instance/service.js"
import type { Model } from "../model.js"
import { Plugin } from "../plugin/service.js"
import type { Session } from "../session.js"
import { AgentNotFoundError } from "./error.js"
import type { SessionSchema } from "./schema.js"
import type { SubagentJob } from "./subagent-job.js"

const preamble = "You are a subagent spawned by another session."

export type Input = {
  readonly sessionID: SessionSchema.ID
  readonly text: string
  readonly description: string
  readonly agent?: Agent.ID
  readonly model?: Model.Ref
  /** Copies the parent's settled history into the child instead of starting with fresh context. */
  readonly fork?: boolean
  /** False admits the completion notice to the parent without resuming it. */
  readonly resume?: boolean
}

/** Starts a background child Session whose outcome is delivered to the parent when it settles. */
export const spawn = Effect.fn("SessionSubagent.spawn")(function* (
  sessions: Session.Interface,
  subagents: SubagentJob.Runner,
  input: Input,
) {
  const instances = yield* Instance.Service
  const parent = yield* sessions.get(input.sessionID)
  const selected = yield* Plugin.awaitActivation.pipe(
    Effect.andThen(Agent.Service),
    Effect.flatMap((agents) => agents.select(input.agent ?? parent.agent)),
    instances.provide(parent),
  )
  if (input.agent !== undefined && selected.info === undefined)
    return yield* new AgentNotFoundError({ sessionID: parent.id, agent: input.agent })

  const create = sessions.create({
    parentID: parent.id,
    title: input.description,
    agent: selected.id,
    model: input.model ?? selected.info?.model ?? parent.model,
  })
  const child = input.fork
    ? yield* sessions.fork({ sessionID: parent.id, child: true }).pipe(
        Effect.tap((forked) => {
          // A fork inherits the parent's agent and model; an explicit model wins over the switched agent's model.
          const switched = forked.agent !== selected.id
          const model = input.model ?? (switched ? selected.info?.model : undefined)
          return Effect.all(
            [
              sessions.rename({ sessionID: forked.id, title: input.description }),
              switched ? sessions.switchAgent({ sessionID: forked.id, agent: selected.id }) : Effect.void,
              model === undefined ? Effect.void : sessions.switchModel({ sessionID: forked.id, model }),
            ],
            { discard: true },
          )
        }),
        Effect.catchTag("Session.ForkEmptyError", () => create),
        // Only a `before` boundary can be missing.
        Effect.catchTag("Session.MessageNotFoundError", Effect.die),
      )
    : yield* create

  // A text-only prompt without an explicit ID cannot fail admission on the Session just created.
  yield* sessions
    .prompt({ sessionID: child.id, text: [preamble, input.text].join("\n"), resume: false })
    .pipe(Effect.orDie)
  const recovery = {
    kind: "subagent" as const,
    parentSessionID: parent.id,
    childSessionID: child.id,
    agent: selected.id,
    description: input.description,
    ...(input.resume === false ? { resume: false } : {}),
  }
  yield* subagents.start(recovery)
  yield* subagents.background(recovery)
  return yield* sessions.get(child.id)
})
