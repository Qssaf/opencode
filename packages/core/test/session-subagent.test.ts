import { describe, expect } from "bun:test"
import path from "path"
import { Effect, Layer, Stream } from "effect"
import { LanguageModel } from "@opencode/ai"
import { OpenAIChat } from "@opencode/ai/protocols/openai-chat"
import { TestLLM } from "@opencode/ai/testing"
import { Agent } from "@opencode/core/agent"
import { Bus } from "@opencode/core/bus"
import { AppNodeBuilder } from "@opencode/core/effect/app-node-builder"
import { LayerNodePlatform } from "@opencode/core/effect/app-node-platform"
import { Watcher } from "@opencode/core/filesystem/watcher"
import { LocationServiceMap } from "@opencode/core/location-service-map"
import { Model } from "@opencode/core/model"
import { Provider } from "@opencode/core/provider"
import { AbsolutePath } from "@opencode/core/schema"
import { Session } from "@opencode/core/session"
import { SessionRunnerModel } from "@opencode/core/session/runner/model"
import { LayerNode } from "@opencode/util/effect/layer-node"
import { Global } from "@opencode/util/global"
import { tempGlobalLayer } from "./fixture/global"
import { offlineModels } from "./fixture/models"
import { tmpdirScoped } from "./fixture/tmpdir"
import { testEffect } from "./lib/effect"

const llmLayer = TestLLM.testLayer({ fallback: TestLLM.text("Docs updated", "docs") })
const it = testEffect(
  Layer.merge(
    llmLayer,
    AppNodeBuilder.build(LayerNode.group([Session.node, LocationServiceMap.node]), [
      Global.node.replace(tempGlobalLayer),
      offlineModels,
      Watcher.node.replace(Watcher.configured({ enabled: false })),
      LayerNodePlatform.llmClient.replace(llmLayer),
      SessionRunnerModel.node.replace(
        Layer.succeed(SessionRunnerModel.Service, {
          resolve: (session) =>
            Effect.succeed(
              SessionRunnerModel.resolved(
                LanguageModel.make({ id: session.model?.id ?? "parent", provider: "test", route: OpenAIChat.route }),
                {
                  capabilities: { tools: true, input: ["text"], output: ["text"] },
                  cost: [],
                  limit: { context: 200_000, output: 32_000 },
                },
              ),
            ),
        }),
      ),
    ]),
  ),
)

const parentModel = Model.Ref.make({ id: Model.ID.make("parent"), providerID: Provider.ID.make("test") })

describe("session subagents", () => {
  it.live("forks settled history into a child and admits its outcome without resuming the parent", () =>
    Effect.gen(function* () {
      const parent = yield* project()
      const sessions = yield* Session.Service
      const llm = yield* TestLLM.Test
      yield* sessions.prompt({ sessionID: parent.id, text: "Earlier question" })
      yield* sessions.wait(parent.id)
      const gate = yield* llm.gate()

      // This must return while the child's model is still blocked.
      const child = yield* sessions.subagent({
        sessionID: parent.id,
        text: "Update the docs",
        description: "Update docs",
        fork: true,
        resume: false,
      })
      yield* gate.started
      expect(child).toMatchObject({
        parentID: parent.id,
        fork: { sessionID: parent.id },
        title: "Update docs",
        agent: "build",
        model: parentModel,
      })
      expect((yield* sessions.list({ parentID: parent.id })).data.map((session) => session.id)).toEqual([child.id])
      expect(
        (yield* sessions.context(child.id)).flatMap((message) => (message.type === "user" ? [message.text] : [])),
      ).toEqual(["Earlier question", "You are a subagent spawned by another session.\nUpdate the docs"])

      yield* gate.release
      const notice = yield* sessions.log({ sessionID: parent.id, follow: true }).pipe(
        Stream.filter(
          (event) =>
            !Bus.isSynced(event) && event.type === "session.inbox.enqueued" && event.data.item.type === "synthetic",
        ),
        Stream.runHead,
      )
      expect(notice).toMatchObject({
        _tag: "Some",
        value: { data: { item: { metadata: { source: "subagent", childID: child.id, state: "completed" } } } },
      })
      expect(yield* sessions.inbox(parent.id)).toMatchObject([{ type: "synthetic" }])
      expect((yield* sessions.context(parent.id)).filter((message) => message.type === "synthetic")).toEqual([])
      expect(yield* llm.requests()).toHaveLength(2)
    }),
  )

  it.live("creates a fresh child with the requested agent when there is no history to fork", () =>
    Effect.gen(function* () {
      const parent = yield* project()
      const sessions = yield* Session.Service
      const llm = yield* TestLLM.Test
      const gate = yield* llm.gate()

      const child = yield* sessions.subagent({
        sessionID: parent.id,
        text: "Review the changes",
        description: "Review changes",
        agent: Agent.ID.make("reviewer"),
        fork: true,
      })
      yield* gate.started
      expect(child).toMatchObject({ parentID: parent.id, agent: "reviewer", model: { id: "child" } })
      expect(child.fork).toBeUndefined()
      yield* gate.release

      expect(
        yield* sessions
          .subagent({ sessionID: parent.id, text: "x", description: "x", agent: Agent.ID.make("missing") })
          .pipe(Effect.flip),
      ).toBeInstanceOf(Session.AgentNotFoundError)
    }),
  )
})

function project() {
  return Effect.gen(function* () {
    const tmp = yield* tmpdirScoped()
    yield* Effect.promise(() =>
      Bun.write(
        path.join(tmp.path, "opencode.json"),
        JSON.stringify({ agents: { reviewer: { mode: "subagent", model: "test/child" } } }),
      ),
    )
    const sessions = yield* Session.Service
    return yield* sessions.create({
      location: { directory: AbsolutePath.make(tmp.path) },
      title: "Parent session",
      agent: Agent.ID.make("build"),
      model: parentModel,
    })
  })
}
