import { expect, test } from "bun:test"
import type { IntegrationInfo } from "@opencode/client"
import { loginOptions } from "../src/commands/handlers/auth/login"

const integration = (value: Partial<IntegrationInfo> & Pick<IntegrationInfo, "id" | "name">): IntegrationInfo => ({
  methods: [{ type: "key" }],
  connections: [],
  ...value,
})

test("distinguishes MCP servers from AI providers while retaining login targets and status", () => {
  expect(
    loginOptions([
      integration({ id: "openai", name: "OpenAI" }),
      integration({ id: "linear", name: "Linear", metadata: { source: "mcp" } }),
      integration({ id: "opencode", name: "OpenCode Zen" }),
      integration({ id: "opencode-go", name: "OpenCode Go", connections: [{ type: "env", name: "GO_KEY" }] }),
      integration({ id: "unused", name: "Unused", methods: [{ type: "env", names: ["UNUSED_KEY"] }] }),
    ]),
  ).toEqual([
    { value: "linear", label: "MCP server · Linear", hint: "linear" },
    { value: "opencode-go", label: "AI provider · OpenCode Go", hint: "connected" },
    { value: "opencode", label: "AI provider · OpenCode Zen", hint: "recommended" },
    { value: "openai", label: "AI provider · OpenAI", hint: "openai" },
  ])
})

test("sorts MCP servers by name ahead of providers regardless of ID", () => {
  expect(
    loginOptions([
      integration({ id: "openai", name: "OpenAI" }),
      integration({ id: "zulu", name: "Zulu", metadata: { source: "mcp" } }),
      integration({ id: "alpha", name: "Alpha", metadata: { source: "mcp" } }),
    ]).map((option) => option.value),
  ).toEqual(["alpha", "zulu", "openai"])
})
