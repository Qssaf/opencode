/** @jsxImportSource @opentui/solid */
import {
  BoxRenderable,
  DiffRenderable,
  LineNumberRenderable,
  type ColorInput,
  type ScrollBoxRenderable,
} from "@opentui/core"
import type { JSX } from "@opentui/solid"
import { useRenderer } from "@opentui/solid"
import { createEffect, createMemo, createSignal, For, onCleanup, Show, splitProps } from "solid-js"
import { splitAddedPatch, splitPatchHunks, type AddedPatchChunk } from "../util/diff"
import { stringWidth } from "../util/string-width"

export interface PatchDiffRef {
  readonly hunks: () => readonly (DiffRenderable | BoxRenderable)[]
}

const VIRTUAL_CHUNK_LINES = 128

type Props = Omit<JSX.IntrinsicElements["diff"], "diff" | "lineNumberBg" | "ref"> & {
  diff: string
  hunkFg: ColorInput
  lineNumberBg: ColorInput
  ref?: (value: PatchDiffRef) => void
  virtualScroll?: () => ScrollBoxRenderable | undefined
  viewportWidth?: number
}

export function PatchDiff(props: Props) {
  const [local, diffProps] = splitProps(props, [
    "diff",
    "hunkFg",
    "lineNumberBg",
    "ref",
    "virtualScroll",
    "viewportWidth",
  ])
  const hunks = createMemo(() => splitPatchHunks(local.diff))
  const chunks = createMemo(() => local.virtualScroll && splitAddedPatch(local.diff, VIRTUAL_CHUNK_LINES))
  const nodes = new Map<number, DiffRenderable>()
  let virtualRoot: BoxRenderable | undefined
  local.ref?.({
    hunks: () => {
      if (chunks()) return virtualRoot && !virtualRoot.isDestroyed ? [virtualRoot] : []
      return [...nodes.entries()]
        .sort(([left], [right]) => left - right)
        .map(([, node]) => node)
        .filter((node) => !node.isDestroyed)
    },
  })
  const syncGutters = (attempt = 0) => {
    requestAnimationFrame(() => {
      const sides = [...nodes.values()]
        .filter((item) => !item.isDestroyed)
        .flatMap((item) => item.getChildren().filter((side) => side instanceof LineNumberRenderable))
      const lineNumbers = sides.map((side) => new Map([...side.getLineNumbers()].filter(([line]) => line >= 0)))
      const digits = lineNumbers.map((numbers) => Math.max(0, ...numbers.values()).toString().length)
      const after = sides.map((side) =>
        Math.max(
          0,
          ...[...side.getLineSigns()].filter(([line]) => line >= 0).map(([, sign]) => stringWidth(sign.after ?? "")),
        ),
      )
      const maxDigits = Math.max(...digits)
      const maxAfter = Math.max(...after)
      if (!maxDigits && attempt < 2) return syncGutters(attempt + 1)
      if (!maxDigits) return
      sides.forEach((side) => {
        const index = sides.indexOf(side)
        const signs = new Map([...side.getLineSigns()].filter(([line]) => line >= 0))
        signs.set(-1, { after: " ".repeat(maxAfter + maxDigits - digits[index]) })
        side.setLineNumbers(lineNumbers[index])
        side.setLineSigns(signs)
      })
    })
  }
  const register = (index: number, node: DiffRenderable) => {
    nodes.set(index, node)
    onCleanup(() => nodes.delete(index))
    syncGutters()
  }

  return (
    <Show
      when={chunks()}
      fallback={
        <For each={hunks()}>
          {(hunk, index) => (
            <>
              <Show when={index() > 0}>
                <box width="100%" height={1} backgroundColor={local.lineNumberBg}>
                  <text fg={local.hunkFg} bg={local.lineNumberBg}>
                    {` ${hunk.header ?? ""}`}
                  </text>
                </box>
              </Show>
              <diff
                {...diffProps}
                ref={(node: DiffRenderable) => register(index(), node)}
                diff={hunk.patch}
                minHeight={hunk.rows}
                lineNumberBg={local.lineNumberBg}
              />
            </>
          )}
        </For>
      }
    >
      {(items) => (
        <VirtualAddedPatch
          chunks={items()}
          width={local.viewportWidth ?? 80}
          scroll={local.virtualScroll!}
          diffProps={diffProps}
          lineNumberBg={local.lineNumberBg}
          register={register}
          registerRoot={(root) => (virtualRoot = root)}
        />
      )}
    </Show>
  )
}

function VirtualAddedPatch(props: {
  chunks: readonly AddedPatchChunk[]
  width: number
  scroll: () => ScrollBoxRenderable | undefined
  diffProps: Omit<JSX.IntrinsicElements["diff"], "diff" | "lineNumberBg" | "ref">
  lineNumberBg: ColorInput
  register: (index: number, node: DiffRenderable) => void
  registerRoot: (root: BoxRenderable) => void
}) {
  const renderer = useRenderer()
  const [visible, setVisible] = createSignal(0)
  const [measured, setMeasured] = createSignal<ReadonlyMap<number, number>>(new Map())
  createEffect(() => {
    props.width
    props.chunks
    setMeasured(new Map())
  })
  // Offscreen chunks need heights for scroll jumps before OpenTUI has measured them.
  // Replace those estimates with actual rendered heights as chunks enter the viewport.
  const estimates = createMemo(() => {
    const codeWidth = Math.max(
      1,
      props.width - String(props.chunks.reduce((count, chunk) => count + chunk.rows, 0)).length - 5,
    )
    return props.chunks.map((chunk) =>
      chunk.lines.reduce((height, line) => height + Math.max(1, Math.ceil(stringWidth(line.slice(1)) / codeWidth)), 0),
    )
  })
  const heights = createMemo(() => estimates().map((estimate, index) => measured().get(index) ?? estimate))

  return (
    <box
      width="100%"
      ref={(root: BoxRenderable) => {
        props.registerRoot(root)
        root.onLifecyclePass = () => {
          const scroll = props.scroll()
          if (!scroll) return
          // ScrollBox's scroll position is not a Solid signal; observe it during the render pass.
          const offset = root.y - scroll.content.y
          const top = scroll.scrollTop - offset
          const sizes = heights()
          if (top + scroll.viewport.height < 0 || top > sizes.reduce((sum, height) => sum + height, 0)) {
            setVisible(-1)
            return
          }
          let position = 0
          const index = sizes.findIndex((height) => (position += height) > top)
          setVisible(index < 0 ? sizes.length - 1 : index)
        }
        renderer.registerLifecyclePass(root)
        onCleanup(() => renderer.unregisterLifecyclePass(root))
      }}
    >
      <For each={props.chunks}>
        {(chunk, index) => (
          <Show
            when={visible() >= 0 && Math.abs(index() - visible()) <= 2}
            fallback={<box height={heights()[index()]} />}
          >
            <diff
              {...props.diffProps}
              ref={(node: DiffRenderable) => {
                props.register(index(), node)
                node.onSizeChange = () => {
                  if (node.height <= 0 || measured().get(index()) === node.height) return
                  const scroll = props.scroll()
                  const atEnd = scroll && scroll.scrollTop >= scroll.scrollHeight - scroll.viewport.height - 1
                  setMeasured((known) => new Map(known).set(index(), node.height))
                  // Keep G pinned to the end when a newly mounted chunk changes total height.
                  if (atEnd) requestAnimationFrame(() => scroll.scrollTo(Infinity))
                }
              }}
              diff={chunk.patch}
              lineNumberBg={props.lineNumberBg}
            />
          </Show>
        )}
      </For>
    </box>
  )
}
