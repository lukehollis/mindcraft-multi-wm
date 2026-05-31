'use client'

import type { Dispatch, ReactNode, SetStateAction } from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Badge } from '@/components/ui/8bit/badge'
import { Progress } from '@/components/ui/8bit/progress'
import { ScrollArea } from '@/components/ui/8bit/scroll-area'
import type { ActivityEvent, AgentState, BlockSample, DashboardSnapshot, Vec3 } from '@/lib/types'

const BLOCK_COLORS: Record<string, string> = {
  wood: '#00f5a0',
  stone: '#0d6f80',
  iron: '#d6fbff',
  coal: '#0b2026',
  diamond: '#00e5ff',
  water: '#0077ff',
  lava: '#ff355e',
  station: '#f8ff6a',
  other: '#07343d'
}

const AGENT_COLORS = ['#00f5ff', '#00ffa3', '#f8ff6a', '#ff2bd6', '#8b5cff', '#ff7a18']
const BRIDGE_HTTP_ENV = process.env.NEXT_PUBLIC_BRIDGE_HTTP
const BRIDGE_WS_ENV = process.env.NEXT_PUBLIC_BRIDGE_WS
const MAX_INLINE_VIEWERS = 9
const MAX_ACTIVE_WEBGL_VIEWERS = 9
const MAX_OVERVIEW_WEBGL_AGENTS = configInt(process.env.NEXT_PUBLIC_MAX_OVERVIEW_WEBGL_AGENTS, 64, 0)
const CLIENT_COMMIT_INTERVAL_MS = configInt(process.env.NEXT_PUBLIC_CLIENT_COMMIT_INTERVAL_MS, 750, 100)
const CLIENT_BLOCK_LIMIT = configInt(process.env.NEXT_PUBLIC_CLIENT_BLOCK_LIMIT, 64)
const CLIENT_HIDDEN_BLOCK_LIMIT = configInt(process.env.NEXT_PUBLIC_CLIENT_HIDDEN_BLOCK_LIMIT, 8)
const CLIENT_ENTITY_LIMIT = configInt(process.env.NEXT_PUBLIC_CLIENT_ENTITY_LIMIT, 8)
const CLIENT_TRAIL_LIMIT = configInt(process.env.NEXT_PUBLIC_CLIENT_TRAIL_LIMIT, 72)
const CLIENT_EDGE_LIMIT = configInt(process.env.NEXT_PUBLIC_CLIENT_EDGE_LIMIT, 160)
const CLIENT_ACTIVITY_LIMIT = configInt(process.env.NEXT_PUBLIC_CLIENT_ACTIVITY_LIMIT, 80)
const VIEWER_IFRAME_FPS = configNumber(process.env.NEXT_PUBLIC_VIEWER_FPS, 4, 0)
const VIEWER_IFRAME_DPR = configNumber(process.env.NEXT_PUBLIC_VIEWER_DPR, 1, 0.1)

type DashboardHistoryPoint = {
  ts: number
  nearestDistance: number
  inventoryValue: number
  craftProgress: number
  successRate: number
  actionDiversity: number
  meanDurationMs: number
}

type CraftNode = {
  id: string
  label: string
  aliases: string[]
  deps: string[]
  tier: number
  lane: number
}

type CraftUnlock = {
  firstSeenAt: number
  lastSeenAt: number
  count: number
  agents: string[]
}

type CraftObservation = {
  seen: boolean
  active: boolean
  count: number
  agents: string[]
}

const CRAFTING_TREE: CraftNode[] = [
  { id: 'logs', label: 'logs', aliases: ['log'], deps: [], tier: 0, lane: 1 },
  { id: 'planks', label: 'planks', aliases: ['planks'], deps: ['logs'], tier: 1, lane: 1 },
  { id: 'table', label: 'table', aliases: ['crafting_table'], deps: ['planks'], tier: 2, lane: 0 },
  { id: 'sticks', label: 'sticks', aliases: ['stick'], deps: ['planks'], tier: 2, lane: 2 },
  { id: 'wood_pick', label: 'wood pick', aliases: ['wooden_pickaxe'], deps: ['table', 'sticks', 'planks'], tier: 3, lane: 1 },
  { id: 'cobble', label: 'cobble', aliases: ['cobblestone'], deps: ['wood_pick'], tier: 4, lane: 1 },
  { id: 'stone_pick', label: 'stone pick', aliases: ['stone_pickaxe'], deps: ['cobble', 'sticks', 'table'], tier: 5, lane: 0 },
  { id: 'furnace', label: 'furnace', aliases: ['furnace'], deps: ['cobble'], tier: 5, lane: 2 },
  { id: 'coal', label: 'coal', aliases: ['coal', 'charcoal'], deps: ['wood_pick'], tier: 6, lane: 0 },
  { id: 'raw_iron', label: 'raw iron', aliases: ['raw_iron', 'iron_ore'], deps: ['stone_pick'], tier: 6, lane: 1 },
  { id: 'iron_ingot', label: 'iron ingot', aliases: ['iron_ingot'], deps: ['raw_iron', 'furnace', 'coal'], tier: 7, lane: 1 },
  { id: 'iron_pick', label: 'iron pick', aliases: ['iron_pickaxe'], deps: ['iron_ingot', 'sticks', 'table'], tier: 8, lane: 1 },
  { id: 'diamond', label: 'diamond', aliases: ['diamond'], deps: ['iron_pick'], tier: 9, lane: 1 }
]

export function MindcraftDashboard() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null)
  const [history, setHistory] = useState<DashboardHistoryPoint[]>([])
  const [connected, setConnected] = useState(false)
  const [browserHost, setBrowserHost] = useState('localhost')
  const [webglAvailable, setWebglAvailable] = useState(true)
  const [overflowOpen, setOverflowOpen] = useState(false)
  const [activeWebglLimit, setActiveWebglLimit] = useState(MAX_ACTIVE_WEBGL_VIEWERS)
  const [craftUnlocks, setCraftUnlocks] = useState<Record<string, CraftUnlock>>({})

  useEffect(() => {
    let cancelled = false
    let ws: WebSocket | null = null
    let reconnectTimer: number | null = null
    let commitTimer: number | null = null
    let pollTimer: number | null = null
    let pendingSnapshot: DashboardSnapshot | null = null
    let lastCommitAt = 0
    const endpoints = resolveBridgeEndpoints()
    setBrowserHost(window.location.hostname || 'localhost')
    setWebglAvailable(hasWebglSupport())
    setActiveWebglLimit(readWebglLimit())

    const flushSnapshot = () => {
      commitTimer = null
      if (cancelled || !pendingSnapshot) return
      commitSnapshot(pendingSnapshot, setSnapshot, setHistory, setCraftUnlocks)
      pendingSnapshot = null
      lastCommitAt = window.performance.now()
    }

    const enqueueSnapshot = (next: DashboardSnapshot) => {
      pendingSnapshot = next
      const delay = Math.max(0, CLIENT_COMMIT_INTERVAL_MS - (window.performance.now() - lastCommitAt))
      if (delay === 0) {
        if (commitTimer) {
          window.clearTimeout(commitTimer)
          commitTimer = null
        }
        flushSnapshot()
        return
      }
      if (!commitTimer) commitTimer = window.setTimeout(flushSnapshot, delay)
    }

    const pollSnapshot = async () => {
      try {
        const response = await fetch(`${endpoints.http}/snapshot`, { cache: 'no-store' })
        if (!response.ok || cancelled) throw new Error('snapshot unavailable')
        const next = (await response.json()) as DashboardSnapshot
        enqueueSnapshot(next)
        setConnected(true)
      } catch {
        if (!cancelled) setConnected(false)
      }
    }

    const schedulePolling = (delay = 0) => {
      if (pollTimer) window.clearTimeout(pollTimer)
      pollTimer = window.setTimeout(async () => {
        pollTimer = null
        if (cancelled) return
        const websocketOpen = ws?.readyState === WebSocket.OPEN
        if (!websocketOpen) await pollSnapshot()
        schedulePolling(websocketOpen ? 5000 : 1500)
      }, delay)
    }

    const connect = async () => {
      await pollSnapshot()
      if (!endpoints.ws) {
        schedulePolling(1500)
        return
      }
      if (cancelled) return
      ws = new WebSocket(endpoints.ws)
      ws.onopen = () => {
        setConnected(true)
        schedulePolling(5000)
      }
      ws.onmessage = (event) => {
        try {
          const next = JSON.parse(event.data) as DashboardSnapshot
          enqueueSnapshot(next)
        } catch {
          // Skip malformed telemetry frames so a single oversized/bad payload cannot kill the UI.
        }
      }
      ws.onerror = () => setConnected(false)
      ws.onclose = () => {
        setConnected(false)
        if (!cancelled) {
          schedulePolling(0)
          reconnectTimer = window.setTimeout(connect, 1500)
        }
      }
    }

    connect()
    return () => {
      cancelled = true
      if (reconnectTimer) window.clearTimeout(reconnectTimer)
      if (commitTimer) window.clearTimeout(commitTimer)
      if (pollTimer) window.clearTimeout(pollTimer)
      if (ws) ws.close()
    }
  }, [])

  const readyAgents = snapshot?.agents.filter((agent) => agent.ready).length ?? 0
  const minDistance = snapshot ? minAgentDistance(snapshot) : null
  const renderedFeeds = snapshot?.agents.filter((agent) => agent.viewer).length ?? 0
  const activeTasks = snapshot?.agents.filter((agent) => agent.active).length ?? 0
  const overviewViewer = snapshot?.server.overview_viewer ?? null
  const inlineViewerSlots = Math.min(renderedFeeds, Math.max(0, MAX_INLINE_VIEWERS))
  const activeWebglSlots = Math.min(inlineViewerSlots, Math.max(0, activeWebglLimit))
  const primaryAgents = snapshot?.agents.slice(0, MAX_INLINE_VIEWERS) ?? []
  const overflowAgents = snapshot?.agents.slice(MAX_INLINE_VIEWERS) ?? []
  const overviewWebglAvailable = webglAvailable && (snapshot?.agents.length ?? 0) <= MAX_OVERVIEW_WEBGL_AGENTS
  const feedMetrics: [string, string | number][] = [
    ['status', connected ? 'live' : 'reconnecting'],
    ['viewer slots', `${inlineViewerSlots}/${renderedFeeds}`],
    ['webgl', `${activeWebglSlots}/${inlineViewerSlots}`],
    ['render', `${VIEWER_IFRAME_DPR}x @ ${VIEWER_IFRAME_FPS}fps`],
    ['active tasks', activeTasks],
    ['agents', `${readyAgents}/${snapshot?.agents.length ?? 0}`],
    ['hidden', overflowAgents.length]
  ]
  if (snapshot) {
    feedMetrics.splice(1, 0, ['minecraft', snapshot.server.version ?? 'auto'])
    feedMetrics.splice(2, 0, ['bridge', snapshot.server.bridge_port])
  }

  return (
    <main className="dashboard-shell">
      <section className="panel feeds-panel">
        <PanelHead
          title="Agent Feeds"
          connected={connected}
          metrics={feedMetrics}
        />
        <div className="feed-grid">
          {snapshot ? primaryAgents.map((agent, index) => (
            <AgentFeed
              key={agent.name}
              agent={agent}
              color={agentColor(index)}
              browserHost={browserHost}
              webglAvailable={webglAvailable}
              viewerEnabled={index < activeWebglSlots}
            />
          )) : <EmptyState label="Waiting for agents" />}
        </div>
        {overflowAgents.length > 0 ? (
          <section className="agent-overflow">
            <button
              className="agent-overflow-trigger"
              type="button"
              aria-expanded={overflowOpen}
              onClick={() => setOverflowOpen((open) => !open)}
            >
              <span>{overflowOpen ? 'hide' : 'show'} overflow agents</span>
              <span>{overflowAgents.length} hidden</span>
            </button>
            {overflowOpen ? (
              <div className="feed-grid overflow-feed-grid">
                {overflowAgents.map((agent, index) => (
                  <AgentFeed
                    key={agent.name}
                    agent={agent}
                    color={agentColor(index + MAX_INLINE_VIEWERS)}
                    browserHost={browserHost}
                    webglAvailable={webglAvailable}
                    viewerEnabled={false}
                  />
                ))}
              </div>
            ) : null}
          </section>
        ) : null}
      </section>

      <section className="panel overview-panel">
        <PanelHead
          title="Society View"
          metrics={[
            ['ready', `${readyAgents}/${snapshot?.agents.length ?? 0}`],
            ['nearest', minDistance === null ? '-' : `${fmt(minDistance)}m`],
            ['world camera', overviewViewer && overviewWebglAvailable ? `:${overviewViewer.port}` : 'canvas'],
            ['events', snapshot?.activity.length ?? 0]
          ]}
        />
        <div className="overview-grid">
          <WorldCamera
            viewer={overviewViewer}
            browserHost={browserHost}
            webglAvailable={overviewWebglAvailable}
            snapshot={snapshot}
          />
          <div className="society-stage">
            <span className="stage-label">Society Map</span>
            <SocietyCanvas snapshot={snapshot} />
            <MapLegend />
          </div>
          <ActivityPanel events={snapshot?.activity ?? []} />
        </div>
      </section>

      <section className="graphs-grid">
        <GraphCard title="Coordination">
          <DistanceGraph history={history} />
        </GraphCard>
        <GraphCard title="Inventory Progress">
          <InventoryGraph history={history} />
        </GraphCard>
        <GraphCard title="Action Mix">
          <ActionGraph snapshot={snapshot} />
        </GraphCard>
        <GraphCard title="Crafting Tree" className="graph-card-wide">
          <CraftingTreeGraph snapshot={snapshot} unlocks={craftUnlocks} />
        </GraphCard>
        <GraphCard title="Learning Signals">
          <LearningGraph history={history} />
        </GraphCard>
        <GraphCard title="Capability Matrix" className="graph-card-full">
          <CapabilityGraph snapshot={snapshot} />
        </GraphCard>
      </section>
    </main>
  )
}

function commitSnapshot(
  next: DashboardSnapshot,
  setSnapshot: (snapshot: DashboardSnapshot) => void,
  setHistory: Dispatch<SetStateAction<DashboardHistoryPoint[]>>,
  setCraftUnlocks: Dispatch<SetStateAction<Record<string, CraftUnlock>>>
) {
  const compact = compactSnapshot(next)
  const metrics = learningMetrics(compact)
  setSnapshot(compact)
  setCraftUnlocks((items) => mergeCraftUnlocks(items, compact))
  setHistory((items) => [
    ...items.slice(-239),
    {
      ts: compact.ts,
      nearestDistance: minAgentDistance(compact) ?? 0,
      inventoryValue: inventoryValue(compact),
      craftProgress: metrics.craftProgress,
      successRate: metrics.successRate,
      actionDiversity: metrics.actionDiversity,
      meanDurationMs: metrics.meanDurationMs
    }
  ])
}

function compactSnapshot(snapshot: DashboardSnapshot): DashboardSnapshot {
  return {
    ...snapshot,
    agents: snapshot.agents.map((agent, index) => {
      const blockLimit = index < MAX_INLINE_VIEWERS ? CLIENT_BLOCK_LIMIT : CLIENT_HIDDEN_BLOCK_LIMIT
      return {
        ...agent,
        blocks: (agent.blocks ?? []).slice(0, blockLimit),
        entities: (agent.entities ?? []).slice(0, CLIENT_ENTITY_LIMIT),
        trail: (agent.trail ?? []).slice(-CLIENT_TRAIL_LIMIT)
      }
    }),
    society: {
      ...snapshot.society,
      edges: (snapshot.society.edges ?? []).slice(0, CLIENT_EDGE_LIMIT)
    },
    activity: (snapshot.activity ?? []).slice(-CLIENT_ACTIVITY_LIMIT)
  }
}

function PanelHead({
  title,
  metrics,
  connected
}: {
  title: string
  metrics: [string, string | number][]
  connected?: boolean
}) {
  return (
    <div className="panel-head">
      <h2>{title}</h2>
      <div className="metric-row">
        {typeof connected === 'boolean' ? <span className={`dot ${connected ? 'ready' : ''}`} /> : null}
        {metrics.map(([label, value]) => (
          <Badge className="pill metric-badge" key={label} variant="outline">
            {label}: {value}
          </Badge>
        ))}
      </div>
    </div>
  )
}

function AgentFeed({
  agent,
  color,
  browserHost,
  webglAvailable,
  viewerEnabled
}: {
  agent: AgentState
  color: string
  browserHost: string
  webglAvailable: boolean
  viewerEnabled: boolean
}) {
  const [visible, setVisible] = useState(false)
  const cardRef = useRef<HTMLElement | null>(null)
  const viewerUrl = webglAvailable && viewerEnabled && visible && agent.viewer
    ? viewerHref(browserHost, agent.viewer.port, agent.viewer.path)
    : null

  useEffect(() => {
    const el = cardRef.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { rootMargin: '260px 0px' }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  return (
    <article className="feed-card" ref={cardRef}>
      <div className="feed-head">
        <div className="feed-title">
          <strong>{agent.name}</strong>
          <span>{agent.position ? `x ${fmt(agent.position.x)} y ${fmt(agent.position.y)} z ${fmt(agent.position.z)}` : 'offline'}</span>
        </div>
        <Badge className="pill task-badge" variant={agent.active ? 'secondary' : 'outline'}>
          {agent.active?.tool ?? 'idle'}
        </Badge>
      </div>
      <div className="feed-stage">
        {viewerUrl ? (
          <iframe title={`${agent.name} rendered feed`} src={viewerUrl} />
        ) : (
          <FallbackFeed agent={agent} color={color} />
        )}
      </div>
      <div className="feed-data">
        <div>yaw {fmt(radToDeg(agent.yaw ?? 0), 0)} | sight {agent.line_of_sight ?? '-'}</div>
        <div>health {fmt(agent.health, 0)} | food {fmt(agent.food, 0)} | held {agent.equipped ?? '-'}</div>
        <div className="feed-meter">
          <span>health</span>
          <Progress value={normalizePercent(agent.health, 20)} variant="retro" progressBg="bg-[#00ffa3]" className="pixel-progress h-3" />
        </div>
        <div className="feed-meter">
          <span>food</span>
          <Progress value={normalizePercent(agent.food, 20)} variant="retro" progressBg="bg-[#f8ff6a]" className="pixel-progress h-3" />
        </div>
        <div className="inventory">
          {Object.entries(agent.inventory ?? {})
            .sort((a, b) => b[1] - a[1])
            .slice(0, 14)
            .map(([name, count]) => (
              <Badge className="item" key={name} variant="secondary">
                {name} x{count}
              </Badge>
            ))}
        </div>
      </div>
    </article>
  )
}

function WorldCamera({
  viewer,
  browserHost,
  webglAvailable,
  snapshot
}: {
  viewer: DashboardSnapshot['server']['overview_viewer'] | null | undefined
  browserHost: string
  webglAvailable: boolean
  snapshot: DashboardSnapshot | null
}) {
  const url = webglAvailable && viewer ? viewerHref(browserHost, viewer.port, '/') : null
  return (
    <div className="world-stage">
      <span className="stage-label">World Camera</span>
      {url ? (
        <iframe title="Minecraft overview world camera" src={url} />
      ) : (
        <OverviewFallback
          snapshot={snapshot}
          label={snapshot ? 'canvas overview' : 'waiting for overview data'}
        />
      )}
    </div>
  )
}

function OverviewFallback({ label, snapshot }: { label: string; snapshot: DashboardSnapshot | null }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = fitCanvas(canvas)
    drawWorldOverview(ctx, canvas, snapshot, label)
  }, [label, snapshot])
  return <canvas ref={canvasRef} />
}

function FallbackFeed({ agent, color }: { agent: AgentState; color: string }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = fitCanvas(canvas)
    drawFallbackFeed(ctx, canvas, agent, color)
  }, [agent, color])
  return <canvas ref={canvasRef} />
}

function SocietyCanvas({ snapshot }: { snapshot: DashboardSnapshot | null }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = fitCanvas(canvas)
    drawSociety(ctx, canvas, snapshot)
  }, [snapshot])
  return <canvas ref={canvasRef} />
}

function ActivityPanel({ events }: { events: ActivityEvent[] }) {
  return (
    <aside className="activity-panel">
      <h2>Activity</h2>
      <ScrollArea className="activity-scroll">
        <ol>
          {events.slice(-40).reverse().map((event, index) => (
            <li className="event" key={`${event.ts}-${index}`}>
              <strong>{event.bot ?? 'system'}</strong> {event.tool ?? event.level ?? 'event'}
              <span>{event.success ? 'ok' : 'fail'} {event.duration_ms ?? 0}ms | {timeAgo(event.ts)} ago</span>
              <span>{event.item ?? event.block ?? event.message ?? summarizeResult(event.result)}</span>
            </li>
          ))}
        </ol>
      </ScrollArea>
    </aside>
  )
}

function MapLegend() {
  const items = [
    ['Agent', AGENT_COLORS[0]],
    ['Trail', AGENT_COLORS[1]],
    ['Wood/leaves', BLOCK_COLORS.wood],
    ['Stone/terrain', BLOCK_COLORS.stone],
    ['Iron', BLOCK_COLORS.iron],
    ['Coal', BLOCK_COLORS.coal],
    ['Diamond', BLOCK_COLORS.diamond],
    ['Water', BLOCK_COLORS.water],
    ['Lava', BLOCK_COLORS.lava],
    ['Workstation', BLOCK_COLORS.station],
    ['Other block', BLOCK_COLORS.other]
  ]
  return (
    <div className="map-legend">
      {items.map(([label, color]) => (
        <Badge className="legend-item" key={label} variant="outline">
          <span className="legend-swatch" style={{ background: color }} />
          {label}
        </Badge>
      ))}
    </div>
  )
}

function GraphCard({ title, children, className }: { title: string; children: ReactNode; className?: string }) {
  return (
    <section className={`graph-card ${className ?? ''}`}>
      <div className="graph-card-head">
        <h2>{title}</h2>
      </div>
      <div className="graph-card-content">{children}</div>
    </section>
  )
}

function DistanceGraph({ history }: { history: DashboardHistoryPoint[] }) {
  const values = useMemo(() => history.map((point) => point.nearestDistance), [history])
  return <LineGraph values={values} color="#00ffa3" label="nearest agent distance" />
}

function InventoryGraph({ history }: { history: DashboardHistoryPoint[] }) {
  const values = useMemo(() => history.map((point) => point.inventoryValue), [history])
  return <LineGraph values={values} color="#00e5ff" label="inventory value" />
}

function LineGraph({ values, color, label }: { values: number[]; color: string; label: string }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = fitCanvas(canvas)
    drawLineChart(ctx, canvas, values, color, label)
  }, [values, color, label])
  return <canvas ref={canvasRef} className="graph-canvas" />
}

function ActionGraph({ snapshot }: { snapshot: DashboardSnapshot | null }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = fitCanvas(canvas)
    drawActionGraph(ctx, canvas, snapshot)
  }, [snapshot])
  return <canvas ref={canvasRef} className="graph-canvas" />
}

function CraftingTreeGraph({
  snapshot,
  unlocks
}: {
  snapshot: DashboardSnapshot | null
  unlocks: Record<string, CraftUnlock>
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = fitCanvas(canvas)
    drawCraftingTree(ctx, canvas, snapshot, unlocks)
  }, [snapshot, unlocks])
  return <canvas ref={canvasRef} className="graph-canvas graph-canvas-wide" />
}

function LearningGraph({ history }: { history: DashboardHistoryPoint[] }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = fitCanvas(canvas)
    drawLearningSignals(ctx, canvas, history)
  }, [history])
  return <canvas ref={canvasRef} className="graph-canvas" />
}

function CapabilityGraph({ snapshot }: { snapshot: DashboardSnapshot | null }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = fitCanvas(canvas)
    drawCapabilityMatrix(ctx, canvas, snapshot)
  }, [snapshot])
  return <canvas ref={canvasRef} className="graph-canvas" />
}

function EmptyState({ label }: { label: string }) {
  return <div className="empty-state">{label}</div>
}

function drawFallbackFeed(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, agent: AgentState, color: string) {
  const { width, height } = canvas
  ctx.clearRect(0, 0, width, height)
  ctx.fillStyle = '#000407'
  ctx.fillRect(0, 0, width, height)
  ctx.fillStyle = '#00151b'
  ctx.fillRect(0, 0, width, height * 0.48)
  ctx.fillStyle = '#001008'
  ctx.fillRect(0, height * 0.48, width, height * 0.52)
  drawGrid(ctx, width, height, 'rgba(0, 229, 255, 0.2)')
  if (!agent.ready || !agent.position) {
    centerText(ctx, width, height, 'offline', '#00e5ff')
    return
  }
  const cx = width / 2
  const horizon = height * 0.52
  const yaw = agent.yaw ?? 0
  for (const block of (agent.blocks ?? []).slice(0, 180)) {
    const relX = block.x - agent.position.x
    const relZ = block.z - agent.position.z
    const forward = relX * -Math.sin(yaw) + relZ * -Math.cos(yaw)
    const side = relX * Math.cos(yaw) - relZ * Math.sin(yaw)
    if (forward < -2 || Math.abs(side) > 18) continue
    const depth = Math.max(2, forward + 12)
    const x = cx + (side / depth) * width * 0.9
    const y = horizon + ((agent.position.y - block.y) / depth) * height * 0.7
    const size = Math.max(3, 34 / depth)
    ctx.fillStyle = BLOCK_COLORS[block.kind] || BLOCK_COLORS.other
    ctx.globalAlpha = 0.75
    ctx.fillRect(x - size / 2, y - size / 2, size, size)
  }
  ctx.globalAlpha = 1
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(cx - 18, horizon)
  ctx.lineTo(cx + 18, horizon)
  ctx.moveTo(cx, horizon - 18)
  ctx.lineTo(cx, horizon + 18)
  ctx.stroke()
}

function drawWorldOverview(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  snapshot: DashboardSnapshot | null,
  label: string
) {
  const { width, height } = canvas
  ctx.clearRect(0, 0, width, height)
  ctx.fillStyle = '#000407'
  ctx.fillRect(0, 0, width, height)
  drawGrid(ctx, width, height, 'rgba(0, 229, 255, 0.16)')
  if (!snapshot) {
    centerText(ctx, width, height, label, '#00e5ff')
    return
  }

  const bounds = worldBounds(snapshot)
  const toCanvas = (point: Vec3) => mapPoint(point, bounds, width, height)
  const blocks = snapshot.agents.flatMap((agent) => agent.blocks ?? [])
  for (const block of blocks.slice(0, 700)) {
    const p = toCanvas(block)
    const size = block.kind === 'other' ? 2 : 3.5
    ctx.fillStyle = BLOCK_COLORS[block.kind] || BLOCK_COLORS.other
    ctx.globalAlpha = block.kind === 'other' ? 0.12 : 0.46
    ctx.fillRect(p.x - size / 2, p.y - size / 2, size, size)
  }
  ctx.globalAlpha = 1

  for (const [index, agent] of snapshot.agents.entries()) {
    const color = agentColor(index)
    const trail = agent.trail ?? []
    if (trail.length > 1) {
      ctx.strokeStyle = color
      ctx.globalAlpha = 0.42
      ctx.lineWidth = 1.5 * dpr()
      ctx.beginPath()
      for (const [i, point] of trail.entries()) {
        const p = toCanvas(point)
        if (i === 0) ctx.moveTo(p.x, p.y)
        else ctx.lineTo(p.x, p.y)
      }
      ctx.stroke()
      ctx.globalAlpha = 1
    }
    if (!agent.position) continue
    const p = toCanvas(agent.position)
    ctx.fillStyle = color
    ctx.strokeStyle = '#000407'
    ctx.lineWidth = 2 * dpr()
    ctx.beginPath()
    ctx.arc(p.x, p.y, 6 * dpr(), 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
    ctx.strokeStyle = color
    ctx.lineWidth = 1.5 * dpr()
    ctx.beginPath()
    ctx.moveTo(p.x, p.y)
    ctx.lineTo(p.x - Math.sin(agent.yaw ?? 0) * 15 * dpr(), p.y - Math.cos(agent.yaw ?? 0) * 15 * dpr())
    ctx.stroke()
  }

  ctx.fillStyle = 'rgba(0, 4, 7, 0.7)'
  ctx.fillRect(10 * dpr(), canvas.height - 30 * dpr(), 150 * dpr(), 20 * dpr())
  ctx.fillStyle = '#00e5ff'
  ctx.font = `${10 * dpr()}px system-ui, sans-serif`
  ctx.fillText(label, 18 * dpr(), canvas.height - 16 * dpr())
}

function drawSociety(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, snapshot: DashboardSnapshot | null) {
  const { width, height } = canvas
  ctx.clearRect(0, 0, width, height)
  ctx.fillStyle = '#000407'
  ctx.fillRect(0, 0, width, height)
  drawGrid(ctx, width, height, 'rgba(0, 229, 255, 0.16)')
  if (!snapshot) {
    centerText(ctx, width, height, 'connecting', '#00e5ff')
    return
  }
  const bounds = worldBounds(snapshot)
  const toCanvas = (point: Vec3) => mapPoint(point, bounds, width, height)
  const agentsByName = new Map(snapshot.agents.map((agent) => [agent.name, agent]))
  const allBlocks = snapshot.agents.flatMap((agent) => agent.blocks ?? [])
  for (const block of allBlocks.slice(0, 900)) {
    const p = toCanvas(block)
    ctx.fillStyle = BLOCK_COLORS[block.kind] || BLOCK_COLORS.other
    ctx.globalAlpha = block.kind === 'other' ? 0.16 : 0.55
    ctx.fillRect(p.x - 2, p.y - 2, 4, 4)
  }
  ctx.globalAlpha = 1

  for (const edge of snapshot.society.edges ?? []) {
    const a = agentsByName.get(edge.source)
    const b = agentsByName.get(edge.target)
    if (!a?.position || !b?.position) continue
    const pa = toCanvas(a.position)
    const pb = toCanvas(b.position)
    ctx.strokeStyle = edge.collaboration ? '#f8ff6a' : edge.distance !== null && edge.distance < 12 ? '#00ffa3' : '#07515f'
    ctx.lineWidth = edge.collaboration ? 3 : 1
    ctx.setLineDash(edge.collaboration ? [7, 5] : [])
    ctx.beginPath()
    ctx.moveTo(pa.x, pa.y)
    ctx.lineTo(pb.x, pb.y)
    ctx.stroke()
  }
  ctx.setLineDash([])

  for (const [index, agent] of snapshot.agents.entries()) {
    const color = agentColor(index)
    const trail = agent.trail ?? []
    if (trail.length > 1) {
      ctx.strokeStyle = color
      ctx.globalAlpha = 0.55
      ctx.lineWidth = 2
      ctx.beginPath()
      for (const [i, point] of trail.entries()) {
        const p = toCanvas(point)
        if (i === 0) ctx.moveTo(p.x, p.y)
        else ctx.lineTo(p.x, p.y)
      }
      ctx.stroke()
      ctx.globalAlpha = 1
    }
    if (agent.position) {
      const p = toCanvas(agent.position)
      drawAgentMarker(ctx, p.x, p.y, agent.yaw ?? 0, color, agent.name, agent.active?.tool)
    }
  }
}

function drawAgentMarker(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  yaw: number,
  color: string,
  name: string,
  task?: string
) {
  ctx.fillStyle = color
  ctx.strokeStyle = '#000407'
  ctx.lineWidth = 3
  ctx.beginPath()
  ctx.arc(x, y, 10 * dpr(), 0, Math.PI * 2)
  ctx.fill()
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(x, y)
  ctx.lineTo(x - Math.sin(yaw) * 22 * dpr(), y - Math.cos(yaw) * 22 * dpr())
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.stroke()
  const label = task ? `${name} | ${task}` : name
  ctx.font = `${12 * dpr()}px system-ui, sans-serif`
  const w = Math.min(260 * dpr(), ctx.measureText(label).width + 18 * dpr())
  ctx.fillStyle = 'rgba(0, 4, 7, 0.82)'
  ctx.fillRect(x + 13 * dpr(), y - 18 * dpr(), w, 30 * dpr())
  ctx.fillStyle = '#d6fbff'
  ctx.fillText(label, x + 22 * dpr(), y + 1 * dpr())
}

function drawLineChart(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, values: number[], color: string, label: string) {
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = '#000407'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  drawGrid(ctx, canvas.width, canvas.height, 'rgba(0, 229, 255, 0.16)')
  ctx.fillStyle = '#00e5ff'
  ctx.font = `${12 * dpr()}px system-ui, sans-serif`
  ctx.fillText(label, 12 * dpr(), 20 * dpr())
  if (values.length < 2) return
  const max = Math.max(1, ...values)
  const min = Math.min(0, ...values)
  const left = 28 * dpr()
  const right = canvas.width - 14 * dpr()
  const top = 32 * dpr()
  const bottom = canvas.height - 22 * dpr()
  ctx.strokeStyle = color
  ctx.lineWidth = 2 * dpr()
  ctx.beginPath()
  for (const [i, value] of values.entries()) {
    const x = left + (i / Math.max(1, values.length - 1)) * (right - left)
    const y = bottom - ((value - min) / Math.max(1, max - min)) * (bottom - top)
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()
  ctx.fillStyle = '#d6fbff'
  ctx.fillText(fmt(values[values.length - 1]), right - 54 * dpr(), top + 4 * dpr())
}

function drawActionGraph(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, snapshot: DashboardSnapshot | null) {
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = '#000407'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  drawGrid(ctx, canvas.width, canvas.height, 'rgba(0, 229, 255, 0.16)')
  if (!snapshot) return
  const counts: Record<string, number> = {}
  for (const event of snapshot.activity ?? []) {
    if (!event.tool) continue
    counts[event.tool] = (counts[event.tool] ?? 0) + 1
  }
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8)
  const max = Math.max(1, ...entries.map((entry) => entry[1]))
  const barH = Math.max(16 * dpr(), (canvas.height - 32 * dpr()) / Math.max(1, entries.length) - 6 * dpr())
  ctx.font = `${12 * dpr()}px system-ui, sans-serif`
  for (const [i, [name, count]] of entries.entries()) {
    const y = 20 * dpr() + i * (barH + 6 * dpr())
    const w = (canvas.width - 170 * dpr()) * (count / max)
    ctx.fillStyle = AGENT_COLORS[i % AGENT_COLORS.length]
    ctx.fillRect(135 * dpr(), y, w, barH)
    ctx.fillStyle = '#d6fbff'
    ctx.fillText(name, 12 * dpr(), y + barH - 4 * dpr())
    ctx.fillText(String(count), 142 * dpr() + w, y + barH - 4 * dpr())
  }
}

function drawCraftingTree(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  snapshot: DashboardSnapshot | null,
  unlocks: Record<string, CraftUnlock>
) {
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = '#000407'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  drawGrid(ctx, canvas.width, canvas.height, 'rgba(0, 229, 255, 0.16)')

  const observations = craftObservations(snapshot)
  const completed = new Set<string>()
  for (const node of CRAFTING_TREE) {
    if (observations.get(node.id)?.seen || unlocks[node.id]) completed.add(node.id)
  }
  const nextTargets = CRAFTING_TREE
    .filter((node) => !completed.has(node.id) && node.deps.every((dep) => completed.has(dep)))
    .map((node) => node.id)
  const progress = completed.size / Math.max(1, CRAFTING_TREE.length)
  const maxTier = Math.max(...CRAFTING_TREE.map((node) => node.tier))
  const left = 30 * dpr()
  const right = canvas.width - 26 * dpr()
  const top = 58 * dpr()
  const bottom = canvas.height - 34 * dpr()
  const laneHeight = (bottom - top) / 2
  const nodeW = Math.min(98 * dpr(), ((right - left) / Math.max(1, maxTier)) * 0.76)
  const nodeH = 28 * dpr()
  const position = (node: CraftNode) => ({
    x: left + (node.tier / Math.max(1, maxTier)) * (right - left),
    y: top + node.lane * laneHeight
  })

  ctx.font = `${10 * dpr()}px system-ui, sans-serif`
  ctx.fillStyle = '#00e5ff'
  ctx.fillText('recipe unlock path', 12 * dpr(), 20 * dpr())
  ctx.fillStyle = '#d6fbff'
  ctx.fillText(`${Math.round(progress * 100)}%`, canvas.width - 58 * dpr(), 20 * dpr())
  ctx.fillStyle = nextTargets.length ? '#f8ff6a' : '#00ffa3'
  ctx.fillText(
    nextTargets.length ? `next: ${nextTargets.map((id) => CRAFTING_TREE.find((node) => node.id === id)?.label).join(', ')}` : 'goal path complete',
    12 * dpr(),
    40 * dpr()
  )

  for (const node of CRAFTING_TREE) {
    const to = position(node)
    for (const depId of node.deps) {
      const dep = CRAFTING_TREE.find((item) => item.id === depId)
      if (!dep) continue
      const from = position(dep)
      const lit = completed.has(dep.id) && (completed.has(node.id) || nextTargets.includes(node.id))
      ctx.strokeStyle = lit ? 'rgba(0, 255, 163, 0.82)' : 'rgba(0, 229, 255, 0.22)'
      ctx.lineWidth = lit ? 2 * dpr() : 1 * dpr()
      ctx.setLineDash(lit ? [] : [5 * dpr(), 5 * dpr()])
      ctx.beginPath()
      ctx.moveTo(from.x + nodeW / 2, from.y)
      ctx.lineTo(to.x - nodeW / 2, to.y)
      ctx.stroke()
    }
  }
  ctx.setLineDash([])

  for (const node of CRAFTING_TREE) {
    const point = position(node)
    const observed = observations.get(node.id)
    const done = completed.has(node.id)
    const next = nextTargets.includes(node.id)
    const active = observed?.active
    const x = point.x - nodeW / 2
    const y = point.y - nodeH / 2
    ctx.fillStyle = done ? 'rgba(0, 255, 163, 0.28)' : next ? 'rgba(248, 255, 106, 0.2)' : 'rgba(0, 12, 16, 0.82)'
    ctx.strokeStyle = active ? '#ff2bd6' : done ? '#00ffa3' : next ? '#f8ff6a' : 'rgba(0, 229, 255, 0.34)'
    ctx.lineWidth = (active ? 2.5 : 1.5) * dpr()
    ctx.fillRect(x, y, nodeW, nodeH)
    ctx.strokeRect(x, y, nodeW, nodeH)
    ctx.fillStyle = done ? '#eaffff' : next ? '#f8ff6a' : '#79f8ff'
    ctx.font = `${8.5 * dpr()}px system-ui, sans-serif`
    ctx.fillText(node.label, x + 6 * dpr(), y + 12 * dpr())
    ctx.fillStyle = done ? '#00ffa3' : '#00e5ff'
    ctx.font = `${7 * dpr()}px system-ui, sans-serif`
    const count = Math.max(observed?.count ?? 0, unlocks[node.id]?.count ?? 0)
    const agents = Math.max(observed?.agents.length ?? 0, unlocks[node.id]?.agents.length ?? 0)
    ctx.fillText(done ? `${count || 1} item | ${agents || 1} agent` : next ? 'ready' : 'locked', x + 6 * dpr(), y + 23 * dpr())
  }
}

function drawLearningSignals(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, history: DashboardHistoryPoint[]) {
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = '#000407'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  drawGrid(ctx, canvas.width, canvas.height, 'rgba(0, 229, 255, 0.16)')
  const points = history.slice(-160)
  const series = [
    { label: 'craft tree', color: '#00e5ff', values: points.map((point) => point.craftProgress * 100) },
    { label: 'success', color: '#00ffa3', values: points.map((point) => point.successRate * 100) },
    { label: 'diversity', color: '#f8ff6a', values: points.map((point) => point.actionDiversity * 100) },
    { label: 'speed', color: '#ff2bd6', values: points.map((point) => Math.max(0, 100 - (point.meanDurationMs / 3500) * 100)) }
  ]
  ctx.fillStyle = '#00e5ff'
  ctx.font = `${12 * dpr()}px system-ui, sans-serif`
  ctx.fillText('learning proxy: unlocks + successful behavior + tool diversity', 12 * dpr(), 20 * dpr())
  if (points.length < 2) {
    centerText(ctx, canvas.width, canvas.height, 'waiting for telemetry history', '#00e5ff')
    return
  }

  const left = 32 * dpr()
  const right = canvas.width - 18 * dpr()
  const top = 42 * dpr()
  const bottom = canvas.height - 28 * dpr()
  ctx.strokeStyle = 'rgba(214, 251, 255, 0.3)'
  ctx.lineWidth = 1 * dpr()
  ctx.beginPath()
  ctx.moveTo(left, top)
  ctx.lineTo(left, bottom)
  ctx.lineTo(right, bottom)
  ctx.stroke()

  for (const item of series) {
    ctx.strokeStyle = item.color
    ctx.lineWidth = 2 * dpr()
    ctx.beginPath()
    for (const [i, value] of item.values.entries()) {
      const x = left + (i / Math.max(1, item.values.length - 1)) * (right - left)
      const y = bottom - (Math.max(0, Math.min(100, value)) / 100) * (bottom - top)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()
  }

  const legendY = canvas.height - 10 * dpr()
  let legendX = 12 * dpr()
  ctx.font = `${9 * dpr()}px system-ui, sans-serif`
  for (const item of series) {
    const last = item.values[item.values.length - 1] ?? 0
    ctx.fillStyle = item.color
    ctx.fillRect(legendX, legendY - 8 * dpr(), 7 * dpr(), 7 * dpr())
    ctx.fillStyle = '#d6fbff'
    const label = `${item.label} ${Math.round(last)}`
    ctx.fillText(label, legendX + 11 * dpr(), legendY)
    legendX += Math.min(128 * dpr(), ctx.measureText(label).width + 26 * dpr())
  }
}

function drawCapabilityMatrix(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, snapshot: DashboardSnapshot | null) {
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = '#000407'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  drawGrid(ctx, canvas.width, canvas.height, 'rgba(0, 229, 255, 0.16)')
  if (!snapshot) {
    centerText(ctx, canvas.width, canvas.height, 'waiting for agent capabilities', '#00e5ff')
    return
  }
  const inventories = observedInventoriesByAgent(snapshot)
  const agents = snapshot.agents.slice(0, 32)
  const nodes = CRAFTING_TREE
  const left = 64 * dpr()
  const top = 34 * dpr()
  const right = canvas.width - 14 * dpr()
  const bottom = canvas.height - 18 * dpr()
  const cellW = (right - left) / nodes.length
  const rowH = Math.max(4 * dpr(), Math.min(12 * dpr(), (bottom - top) / Math.max(1, agents.length)))

  ctx.fillStyle = '#00e5ff'
  ctx.font = `${11 * dpr()}px system-ui, sans-serif`
  ctx.fillText('agent capability unlocks', 12 * dpr(), 20 * dpr())
  ctx.font = `${7 * dpr()}px system-ui, sans-serif`
  for (const [i, node] of nodes.entries()) {
    ctx.fillStyle = i % 2 ? '#79f8ff' : '#d6fbff'
    ctx.fillText(shortCraftLabel(node), left + i * cellW + 2 * dpr(), top - 8 * dpr())
  }

  for (const [row, agent] of agents.entries()) {
    const y = top + row * rowH
    const inventory = inventories.get(agent.name) ?? {}
    const level = nodes.filter((node) => itemCountForAliases(inventory, node.aliases) > 0).length
    ctx.fillStyle = row % 2 ? 'rgba(0, 229, 255, 0.05)' : 'rgba(0, 255, 163, 0.035)'
    ctx.fillRect(0, y, canvas.width, rowH)
    ctx.fillStyle = agent.ready ? '#d6fbff' : '#37616a'
    ctx.font = `${7 * dpr()}px system-ui, sans-serif`
    ctx.fillText(agent.name.replace('agent_', 'a'), 12 * dpr(), y + rowH * 0.72)
    for (const [col, node] of nodes.entries()) {
      const count = itemCountForAliases(inventory, node.aliases)
      const x = left + col * cellW
      const complete = count > 0
      ctx.fillStyle = complete ? capabilityColor(col, level / nodes.length) : 'rgba(0, 229, 255, 0.08)'
      ctx.fillRect(x + 1 * dpr(), y + 1 * dpr(), Math.max(1, cellW - 2 * dpr()), Math.max(1, rowH - 2 * dpr()))
      if (complete && rowH > 7 * dpr()) {
        ctx.fillStyle = '#001014'
        ctx.fillText(String(Math.min(99, count)), x + 3 * dpr(), y + rowH * 0.72)
      }
    }
  }
}

function drawGrid(ctx: CanvasRenderingContext2D, width: number, height: number, color: string) {
  ctx.strokeStyle = color
  ctx.lineWidth = 1
  const step = 48 * dpr()
  ctx.beginPath()
  for (let x = 0; x <= width; x += step) {
    ctx.moveTo(x, 0)
    ctx.lineTo(x, height)
  }
  for (let y = 0; y <= height; y += step) {
    ctx.moveTo(0, y)
    ctx.lineTo(width, y)
  }
  ctx.stroke()
}

function fitCanvas(canvas: HTMLCanvasElement) {
  const rect = canvas.getBoundingClientRect()
  const scale = dpr()
  const width = Math.max(1, Math.round(rect.width * scale))
  const height = Math.max(1, Math.round(rect.height * scale))
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
  }
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('canvas context unavailable')
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  return ctx
}

function worldBounds(snapshot: DashboardSnapshot) {
  const points: Vec3[] = []
  for (const agent of snapshot.agents) {
    if (agent.position) points.push(agent.position)
    points.push(...(agent.trail ?? []))
    points.push(...((agent.blocks ?? []).filter((block) => block.kind !== 'other').slice(0, 80) as BlockSample[]))
  }
  if (!points.length) return { minX: -24, maxX: 24, minZ: -24, maxZ: 24 }
  let minX = Infinity
  let maxX = -Infinity
  let minZ = Infinity
  let maxZ = -Infinity
  for (const point of points) {
    minX = Math.min(minX, point.x)
    maxX = Math.max(maxX, point.x)
    minZ = Math.min(minZ, point.z)
    maxZ = Math.max(maxZ, point.z)
  }
  const pad = Math.max(12, (Math.max(maxX - minX, maxZ - minZ) || 24) * 0.14)
  return { minX: minX - pad, maxX: maxX + pad, minZ: minZ - pad, maxZ: maxZ + pad }
}

function mapPoint(point: Vec3, bounds: ReturnType<typeof worldBounds>, width: number, height: number) {
  return {
    x: ((point.x - bounds.minX) / Math.max(1, bounds.maxX - bounds.minX)) * width,
    y: ((point.z - bounds.minZ) / Math.max(1, bounds.maxZ - bounds.minZ)) * height
  }
}

function minAgentDistance(snapshot: DashboardSnapshot) {
  const distances = (snapshot.society.edges ?? [])
    .filter((edge) => typeof edge.distance === 'number')
    .map((edge) => edge.distance as number)
  return distances.length ? Math.min(...distances) : null
}

function inventoryValue(snapshot: DashboardSnapshot) {
  let total = 0
  for (const inventory of observedInventoriesByAgent(snapshot).values()) {
    for (const [name, count] of Object.entries(inventory)) {
      if (name.includes('log')) total += count
      else if (name.includes('planks')) total += count * 0.25
      else if (name.includes('pickaxe')) total += count * 4
      else if (name.includes('iron')) total += count * 8
      else if (name.includes('diamond')) total += count * 60
      else total += count * 0.1
    }
  }
  return total
}

function learningMetrics(snapshot: DashboardSnapshot) {
  const events = (snapshot.activity ?? []).filter((event) => typeof event.success === 'boolean')
  const recent = events.slice(-48)
  const successes = recent.filter((event) => event.success).length
  const successRate = recent.length ? successes / recent.length : 0
  const tools = new Set(recent.map((event) => event.tool).filter(Boolean))
  const actionDiversity = Math.min(1, tools.size / 8)
  const durations = recent.map((event) => Number(event.duration_ms)).filter((value) => Number.isFinite(value))
  const meanDurationMs = durations.length ? durations.reduce((sum, value) => sum + value, 0) / durations.length : 0
  const craftProgress = craftProgressFromObservations(craftObservations(snapshot))
  return { craftProgress, successRate, actionDiversity, meanDurationMs }
}

function mergeCraftUnlocks(previous: Record<string, CraftUnlock>, snapshot: DashboardSnapshot) {
  const observations = craftObservations(snapshot)
  let changed = false
  const next: Record<string, CraftUnlock> = { ...previous }
  for (const node of CRAFTING_TREE) {
    const observation = observations.get(node.id)
    if (!observation?.seen) continue
    const existing = next[node.id]
    const agents = Array.from(new Set([...(existing?.agents ?? []), ...observation.agents])).sort()
    next[node.id] = {
      firstSeenAt: existing?.firstSeenAt ?? snapshot.ts,
      lastSeenAt: snapshot.ts,
      count: Math.max(existing?.count ?? 0, observation.count),
      agents
    }
    changed = true
  }
  return changed ? next : previous
}

function craftProgressFromObservations(observations: Map<string, CraftObservation>) {
  const seen = CRAFTING_TREE.filter((node) => observations.get(node.id)?.seen).length
  return seen / Math.max(1, CRAFTING_TREE.length)
}

function craftObservations(snapshot: DashboardSnapshot | null) {
  const observations = new Map<string, CraftObservation>()
  for (const node of CRAFTING_TREE) {
    observations.set(node.id, { seen: false, active: false, count: 0, agents: [] })
  }
  if (!snapshot) return observations

  const inventories = observedInventoriesByAgent(snapshot)
  for (const [agentName, inventory] of inventories) {
    for (const node of CRAFTING_TREE) {
      const count = itemCountForAliases(inventory, node.aliases)
      if (count <= 0) continue
      const observation = observations.get(node.id)
      if (!observation) continue
      observation.seen = true
      observation.count += count
      if (!observation.agents.includes(agentName)) observation.agents.push(agentName)
    }
  }

  for (const agent of snapshot.agents ?? []) {
    const activeText = activeTaskText(agent)
    if (!activeText) continue
    for (const node of CRAFTING_TREE) {
      if (!node.aliases.some((alias) => activeText.includes(alias))) continue
      const observation = observations.get(node.id)
      if (observation) observation.active = true
    }
  }

  return observations
}

function observedInventoriesByAgent(snapshot: DashboardSnapshot) {
  const inventories = new Map<string, Record<string, number>>()
  for (const agent of snapshot.agents ?? []) {
    inventories.set(agent.name, normalizeInventory(agent.inventory))
  }

  for (const event of snapshot.activity ?? []) {
    if (!event.bot) continue
    const existing = inventories.get(event.bot) ?? {}
    const result = event.result ?? {}
    const inventory = normalizeInventory(result.inventory)
    const crafted = normalizeInventory(result.crafted)
    const collected = normalizeInventory(result.collected)
    const next = { ...existing }
    mergeInventoryMax(next, inventory)
    mergeInventoryMax(next, crafted)
    mergeInventoryMax(next, collected)
    if (typeof event.item === 'string') next[event.item] = Math.max(next[event.item] ?? 0, 1)
    if (typeof event.block === 'string') next[event.block] = Math.max(next[event.block] ?? 0, 1)
    inventories.set(event.bot, next)
  }

  return inventories
}

function normalizeInventory(value: unknown) {
  const inventory: Record<string, number> = {}
  if (!isRecord(value)) return inventory
  for (const [name, rawCount] of Object.entries(value)) {
    const count = Number(rawCount)
    if (!Number.isFinite(count) || count <= 0) continue
    inventory[name] = Math.max(inventory[name] ?? 0, count)
  }
  return inventory
}

function mergeInventoryMax(target: Record<string, number>, source: Record<string, number>) {
  for (const [name, count] of Object.entries(source)) {
    target[name] = Math.max(target[name] ?? 0, count)
  }
}

function itemCountForAliases(inventory: Record<string, number>, aliases: string[]) {
  let count = 0
  for (const [name, value] of Object.entries(inventory)) {
    if (aliases.some((alias) => itemMatchesAlias(name, alias))) count += value
  }
  return count
}

function itemMatchesAlias(name: string, alias: string) {
  const normalized = name.toLowerCase()
  const target = alias.toLowerCase()
  if (target === 'log') return normalized === 'log' || normalized.endsWith('_log')
  if (target === 'planks') return normalized === 'planks' || normalized.endsWith('_planks')
  return normalized === target
}

function activeTaskText(agent: AgentState) {
  if (!agent.active) return ''
  return `${agent.active.tool} ${JSON.stringify(agent.active.payload ?? {})}`.toLowerCase()
}

function shortCraftLabel(node: CraftNode) {
  const labels: Record<string, string> = {
    logs: 'log',
    planks: 'plk',
    table: 'tbl',
    sticks: 'stk',
    wood_pick: 'wp',
    cobble: 'cob',
    stone_pick: 'sp',
    furnace: 'fur',
    coal: 'col',
    raw_iron: 'ore',
    iron_ingot: 'ing',
    iron_pick: 'ip',
    diamond: 'dia'
  }
  return labels[node.id] ?? node.id.slice(0, 3)
}

function capabilityColor(index: number, level: number) {
  if (level > 0.72) return '#00ffa3'
  if (index % 4 === 0) return '#00e5ff'
  if (index % 4 === 1) return '#00ffa3'
  if (index % 4 === 2) return '#f8ff6a'
  return '#ff2bd6'
}

function summarizeResult(result: ActivityEvent['result']) {
  if (!result) return ''
  if (typeof result.message === 'string') return result.message
  if (isRecord(result.crafted)) return `crafted ${Object.keys(result.crafted).join(', ')}`
  if (isRecord(result.collected)) return `collected ${Object.keys(result.collected).join(', ')}`
  if (typeof result.placed === 'string') return `placed ${result.placed}`
  return ''
}

function centerText(ctx: CanvasRenderingContext2D, width: number, height: number, text: string, color: string) {
  ctx.fillStyle = color
  ctx.font = `${14 * dpr()}px system-ui, sans-serif`
  ctx.textAlign = 'center'
  ctx.fillText(text, width / 2, height / 2)
  ctx.textAlign = 'start'
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

function agentColor(index: number) {
  return AGENT_COLORS[index % AGENT_COLORS.length]
}

function configInt(raw: string | undefined, fallback: number, min = 0) {
  const value = Number.parseInt(raw ?? '', 10)
  return Number.isFinite(value) ? Math.max(min, value) : fallback
}

function configNumber(raw: string | undefined, fallback: number, min = 0) {
  const value = Number.parseFloat(raw ?? '')
  return Number.isFinite(value) ? Math.max(min, value) : fallback
}

function fmt(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  return Number(value).toFixed(digits)
}

function normalizePercent(value: number | null | undefined, max: number) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 0
  return Math.max(0, Math.min(100, (Number(value) / max) * 100))
}

function timeAgo(ts: number) {
  const seconds = Math.max(0, Math.round((Date.now() - ts) / 1000))
  if (seconds < 60) return `${seconds}s`
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
}

function radToDeg(value: number) {
  return (value * 180) / Math.PI
}

function dpr() {
  return typeof window === 'undefined' ? 1 : window.devicePixelRatio || 1
}

function hasWebglSupport() {
  const canvas = document.createElement('canvas')
  return !!(canvas.getContext('webgl2') || canvas.getContext('webgl'))
}

function resolveBridgeEndpoints() {
  const host = window.location.hostname || 'localhost'
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const bridgeHost = formatHostForUrl(host)
  const bridgeHttp = window.location.origin
  const directBridgeWs = shouldUseDirectPorts(host) ? `${protocol}://${bridgeHost}:8780/stream` : null
  return {
    http: BRIDGE_HTTP_ENV || bridgeHttp,
    ws: BRIDGE_WS_ENV || directBridgeWs
  }
}

function formatHostForUrl(host: string) {
  return host.includes(':') && !host.startsWith('[') ? `[${host}]` : host
}

function shouldUseDirectPorts(host: string) {
  const port = typeof window === 'undefined' ? '' : window.location.port
  const normalized = host.toLowerCase()
  return (
    port === '8790' ||
    normalized === 'localhost' ||
    normalized === '127.0.0.1' ||
    normalized === '0.0.0.0' ||
    /^10\.\d+\.\d+\.\d+$/.test(normalized) ||
    /^100\.\d+\.\d+\.\d+$/.test(normalized) ||
    /^192\.168\.\d+\.\d+$/.test(normalized)
  )
}

function readWebglLimit() {
  const raw = new URLSearchParams(window.location.search).get('webgl')
  if (!raw) return MAX_ACTIVE_WEBGL_VIEWERS
  const normalized = raw.toLowerCase()
  if (normalized === 'off' || normalized === 'false' || normalized === 'none') return 0
  return configInt(raw, MAX_ACTIVE_WEBGL_VIEWERS)
}

function viewerHref(host: string, port: number, path = '/') {
  const normalizedPath = (path || '/').startsWith('/') ? path || '/' : `/${path}`
  const params = new URLSearchParams()
  if (VIEWER_IFRAME_FPS > 0) params.set('fps', String(VIEWER_IFRAME_FPS))
  if (VIEWER_IFRAME_DPR > 0) params.set('dpr', String(VIEWER_IFRAME_DPR))
  const query = params.toString()
  const separator = normalizedPath.includes('?') ? '&' : '?'
  if (!shouldUseDirectPorts(host)) {
    return `/viewer/${port}${normalizedPath}${query ? `${separator}${query}` : ''}`
  }
  return `http://${formatHostForUrl(host || 'localhost')}:${port}${normalizedPath}${query ? `${separator}${query}` : ''}`
}
