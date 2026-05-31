export type Vec3 = {
  x: number
  y: number
  z: number
}

export type ViewerInfo = {
  port: number
  path: string
  kind: string
}

export type BlockSample = Vec3 & {
  name: string
  kind: string
}

export type EntitySample = Vec3 & {
  id: number
  name: string
  type: string | null
  kind: string | null
}

export type ActiveTask = {
  tool: string
  payload: Record<string, unknown>
  started_at: number
}

export type AgentState = {
  name: string
  ready: boolean
  viewer: ViewerInfo | null
  position?: Vec3
  yaw?: number
  pitch?: number
  velocity?: Vec3
  health?: number
  food?: number
  inventory?: Record<string, number>
  equipped?: string | null
  line_of_sight?: string | null
  time_of_day?: number | null
  active?: ActiveTask | null
  blocks?: BlockSample[]
  entities?: EntitySample[]
  trail?: (Vec3 & { ts?: number })[]
}

export type SocietyEdge = {
  source: string
  target: string
  distance: number | null
  item?: string | null
  collaboration?: boolean
  ts?: number
}

export type ActivityEvent = {
  ts: number
  level?: string
  bot?: string | null
  target?: string | null
  item?: string | null
  block?: string | null
  tool?: string | null
  success?: boolean
  duration_ms?: number
  result?: Record<string, unknown>
  message?: string
}

export type DashboardSnapshot = {
  type: 'snapshot'
  ts: number
  server: {
    host: string
    port: number
    version: string | null
    bridge_port: number
    dashboard_port: number
    overview_viewer?: { port: number; bot: string } | null
  }
  agents: AgentState[]
  society: {
    edges: SocietyEdge[]
  }
  activity: ActivityEvent[]
}
