export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

const VIEWER_HOST = process.env.DASHBOARD_VIEWER_HOST || 'bridge'
const MIN_VIEWER_PORT = Number(process.env.DASHBOARD_MIN_VIEWER_PORT || 3007)
const MAX_VIEWER_PORT = Number(process.env.DASHBOARD_MAX_VIEWER_PORT || 3040)

type ViewerRouteContext = {
  params: Promise<{
    port: string
    path?: string[]
  }>
}

export async function GET(request: Request, context: ViewerRouteContext) {
  return proxyViewerRequest(request, context)
}

export async function POST(request: Request, context: ViewerRouteContext) {
  return proxyViewerRequest(request, context)
}

export async function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: {
      'cache-control': 'no-store'
    }
  })
}

async function proxyViewerRequest(request: Request, context: ViewerRouteContext) {
  const params = await context.params
  const port = Number(params.port)
  if (!Number.isInteger(port) || port < MIN_VIEWER_PORT || port > MAX_VIEWER_PORT) {
    return Response.json({ error: 'viewer port unavailable' }, { status: 404 })
  }

  const sourceUrl = new URL(request.url)
  const path = viewerTargetPath(params.path ?? [])
  const targetUrl = `http://${VIEWER_HOST}:${port}${path}${sourceUrl.search}`
  const headers = forwardedHeaders(request.headers)
  const body = request.method === 'GET' || request.method === 'HEAD'
    ? undefined
    : await request.arrayBuffer()

  try {
    const upstream = await fetch(targetUrl, {
      method: request.method,
      headers,
      body,
      cache: 'no-store',
      redirect: 'manual'
    })
    const rewritten = await rewrittenViewerBody(upstream, port, params.path ?? [])
    if (rewritten !== null) {
      return new Response(rewritten, {
        status: upstream.status,
        statusText: upstream.statusText,
        headers: responseHeaders(upstream.headers)
      })
    }
    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders(upstream.headers)
    })
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : 'viewer unavailable' },
      { status: 502 }
    )
  }
}

async function rewrittenViewerBody(response: Response, port: number, path: string[]) {
  if (!response.ok) return null
  const contentType = response.headers.get('content-type') || ''
  const routePath = `/${path.join('/')}`
  if (contentType.includes('text/html') || routePath === '/') {
    const html = await response.text()
    const base = `/viewer/${port}/`
    return html
      .replace('<head>', `<head><base href="${base}">`)
      .replace(
        '<script type="text/javascript" src="index.js"></script>',
        `<script>window.__VIEWER_SOCKET_PATH__="${base}socket.io"</script><script type="text/javascript" src="index.js"></script>`
      )
  }
  if (routePath === '/index.js' && contentType.includes('javascript')) {
    const script = await response.text()
    return script.replace(
      'path:window.location.pathname+"socket.io"',
      'path:window.__VIEWER_SOCKET_PATH__||window.location.pathname+"socket.io"'
    )
  }
  return null
}

function forwardedHeaders(headers: Headers) {
  const next = new Headers(headers)
  for (const name of [
    'connection',
    'content-length',
    'host',
    'keep-alive',
    'proxy-authenticate',
    'proxy-authorization',
    'te',
    'trailer',
    'transfer-encoding',
    'upgrade'
  ]) {
    next.delete(name)
  }
  return next
}

function viewerTargetPath(path: string[]) {
  if (path[0] === 'socket.io') {
    const rest = path.slice(1).map(encodeURIComponent).join('/')
    return `/socket.io/${rest}`
  }
  return `/${path.map(encodeURIComponent).join('/')}`
}

function responseHeaders(headers: Headers) {
  const next = new Headers(headers)
  for (const name of [
    'connection',
    'content-encoding',
    'content-length',
    'keep-alive',
    'proxy-authenticate',
    'proxy-authorization',
    'te',
    'trailer',
    'transfer-encoding',
    'upgrade'
  ]) {
    next.delete(name)
  }
  next.set('cache-control', 'no-store')
  return next
}
