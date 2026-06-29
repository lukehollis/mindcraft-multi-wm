import { NextResponse } from 'next/server'

const BRIDGE_HTTP = process.env.DASHBOARD_BRIDGE_HTTP || process.env.BRIDGE_HTTP || 'http://bridge:8780'

export async function GET() {
  try {
    const response = await fetch(`${BRIDGE_HTTP.replace(/\/$/, '')}/snapshot`, {
      cache: 'no-store'
    })
    if (!response.ok) {
      return NextResponse.json(
        { error: 'bridge snapshot unavailable', status: response.status },
        { status: 502 }
      )
    }
    const snapshot = await response.json()
    return NextResponse.json(snapshot, {
      headers: {
        'cache-control': 'no-store'
      }
    })
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'bridge snapshot unavailable' },
      { status: 502 }
    )
  }
}
