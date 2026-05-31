import type { Metadata } from 'next'
import type { ReactNode } from 'react'
import './globals.css'
import { Geist } from 'next/font/google'
import { cn } from '@/lib/utils'

const geist = Geist({ subsets: ['latin'], variable: '--font-sans' })

export const metadata: Metadata = {
  title: 'Mindcraft Dashboard',
  description: 'Live multi-agent Minecraft society dashboard'
}

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en" className={cn('dark font-sans', geist.variable)}>
      <body>{children}</body>
    </html>
  )
}
