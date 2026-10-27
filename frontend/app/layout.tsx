import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'MacReg | Academic calendar research',
  description: 'Explore academic calendar questions against a small synthetic research corpus.',
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
