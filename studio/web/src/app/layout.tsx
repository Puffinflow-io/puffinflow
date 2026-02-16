import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "PuffinFlow Studio",
  description: "Visual AI Agent Builder — build, test, and deploy agent workflows",
};

function Sidebar() {
  return (
    <aside className="w-56 border-r border-border bg-card flex flex-col">
      <div className="p-4 border-b border-border">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <span className="text-primary-foreground font-bold text-sm">PF</span>
          </div>
          <div>
            <h1 className="font-semibold text-sm leading-none">PuffinFlow</h1>
            <span className="text-xs text-muted-foreground">Studio</span>
          </div>
        </Link>
      </div>
      <nav className="flex-1 p-3 space-y-1">
        <Link
          href="/"
          className="flex items-center gap-2 px-3 py-2 rounded-md text-sm hover:bg-accent transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
          </svg>
          Projects
        </Link>
      </nav>
      <div className="p-3 border-t border-border">
        <div className="text-xs text-muted-foreground">PuffinFlow Studio v0.1.0</div>
      </div>
    </aside>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">
        <div className="flex h-screen overflow-hidden">
          <Sidebar />
          <main className="flex-1 overflow-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
