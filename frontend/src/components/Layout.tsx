import { NavLink, Outlet } from 'react-router-dom'
import { BarChart3, LineChart, Moon, Search, Sun, TrendingUp } from 'lucide-react'
import { useTheme } from './ThemeProvider'
import { cn } from '@/lib/utils'

const navItems = [
  { to: '/', label: 'Dashboard', icon: BarChart3 },
  { to: '/stocks', label: 'Stocks', icon: Search },
  { to: '/forecast', label: 'Forecast', icon: LineChart },
]

export function Layout() {
  const { theme, toggle } = useTheme()
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="flex">
        {/* Sidebar */}
        <aside className="sticky top-0 hidden h-screen w-60 shrink-0 border-r bg-card md:flex md:flex-col">
          <div className="flex h-16 items-center gap-2 border-b px-6">
            <TrendingUp className="h-6 w-6 text-success" />
            <span className="font-semibold tracking-tight">Stock MLOps</span>
          </div>
          <nav className="flex-1 space-y-1 p-4">
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-accent text-accent-foreground'
                      : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground'
                  )
                }
              >
                <Icon className="h-4 w-4" />
                {label}
              </NavLink>
            ))}
          </nav>
          <div className="border-t p-4 text-xs text-muted-foreground">
            <p>Stock Prediction</p>
            <p>v1.0 · XGBoost</p>
          </div>
        </aside>

        {/* Main */}
        <main className="flex-1 min-w-0">
          <header className="sticky top-0 z-10 flex h-16 items-center justify-between border-b bg-background/80 px-6 backdrop-blur">
            <div className="flex items-center gap-2 md:hidden">
              <TrendingUp className="h-5 w-5 text-success" />
              <span className="font-semibold">Stock MLOps</span>
            </div>
            <div className="ml-auto flex items-center gap-2">
              <button
                onClick={toggle}
                className="rounded-md border bg-card p-2 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                aria-label="Toggle theme"
              >
                {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </button>
            </div>
          </header>

          {/* Mobile nav */}
          <nav className="flex border-b md:hidden">
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  cn(
                    'flex flex-1 items-center justify-center gap-2 py-3 text-sm font-medium border-b-2',
                    isActive
                      ? 'border-foreground text-foreground'
                      : 'border-transparent text-muted-foreground'
                  )
                }
              >
                <Icon className="h-4 w-4" />
                {label}
              </NavLink>
            ))}
          </nav>

          <div className="p-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}
