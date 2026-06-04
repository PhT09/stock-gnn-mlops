import { useQuery } from '@tanstack/react-query'
import { Activity, ArrowDown, ArrowUp, ShieldCheck, TrendingUp } from 'lucide-react'
import { fetchSummary, fetchTopBuy, fetchTopSell } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle, StatCard } from '@/components/ui/Card'
import { ConfidenceBadge } from '@/components/ui/Badge'
import { Skeleton } from '@/components/ui/Skeleton'
import { formatNumber, formatPercent } from '@/lib/utils'
import { Link } from 'react-router-dom'

export default function Dashboard() {
  const summary = useQuery({ queryKey: ['summary'], queryFn: fetchSummary })
  const topBuy = useQuery({ queryKey: ['top-buy'], queryFn: () => fetchTopBuy(5, 'MEDIUM') })
  const topSell = useQuery({ queryKey: ['top-sell'], queryFn: () => fetchTopSell(5, 'MEDIUM') })

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Market Overview</h1>
        <p className="text-sm text-muted-foreground">
          Dự đoán phiên giao dịch kế tiếp{' '}
          {summary.data?.next_session_date && (
            <span className="font-medium text-foreground">({summary.data.next_session_date})</span>
          )}
        </p>
      </div>

      {/* Stat cards */}
      {summary.isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
      ) : summary.data ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            label="Total Tickers"
            value={formatNumber(summary.data.total_tickers)}
            hint="Cổ phiếu được dự đoán"
            icon={<Activity className="h-5 w-5" />}
          />
          <StatCard
            label="Dự đoán TĂNG"
            value={formatPercent(summary.data.up_percent)}
            hint={`${formatNumber(summary.data.up_count)} mã`}
            icon={<ArrowUp className="h-5 w-5" />}
            trend="up"
          />
          <StatCard
            label="Dự đoán GIẢM"
            value={formatPercent(summary.data.down_percent)}
            hint={`${formatNumber(summary.data.down_count)} mã`}
            icon={<ArrowDown className="h-5 w-5" />}
            trend="down"
          />
          <StatCard
            label="HIGH Confidence"
            value={formatNumber(summary.data.high_confidence_count)}
            hint={`Sentiment: ${summary.data.market_sentiment}`}
            icon={<ShieldCheck className="h-5 w-5" />}
          />
        </div>
      ) : null}

      {/* Top movers */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-success" />
              Top BUY
            </CardTitle>
          </CardHeader>
          <CardContent>
            {topBuy.isLoading ? (
              <Skeleton className="h-40" />
            ) : (
              <MoversTable rows={topBuy.data?.data ?? []} kind="buy" />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 rotate-180 text-danger" />
              Top SELL
            </CardTitle>
          </CardHeader>
          <CardContent>
            {topSell.isLoading ? (
              <Skeleton className="h-40" />
            ) : (
              <MoversTable rows={topSell.data?.data ?? []} kind="sell" />
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function MoversTable({
  rows,
  kind,
}: {
  rows: { ticker: string; probability: number; confidence: 'HIGH' | 'MEDIUM' | 'LOW' }[]
  kind: 'buy' | 'sell'
}) {
  if (!rows.length) return <p className="text-sm text-muted-foreground">Chưa có dữ liệu</p>
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
          <th className="pb-2 font-medium">Ticker</th>
          <th className="pb-2 font-medium">Probability</th>
          <th className="pb-2 font-medium text-right">Confidence</th>
        </tr>
      </thead>
      <tbody className="divide-y">
        {rows.map((r) => (
          <tr key={r.ticker} className="group">
            <td className="py-3">
              <Link
                to={`/forecast?ticker=${r.ticker}`}
                className="font-mono font-semibold transition-colors group-hover:text-primary"
              >
                {r.ticker}
              </Link>
            </td>
            <td className="py-3">
              <span className={kind === 'buy' ? 'text-success' : 'text-danger'}>
                {(r.probability * 100).toFixed(1)}%
              </span>
            </td>
            <td className="py-3 text-right">
              <ConfidenceBadge level={r.confidence} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
