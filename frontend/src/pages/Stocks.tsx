import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { Search } from 'lucide-react'
import { fetchSortedStocks } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { formatNumber } from '@/lib/utils'
import { Link } from 'react-router-dom'
import { cn } from '@/lib/utils'

export default function Stocks() {
  const [sortBy, setSortBy] = useState<'price' | 'volume'>('volume')
  const [order, setOrder] = useState<'asc' | 'desc'>('desc')
  const [query, setQuery] = useState('')

  const { data, isLoading } = useQuery({
    queryKey: ['stocks-sorted', sortBy, order],
    queryFn: () => fetchSortedStocks(sortBy, order, 100),
  })

  const filtered = (data?.data ?? []).filter((s) =>
    query.trim() ? s.ticker.toUpperCase().includes(query.trim().toUpperCase()) : true
  )

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">All Stocks</h1>
        <p className="text-sm text-muted-foreground">
          {data?.total ? `${formatNumber(data.total)} mã đang theo dõi` : 'Đang tải...'}
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Tìm ticker..."
            className="w-full rounded-md border bg-card pl-10 pr-3 py-2 text-sm outline-none focus:border-foreground"
          />
        </div>
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as 'price' | 'volume')}
          className="rounded-md border bg-card px-3 py-2 text-sm"
        >
          <option value="volume">Volume</option>
          <option value="price">Price</option>
        </select>
        <select
          value={order}
          onChange={(e) => setOrder(e.target.value as 'asc' | 'desc')}
          className="rounded-md border bg-card px-3 py-2 text-sm"
        >
          <option value="desc">Cao → Thấp</option>
          <option value="asc">Thấp → Cao</option>
        </select>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Sorted by {sortBy} ({order})</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-96" />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
                    <th className="pb-2 font-medium">Ticker</th>
                    <th className="pb-2 font-medium text-right">Close</th>
                    <th className="pb-2 font-medium text-right">Volume</th>
                    <th className="pb-2 font-medium text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {filtered.slice(0, 50).map((s) => (
                    <tr key={s.ticker} className="hover:bg-accent/40">
                      <td className="py-3 font-mono font-semibold">{s.ticker}</td>
                      <td className="py-3 text-right">{formatNumber(s.close)}</td>
                      <td className="py-3 text-right text-muted-foreground">
                        {formatNumber(s.volume)}
                      </td>
                      <td className="py-3 text-right">
                        <Link
                          to={`/forecast?ticker=${s.ticker}`}
                          className={cn(
                            'inline-flex items-center rounded-md border bg-card px-3 py-1 text-xs font-medium',
                            'hover:bg-accent hover:text-foreground'
                          )}
                        >
                          Xem dự đoán →
                        </Link>
                      </td>
                    </tr>
                  ))}
                  {filtered.length === 0 && (
                    <tr>
                      <td colSpan={4} className="py-8 text-center text-sm text-muted-foreground">
                        Không tìm thấy ticker phù hợp
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
