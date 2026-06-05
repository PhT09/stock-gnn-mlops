import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { useState, useEffect } from 'react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { fetchTickerForecast } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card'
import { ConfidenceBadge, SignalBadge } from '@/components/ui/Badge'
import { Skeleton } from '@/components/ui/Skeleton'

export default function Forecast() {
  const [searchParams, setSearchParams] = useSearchParams()
  const initialTicker = searchParams.get('ticker') ?? 'FPT'
  const [tickerInput, setTickerInput] = useState(initialTicker)
  const [ticker, setTicker] = useState(initialTicker)

  useEffect(() => {
    setSearchParams({ ticker })
  }, [ticker, setSearchParams])

  const { data, isLoading, error } = useQuery({
    queryKey: ['forecast', ticker],
    queryFn: () => fetchTickerForecast(ticker),
    enabled: Boolean(ticker),
  })

  const chartData =
    data?.forecast.map((f) => ({
      day: `D${f.day}`,
      date: f.date,
      probability: Number((f.probability * 100).toFixed(2)),
      prediction: f.prediction,
      signal: f.signal,
      confidence: f.confidence,
    })) ?? []

  const upDays = data?.forecast.filter((f) => f.prediction === 1).length ?? 0
  const downDays = data?.forecast.filter((f) => f.prediction === 0).length ?? 0
  const highConfDays = data?.forecast.filter((f) => f.confidence === 'HIGH').length ?? 0

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">15-Day Forecast</h1>
          <p className="text-sm text-muted-foreground">
            Dự đoán xu hướng 15 phiên giao dịch tiếp theo
          </p>
        </div>
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            setTicker(tickerInput.trim().toUpperCase())
          }}
        >
          <input
            value={tickerInput}
            onChange={(e) => setTickerInput(e.target.value)}
            placeholder="Ticker (vd: FPT)"
            className="rounded-md border bg-card px-3 py-2 text-sm uppercase outline-none focus:border-foreground"
          />
          <button
            type="submit"
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90"
          >
            Xem
          </button>
        </form>
      </div>

      {isLoading && <Skeleton className="h-96" />}

      {error && (
        <Card>
          <CardContent className="p-8 text-center">
            <p className="text-danger">Không có dự đoán cho mã "{ticker}"</p>
            <p className="mt-2 text-sm text-muted-foreground">
              Vui lòng kiểm tra lại mã hoặc thử mã khác (vd: FPT, VCB, HPG)
            </p>
          </CardContent>
        </Card>
      )}

      {data && (
        <>
          {/* Header ticker info */}
          <Card>
            <CardContent className="flex flex-wrap items-center justify-between gap-4 p-6">
              <div>
                <div className="font-mono text-3xl font-bold">{data.ticker}</div>
                <p className="text-sm text-muted-foreground">
                  Latest data: {data.latest_data_date}
                </p>
              </div>
              <div className="flex gap-6 text-sm">
                <div>
                  <p className="text-xs text-muted-foreground">↑ TĂNG</p>
                  <p className="text-xl font-bold text-success">{upDays}/15</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">↓ GIẢM</p>
                  <p className="text-xl font-bold text-danger">{downDays}/15</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">HIGH conf.</p>
                  <p className="text-xl font-bold">{highConfDays}/15</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Probability TĂNG theo ngày</CardTitle>
              <CardDescription>
                Đường ngang 50% là ngưỡng quyết định. Trên 50% = dự đoán TĂNG, dưới = GIẢM.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-72 w-full">
                <ResponsiveContainer>
                  <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="up" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--success))" stopOpacity={0.5} />
                        <stop offset="95%" stopColor="hsl(var(--success))" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="day" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <YAxis
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={12}
                      domain={[0, 100]}
                      tickFormatter={(v) => `${v}%`}
                    />
                    <Tooltip
                      contentStyle={{
                        background: 'hsl(var(--card))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                      formatter={(value: number) => [`${value.toFixed(2)}%`, 'Probability']}
                      labelFormatter={(label, payload) => {
                        const row = (payload as { payload?: { date?: string } }[])?.[0]?.payload
                        return row?.date ? `${label} · ${row.date}` : label
                      }}
                    />
                    <ReferenceLine y={50} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" />
                    <Area
                      type="monotone"
                      dataKey="probability"
                      stroke="hsl(var(--success))"
                      fill="url(#up)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Table */}
          <Card>
            <CardHeader>
              <CardTitle>Chi tiết 15 ngày</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
                      <th className="pb-2 font-medium">Day</th>
                      <th className="pb-2 font-medium">Date</th>
                      <th className="pb-2 font-medium">Signal</th>
                      <th className="pb-2 font-medium text-right">Probability</th>
                      <th className="pb-2 font-medium text-right">Confidence</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {data.forecast.map((f) => (
                      <tr key={f.day} className="hover:bg-accent/40">
                        <td className="py-2 font-mono">{f.day}</td>
                        <td className="py-2 text-muted-foreground">{f.date}</td>
                        <td className="py-2">
                          <SignalBadge prediction={f.prediction} />
                        </td>
                        <td className="py-2 text-right">
                          {(f.probability * 100).toFixed(2)}%
                        </td>
                        <td className="py-2 text-right">
                          <ConfidenceBadge level={f.confidence} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
