import axios from 'axios'

// Khi chạy trong Docker hoặc dev, Vite proxy /api → backend.
// Khi build prod (preview), VITE_API_URL có thể override.
export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? '/api/v1',
  timeout: 30000,
})

// =========================
// Types
// =========================

export interface MarketSummary {
  total_tickers: number
  next_session_date: string
  up_count: number
  down_count: number
  up_percent: number
  down_percent: number
  market_sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
  high_confidence_count: number
  confidence_breakdown: { HIGH: number; MEDIUM: number; LOW: number }
  avg_probability: number
}

export interface TopMover {
  ticker: string
  next_date: string
  signal: string
  probability: number
  confidence: 'HIGH' | 'MEDIUM' | 'LOW'
}

export interface TopMoversResp {
  filter: { signal: string; min_confidence: string }
  count: number
  data: TopMover[]
}

export interface Forecast {
  day: number
  date: string
  prediction: 0 | 1
  signal: string
  probability: number
  confidence: 'HIGH' | 'MEDIUM' | 'LOW'
}

export interface TickerForecast {
  ticker: string
  latest_data_date: string
  forecast: Forecast[]
}

export interface Stock {
  ticker: string
  close: number
  volume: number
}

export interface StocksResp {
  status: string
  count?: number
  total?: number
  data: Stock[]
}

// =========================
// API calls
// =========================

export const fetchSummary = () =>
  api.get<MarketSummary>('/dashboard/summary').then((r) => r.data)

export const fetchTopBuy = (limit = 10, min_confidence = 'MEDIUM') =>
  api
    .get<TopMoversResp>('/predictions/top-buy', { params: { limit, min_confidence } })
    .then((r) => r.data)

export const fetchTopSell = (limit = 10, min_confidence = 'MEDIUM') =>
  api
    .get<TopMoversResp>('/predictions/top-sell', { params: { limit, min_confidence } })
    .then((r) => r.data)

export const fetchTickerForecast = (ticker: string) =>
  api.get<TickerForecast>(`/predictions/${ticker.toUpperCase()}`).then((r) => r.data)

export const fetchStocks = (query?: string, limit = 50) =>
  api
    .get<StocksResp>('/stocks', { params: { query, limit } })
    .then((r) => r.data)

export const fetchSortedStocks = (
  sort_by: 'price' | 'volume',
  order: 'asc' | 'desc' = 'desc',
  limit = 20,
  offset = 0
) =>
  api
    .get<StocksResp>('/stocks/sorted', { params: { sort_by, order, limit, offset } })
    .then((r) => r.data)
