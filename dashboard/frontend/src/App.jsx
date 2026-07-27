import { useState, useCallback } from 'react'
import { C } from './constants'
import { api } from './api'
import { usePolling } from './hooks/usePolling'
import TopBar from './components/TopBar'
import ClockStrip from './components/ClockStrip'
import TickerNav from './components/TickerNav'
import TickerPage from './components/TickerPage'

const TICKER_SYMBOLS = ['AAPL', 'TSLA', 'SPY']

export default function App() {
  const [activeTicker, setActiveTicker] = useState('AAPL')

  const { data: pricesData } = usePolling(
    useCallback(() => api.prices(), []),
    [],
    300000,
  )

  const priceMap = {}
  for (const row of pricesData?.data ?? []) {
    // Convert log return to simple percentage: (e^r - 1) * 100
    priceMap[row.ticker] = row.daily_return != null
      ? (Math.exp(row.daily_return) - 1) * 100
      : null
  }

  const tickers = TICKER_SYMBOLS.map(symbol => ({
    symbol,
    change: priceMap[symbol] ?? null,
  }))

  return (
    <div style={{ background: C.bg, minHeight: '100vh', color: C.text }}>
      <TopBar />
      <ClockStrip />
      <TickerNav
        activeTicker={activeTicker}
        setActiveTicker={setActiveTicker}
        tickers={tickers}
      />
      <TickerPage key={activeTicker} ticker={activeTicker} />
    </div>
  )
}
