import { useState, useCallback } from 'react'
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { C } from '../constants'
import { api } from '../api'
import { usePolling } from '../hooks/usePolling'
import SectionHeader from './SectionHeader'

function WindowToggle({ chartWindow, setChartWindow }) {
  return (
    <div style={{ display: 'flex', gap: 4 }}>
      {[7, 30].map(w => (
        <button
          key={w}
          onClick={() => setChartWindow(w)}
          style={{
            fontSize: 10,
            fontWeight: 500,
            letterSpacing: '0.04em',
            textTransform: 'uppercase',
            padding: '2px 7px',
            borderRadius: 3,
            border: `0.5px solid ${w === chartWindow ? '#1a6644' : C.border3}`,
            background: w === chartWindow ? C.greenBg : 'transparent',
            color: w === chartWindow ? C.green : C.muted,
            cursor: 'pointer',
            outline: 'none',
          }}
        >
          {w}d
        </button>
      ))}
    </div>
  )
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#1c1c1a',
      border: `0.5px solid #2f2f2c`,
      borderRadius: 4,
      padding: '6px 10px',
      fontSize: 11,
      color: C.text,
    }}>
      <div style={{ color: C.muted, marginBottom: 4 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, fontVariantNumeric: 'tabular-nums' }}>
          {p.name}: {p.value != null ? p.value.toFixed(4) : 'N/A'}
        </div>
      ))}
    </div>
  )
}

export default function CorrelationChart({ ticker }) {
  const [chartWindow, setChartWindow] = useState(7)

  const fetchFn = useCallback(() => api.correlation(ticker, chartWindow), [ticker, chartWindow])
  const { data, error, loading } = usePolling(fetchFn, [ticker, chartWindow])

  const rows = (data?.data ?? []).slice(-chartWindow)

  const infoBody = ticker === 'SPY'
    ? 'All headlines scored for this ticker on a given trading day are aggregated into a single confidence-weighted mean sentiment score. That score is paired with the log return recorded for the following session, introducing a one-day lag. This lag is applied because same-session price movements are presumed to already reflect available information. The amber line represents the daily mean sentiment score. The dashed blue line represents the next-session log return. SPY tracks the broader market index, which is subject to macro influences that may not be fully captured in headline sentiment alone, and this is reflected in the weaker correlation observed relative to individual equities.'
    : 'All headlines scored for this ticker on a given trading day are aggregated into a single confidence-weighted mean sentiment score. That score is paired with the log return recorded for the following session, introducing a one-day lag. This lag is applied because same-session price movements are presumed to already reflect available information. The amber line represents the daily mean sentiment score. The dashed blue line represents the next-session log return.'

  const infoExample = ticker === 'SPY'
    ? "All SPY articles on Jun 27 produced a mean sentiment of +0.81, paired with SPY's return of +1.14% on Jun 28"
    : `All ${ticker} articles on Jun 27 produced a mean sentiment of +0.81, paired with ${ticker}'s return of +1.14% on Jun 28`

  return (
    <div style={{ opacity: loading && !data ? 0.3 : 1 }}>
      <SectionHeader
        title="Sentiment vs Return"
        subtitle={`${ticker} · ${chartWindow}d rolling`}
        error={error}
        rightSlot={<WindowToggle chartWindow={chartWindow} setChartWindow={setChartWindow} />}
        infoTitle="Daily sentiment score against forward return"
        infoBody={infoBody}
        infoExample={infoExample}
      />

      <div style={{ padding: '12px 8px 8px', background: C.bg }}>
        <div style={{ display: 'flex', gap: 16, padding: '0 8px', marginBottom: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 16, height: 1.5, background: C.amber }} />
            <span style={{ fontSize: 10, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Mean Sentiment
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 16, height: 0, borderTop: `1.5px dashed ${C.blue}` }} />
            <span style={{ fontSize: 10, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Log Return (T+1)
            </span>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={130}>
          <ComposedChart data={rows} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="#141412" strokeWidth={0.5} vertical={false} />
            <XAxis
              dataKey="market_date"
              tick={{ fontSize: 10, fill: C.muted }}
              axisLine={{ stroke: C.border }}
              tickLine={false}
              interval="preserveStartEnd"
              tickCount={6}
            />
            <YAxis
              yAxisId="left"
              tick={{ fontSize: 10, fill: C.amber }}
              axisLine={{ stroke: C.border }}
              tickLine={false}
              width={36}
              tickFormatter={v => v.toFixed(2)}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              tick={{ fontSize: 10, fill: C.blue }}
              axisLine={{ stroke: C.border }}
              tickLine={false}
              width={40}
              tickFormatter={v => `${(v * 100).toFixed(1)}%`}
            />
            <ReferenceLine yAxisId="left" y={0} stroke={C.border3} strokeWidth={0.5} />
            <Tooltip content={<ChartTooltip />} />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="mean_sentiment"
              name="Sentiment"
              stroke={C.amber}
              strokeWidth={1.5}
              dot={false}
              connectNulls={false}
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="log_return"
              name="Log Return"
              stroke={C.blue}
              strokeWidth={1.5}
              strokeDasharray="4 3"
              dot={false}
              connectNulls={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
