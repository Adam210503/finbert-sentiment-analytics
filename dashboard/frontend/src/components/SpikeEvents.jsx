import { useCallback } from 'react'
import { C } from '../constants'
import { api } from '../api'
import { usePolling } from '../hooks/usePolling'
import SectionHeader from './SectionHeader'

function ReturnCell({ label, value }) {
  const isPos  = value != null && value > 0
  const isNeg  = value != null && value < 0
  const color  = isPos ? C.green : isNeg ? C.red : C.muted
  const display = value != null
    ? `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%`
    : 'N/A'

  return (
    <div style={{
      background: C.bg,
      border: `0.5px solid ${C.border}`,
      borderRadius: 3,
      padding: '4px 3px',
      textAlign: 'center',
    }}>
      <div style={{
        fontSize: 8,
        color: C.muted,
        textTransform: 'uppercase',
        letterSpacing: '0.04em',
        marginBottom: 2,
      }}>
        {label}
      </div>
      <div style={{
        fontSize: 10,
        fontWeight: 500,
        fontVariantNumeric: 'tabular-nums',
        letterSpacing: '-0.01em',
        color,
      }}>
        {display}
      </div>
    </div>
  )
}

function SpikeCard({ event }) {
  const isPos   = event.event_type === 'positive'
  const bdrColor = isPos ? C.greenBdr : C.redBdr

  return (
    <div style={{
      background: C.surface,
      borderRadius: 5,
      padding: '9px 10px',
      border: `0.5px solid ${bdrColor}`,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span style={{
          fontSize: 10,
          color: C.muted,
          fontVariantNumeric: 'tabular-nums',
        }}>
          {event.event_date}
        </span>
        <span style={{
          fontSize: 9,
          textTransform: 'uppercase',
          letterSpacing: '0.04em',
          padding: '2px 6px',
          borderRadius: 3,
          background: isPos ? '#051510' : C.redBg,
          color: isPos ? C.green : C.red,
        }}>
          {event.event_type}
        </span>
      </div>

      <div style={{
        fontSize: 12,
        fontWeight: 500,
        color: C.text,
        fontVariantNumeric: 'tabular-nums',
        letterSpacing: '-0.01em',
        marginBottom: 2,
      }}>
        {event.mean_sentiment != null
          ? `${event.mean_sentiment >= 0 ? '+' : ''}${event.mean_sentiment.toFixed(4)}`
          : '—'
        }
      </div>

      <div style={{ fontSize: 10, color: C.muted, marginBottom: 7 }}>
        mean sentiment
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 3 }}>
        <ReturnCell label="T+1d" value={event.return_t1d} />
        <ReturnCell label="T+2d" value={event.return_t2d} />
        <ReturnCell label="T+5d" value={event.return_t5d} />
      </div>
    </div>
  )
}

export default function SpikeEvents({ ticker }) {
  const fetchFn = useCallback(() => api.events(ticker), [ticker])
  const { data, error, loading } = usePolling(fetchFn, [ticker])

  const events = data?.data ?? []

  return (
    <div style={{ opacity: loading && !data ? 0.3 : 1 }}>
      <SectionHeader
        title="Spike Events"
        subtitle={`${ticker} · ${events.length} events`}
        error={error}
        infoTitle="Positive and negative spike events"
        infoBody="A spike event is identified on any trading day where the confidence-weighted mean sentiment score for this ticker exceeds +0.65 (positive spike) or falls below -0.65 (negative spike). On a scale of -1 to +1, these thresholds represent sessions in which coverage was strongly directional and the model assigned high confidence to its classifications. Forward log returns are recorded at 1, 2, and 5 trading days following a positive or negative spike to assess whether extreme sentiment readings serve as a leading indicator of price direction."
        infoExample="A mean sentiment of +0.96 is classified as a positive spike. Returns are then recorded at t+1d, t+2d, and t+5d to evaluate predictive value."
      />

      {events.length === 0 && !loading ? (
        <div style={{ padding: '24px 16px', fontSize: 11, color: C.muted, textAlign: 'center' }}>
          No spike events recorded for {ticker}
        </div>
      ) : (
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 8,
          padding: '12px 16px',
        }}>
          {events.map((ev, i) => (
            <SpikeCard key={`${ev.event_date}-${ev.event_type}-${i}`} event={ev} />
          ))}
        </div>
      )}
    </div>
  )
}
