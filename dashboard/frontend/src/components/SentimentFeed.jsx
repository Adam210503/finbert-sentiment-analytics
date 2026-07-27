import { useCallback, useState } from 'react'
import { C } from '../constants'
import { api } from '../api'
import { usePolling } from '../hooks/usePolling'
import SectionHeader from './SectionHeader'

function SentimentPill({ label }) {
  const styles = {
    positive: { background: '#051510', color: C.green,  border: `0.5px solid ${C.greenBdr}` },
    negative: { background: C.redBg,  color: C.red,    border: `0.5px solid ${C.redBdr}`  },
    neutral:  { background: '#161614', color: C.subtle, border: `0.5px solid ${C.border3}` },
  }
  const s = styles[label] || styles.neutral
  return (
    <span style={{
      ...s,
      fontSize: 10,
      fontWeight: 500,
      letterSpacing: '0.04em',
      textTransform: 'uppercase',
      borderRadius: 3,
      padding: '2px 7px',
      display: 'inline-block',
      whiteSpace: 'nowrap',
    }}>
      {label}
    </span>
  )
}

function ReturnTag({ value }) {
  if (value == null) return null
  const isPos = value > 0
  const isNeg = value < 0
  const bg    = isPos ? '#051510' : isNeg ? C.redBg : '#161614'
  const col   = isPos ? C.green   : isNeg ? C.red   : C.muted
  const fmt   = `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}% next session`
  return (
    <span style={{
      fontSize: 10,
      borderRadius: 3,
      padding: '2px 5px',
      background: bg,
      color: col,
      fontVariantNumeric: 'tabular-nums',
      letterSpacing: '-0.01em',
    }}>
      {fmt}
    </span>
  )
}

function ConfidenceBar({ confidence, label }) {
  const fillColor = label === 'positive' ? '#1D9E75' : label === 'negative' ? '#C04830' : C.muted
  return (
    <div style={{ textAlign: 'right' }}>
      <div style={{
        fontSize: 11,
        fontWeight: 500,
        color: C.subtle,
        fontVariantNumeric: 'tabular-nums',
        letterSpacing: '-0.01em',
        marginBottom: 4,
      }}>
        {confidence != null ? (confidence * 100).toFixed(1) : '—'}
      </div>
      <div style={{ width: 64, height: 2, background: '#1c1c1a', borderRadius: 1, overflow: 'hidden', marginLeft: 'auto' }}>
        <div style={{
          width: `${(confidence || 0) * 100}%`,
          height: '100%',
          background: fillColor,
          borderRadius: 1,
        }} />
      </div>
    </div>
  )
}

function groupByDate(rows) {
  const groups = []
  let lastDate = null
  for (const row of rows) {
    if (row.market_date !== lastDate) {
      groups.push({ type: 'date', date: row.market_date })
      lastDate = row.market_date
    }
    groups.push({ type: 'row', ...row })
  }
  return groups
}

function FilterButton({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      style={{
        fontSize: 10,
        fontWeight: 500,
        letterSpacing: '0.04em',
        textTransform: 'uppercase',
        padding: '3px 9px',
        borderRadius: 3,
        border: `0.5px solid ${active ? '#1a6644' : '#222220'}`,
        background: active ? '#051510' : 'transparent',
        color: active ? C.green : C.muted,
        cursor: 'pointer',
        outline: 'none',
        transition: 'background 0.1s, border-color 0.1s, color 0.1s',
      }}
    >
      {children}
    </button>
  )
}

export default function SentimentFeed({ ticker }) {
  const [sentimentFilter, setSentimentFilter] = useState('all')
  const [dateFilter, setDateFilter] = useState('all')

  const fetchFn = useCallback(() => api.sentiment(ticker, 200), [ticker])
  const { data, error, loading } = usePolling(fetchFn, [ticker])

  const rows = data?.data ?? []

  const filtered = [...rows]
    .sort((a, b) => b.market_date.localeCompare(a.market_date))
    .filter(row => {
      if (sentimentFilter !== 'all' && row.sentiment_label !== sentimentFilter) return false
      if (dateFilter !== 'all') {
        const rowDate = new Date(row.market_date + 'T00:00:00')
        const cutoff = new Date()
        cutoff.setHours(0, 0, 0, 0)
        cutoff.setDate(cutoff.getDate() - (dateFilter === '7d' ? 6 : 29))
        if (rowDate < cutoff) return false
      }
      return true
    })

  const items = groupByDate(filtered)

  return (
    <div style={{ opacity: loading && !data ? 0.3 : 1 }}>
      <SectionHeader
        title="Scored Headlines"
        subtitle={`${ticker}`}
        error={error}
        rightSlot={
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {[
              { color: '#1D9E75', label: 'Positive' },
              { color: '#C04830', label: 'Negative' },
              { color: C.muted,   label: 'Neutral'  },
            ].map(({ color, label }) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <div style={{ width: 18, height: 2, background: color, borderRadius: 1 }} />
                <span style={{ fontSize: 10, color: C.muted, letterSpacing: '0.02em' }}>{label}</span>
              </div>
            ))}
          </div>
        }
        infoTitle="Scored headlines"
        infoBody="Each headline is passed through the fine-tuned FinBERT checkpoint individually. A sentiment label and confidence score are assigned per article. Multiple articles published on the same date share the same next-session return figure, as that figure reflects the market's aggregate response to all available information on that date rather than any single headline."
        infoExample={`All ${ticker} articles dated Jun 27 display +1.14% because that is what ${ticker} returned on Jun 28`}
      />

      {/* count + colour legend */}
      <div style={{
        background: C.surface2,
        borderBottom: `0.5px solid #111110`,
        padding: '6px 16px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <span style={{ fontSize: 11, color: C.muted }}>All headlines · most recent first</span>
        <span style={{ fontSize: 11, color: C.subtle, fontVariantNumeric: 'tabular-nums' }}>
          {rows.length} total
        </span>
      </div>

      {/* filter bar */}
      <div style={{
        background: '#0f0f0e',
        borderBottom: '0.5px solid #111110',
        padding: '7px 16px',
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        flexWrap: 'wrap',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ fontSize: 10, color: C.muted, letterSpacing: '0.04em', textTransform: 'uppercase', marginRight: 4 }}>Sentiment</span>
          {['all', 'positive', 'neutral', 'negative'].map(s => (
            <FilterButton key={s} active={sentimentFilter === s} onClick={() => setSentimentFilter(s)}>
              {s === 'all' ? 'All' : s}
            </FilterButton>
          ))}
        </div>
        <div style={{ width: '0.5px', height: 16, background: '#222220' }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ fontSize: 10, color: C.muted, letterSpacing: '0.04em', textTransform: 'uppercase', marginRight: 4 }}>Period</span>
          {[['all', 'All'], ['7d', '7d'], ['30d', '30d']].map(([val, lbl]) => (
            <FilterButton key={val} active={dateFilter === val} onClick={() => setDateFilter(val)}>
              {lbl}
            </FilterButton>
          ))}
        </div>
        <span style={{ marginLeft: 'auto', fontSize: 10, color: C.muted, fontVariantNumeric: 'tabular-nums' }}>
          {filtered.length} of {rows.length}
        </span>
      </div>

      {/* column headers */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'auto 1fr 90px',
        gap: '16px',
        padding: '5px 16px',
        background: '#0f0f0e',
        borderBottom: '0.5px solid #111110',
      }}>
        <span style={{ fontSize: 10, color: C.muted, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Sentiment</span>
        <span style={{ fontSize: 10, color: C.muted, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Headline</span>
        <span style={{ fontSize: 10, color: C.muted, letterSpacing: '0.06em', textTransform: 'uppercase', textAlign: 'right' }}>Confidence</span>
      </div>

      <div
        className="feed-scroll"
        style={{
          maxHeight: 340,
          overflowY: 'auto',
          scrollbarWidth: 'thin',
          scrollbarColor: '#2f2f2c #0c0c0b',
        }}
      >
        {items.map((item, i) => {
          if (item.type === 'date') {
            return (
              <div key={`d-${item.date}-${i}`} style={{
                padding: '5px 16px',
                fontSize: 10,
                color: C.muted,
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
                background: C.surface2,
                borderBottom: '0.5px solid #111110',
              }}>
                {item.date}
              </div>
            )
          }

          const row = item

          // Bug 2: strict URL guard — only link when a real http URL is present
          const hasUrl = row.url !== null &&
                         row.url !== undefined &&
                         typeof row.url === 'string' &&
                         row.url.trim().length > 0 &&
                         row.url.startsWith('http')

          return (
            <div
              key={row.headline_hash || `r-${i}`}
              style={{
                display: 'grid',
                // Bug 1: first column uses auto so pill takes its natural width,
                // gap increased to 16px for clear visual separation
                gridTemplateColumns: 'auto 1fr 90px',
                gap: '16px',
                padding: '12px 16px',
                borderBottom: '0.5px solid #111110',
                cursor: 'pointer',
                transition: 'background 0.1s',
              }}
              className="feed-row"
            >
              {/* Bug 1: flex wrapper keeps pill left-aligned with top padding */}
              <div style={{ display: 'flex', justifyContent: 'flex-start', paddingTop: '2px' }}>
                <SentimentPill label={row.sentiment_label} />
              </div>

              {/* Bug 1: minWidth 0 prevents headline from overflowing the grid cell */}
              <div style={{ minWidth: 0 }}>
                <div style={{ marginBottom: 3 }}>
                  {hasUrl ? (
                    <a
                      href={row.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{
                        color: '#c8c8c4',
                        textDecoration: 'none',
                        borderBottom: '0.5px solid #3a3a38',
                        fontSize: '13px',
                        lineHeight: '1.5',
                        cursor: 'pointer',
                        display: 'block',
                      }}
                      onMouseEnter={e => {
                        e.currentTarget.style.color = '#e8e8e6'
                        e.currentTarget.style.borderBottomColor = '#888884'
                      }}
                      onMouseLeave={e => {
                        e.currentTarget.style.color = '#c8c8c4'
                        e.currentTarget.style.borderBottomColor = '#3a3a38'
                      }}
                    >
                      {row.headline}
                    </a>
                  ) : (
                    <span style={{ color: '#c8c8c4', fontSize: '13px', lineHeight: '1.5', display: 'block' }}>
                      {row.headline}
                    </span>
                  )}
                </div>
                <div style={{
                  fontSize: 11,
                  color: C.muted,
                  display: 'flex',
                  gap: 5,
                  flexWrap: 'wrap',
                  alignItems: 'center',
                }}>
                  <span>{row.source}</span>
                  <span>·</span>
                  <span>{row.market_date}</span>
                  {row.log_return != null && (
                    <>
                      <span>·</span>
                      <ReturnTag value={row.log_return} />
                    </>
                  )}
                </div>
              </div>

              <div style={{ paddingTop: 1 }}>
                <ConfidenceBar confidence={row.confidence} label={row.sentiment_label} />
              </div>
            </div>
          )
        })}

        {rows.length === 0 && !loading && (
          <div style={{ padding: '24px 16px', fontSize: 12, color: C.muted, textAlign: 'center' }}>
            No scored headlines for {ticker}
          </div>
        )}
      </div>

      <style>{`
        .feed-row:hover { background: #111110; }
      `}</style>
    </div>
  )
}
