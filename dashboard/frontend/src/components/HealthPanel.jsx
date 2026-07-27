import { C } from '../constants'
import { api } from '../api'
import { usePolling } from '../hooks/usePolling'
import SectionHeader from './SectionHeader'

function tradingDaysAgo(dateStr) {
  if (!dateStr) return Infinity
  const target = new Date(dateStr + 'T00:00:00')
  const today  = new Date()
  today.setHours(0, 0, 0, 0)
  let count = 0
  let d = new Date(today)
  while (d > target) {
    d.setDate(d.getDate() - 1)
    const dow = d.getDay()
    if (dow !== 0 && dow !== 6) count++
  }
  return count
}

function hoursSince(tsStr) {
  if (!tsStr) return Infinity
  const ts = new Date(tsStr)
  return (Date.now() - ts.getTime()) / 3600000
}

function StatusBadge({ status, label }) {
  const styles = {
    green: { background: '#051510', color: C.green,  border: `0.5px solid ${C.greenBdr}` },
    amber: { background: C.warnBg,  color: C.warn,   border: `0.5px solid ${C.warnBdr}`  },
    red:   { background: C.redBg,   color: C.red,    border: `0.5px solid ${C.redBdr}`   },
  }
  const s = styles[status] || styles.green
  return (
    <span style={{
      ...s,
      fontSize: 10,
      padding: '2px 7px',
      borderRadius: 3,
      letterSpacing: '0.04em',
      textTransform: 'uppercase',
      fontWeight: 500,
    }}>
      {label}
    </span>
  )
}

function HealthRow({ label, value, statusColor, tooltipTitle, tooltipBody, tooltipStatus, tooltipStatusLabel }) {
  const dotColor = statusColor === 'green' ? C.green : statusColor === 'amber' ? C.amber : C.red

  return (
    <div
      className="hrow"
      style={{
        display: 'flex',
        alignItems: 'center',
        padding: '8px 16px',
        borderBottom: `0.5px solid #111110`,
        position: 'relative',
        transition: 'background 0.1s',
        cursor: 'default',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1 }}>
        <span style={{
          width: 6, height: 6,
          borderRadius: '50%',
          background: dotColor,
          flexShrink: 0,
          display: 'inline-block',
        }} />
        <span style={{ fontSize: 12, color: C.text }}>{label}</span>
      </div>
      <span style={{ fontSize: 12, color: C.subtle, fontVariantNumeric: 'tabular-nums', letterSpacing: '-0.01em' }}>
        {value}
      </span>

      {/* opacity/pointer-events in CSS so :hover can override them */}
      <div className="htip" style={{
        position: 'absolute',
        bottom: 'calc(100% + 4px)',
        right: 16,
        width: 280,
        background: '#1c1c1a',
        border: '0.5px solid #2f2f2c',
        borderRadius: 6,
        padding: '10px 12px',
        zIndex: 50,
      }}>
        <div style={{ fontSize: 11, fontWeight: 500, color: C.bright, marginBottom: 5 }}>
          {tooltipTitle}
        </div>
        <div style={{ fontSize: 11, color: C.subtle, lineHeight: 1.65, marginBottom: 6 }}>
          {tooltipBody}
        </div>
        <StatusBadge status={tooltipStatus} label={tooltipStatusLabel} />
      </div>

      <style>{`
        .htip { opacity: 0; pointer-events: none; transition: opacity 0.15s; }
        .hrow:hover { background: #111110; }
        .hrow:hover .htip { opacity: 1; pointer-events: auto; }
      `}</style>
    </div>
  )
}

export default function HealthPanel() {
  const { data, error, loading } = usePolling(() => api.health(), [])

  const h = data

  const backlog = h?.headlines_unscored ?? 0
  const backlogStatus = backlog === 0 ? 'green' : backlog <= 20 ? 'amber' : 'red'
  const backlogLabel  = backlog === 0 ? 'No backlog' : `${backlog} pending`

  const priceDaysAgo = tradingDaysAgo(h?.latest_price_date)
  const priceStatus  = priceDaysAgo <= 2 ? 'green' : priceDaysAgo <= 5 ? 'amber' : 'red'
  const priceLabel   = priceDaysAgo === Infinity ? 'No data' : priceDaysAgo === 0 ? 'Today' : `${priceDaysAgo}d ago`

  const recentJobs    = h?.recent_jobs ?? []
  const lastNewsJob   = recentJobs.find(j => j.job_name?.includes('news'))
  const newsHours     = lastNewsJob ? hoursSince(lastNewsJob.ran_at) : Infinity
  const newsStatus    = newsHours < 4 ? 'green' : newsHours < 5 ? 'amber' : 'red'
  const newsLabel     = newsHours === Infinity ? 'Unknown' : `${newsHours.toFixed(1)}h ago`

  const sqliteLabel = h
    ? `${h.headlines_total ?? 0} headlines · ${h.total_price_records ?? 0} prices`
    : '—'

  return (
    <div style={{ opacity: loading && !data ? 0.3 : 1 }}>
      <SectionHeader
        title="System Health"
        error={error}
        infoTitle="Pipeline status"
        infoBody="Live status of all pipeline components. Rows turn amber or red when thresholds are exceeded. Hover any row for details."
      />

      <HealthRow
        label="FinBERT checkpoint"
        value="finetuned-optuna"
        statusColor="green"
        tooltipTitle="FinBERT checkpoint"
        tooltipBody="The finetuned-optuna checkpoint is loaded. A macro F1 of 0.9201 was recorded on the held-out test set, compared to 0.8168 for the base model. Best trial: lr 4.49e-5, wd 0.084, 3 epochs."
        tooltipStatus="green"
        tooltipStatusLabel="Optimal"
      />
      <HealthRow
        label="Scoring backlog"
        value={`${backlog} unscored`}
        statusColor={backlogStatus}
        tooltipTitle="Scoring backlog"
        tooltipBody="All collected headlines have been scored. The scoring job runs at 2-hour intervals and processes all rows with a null sentiment label in the sentiment_scores table."
        tooltipStatus={backlogStatus}
        tooltipStatusLabel={backlogLabel}
      />
      <HealthRow
        label="Latest price date"
        value={h?.latest_price_date ?? '—'}
        statusColor={priceStatus}
        tooltipTitle="Latest price date"
        tooltipBody="The most recent record in price_data. If this falls more than 2 trading days behind today, the correlation chart may reflect stale data. The price job runs only during NYSE hours (09:30 to 16:00 EST)."
        tooltipStatus={priceStatus}
        tooltipStatusLabel={priceLabel}
      />
      <HealthRow
        label="Last news fetch"
        value={newsLabel}
        statusColor={newsStatus}
        tooltipTitle="Last news fetch"
        tooltipBody="The most recent successful completion of the news ingestion job. The job runs at 4-hour intervals. If this exceeds 5 hours, the scheduler may have stopped."
        tooltipStatus={newsStatus}
        tooltipStatusLabel={newsHours < 4 ? 'On schedule' : newsHours < 5 ? 'Slightly late' : 'Overdue'}
      />
      <HealthRow
        label="Headlines total"
        value={h?.headlines_total ?? '—'}
        statusColor="green"
        tooltipTitle="Headlines total"
        tooltipBody="Total rows in sentiment_scores across all tickers."
        tooltipStatus="green"
        tooltipStatusLabel="Operational"
      />
      <HealthRow
        label="Latest scored date"
        value={h?.latest_scored_date ?? '—'}
        statusColor="green"
        tooltipTitle="Latest scored date"
        tooltipBody="The most recent market_date with at least one scored headline in sentiment_scores."
        tooltipStatus="green"
        tooltipStatusLabel="Operational"
      />
      <HealthRow
        label="SQLite"
        value={sqliteLabel}
        statusColor="green"
        tooltipTitle="SQLite"
        tooltipBody={`Row counts across all tables: sentiment_scores, price_data, correlations, spike_events, job_log. Tickers tracked: ${(h?.tickers ?? []).join(', ') || '—'}.`}
        tooltipStatus="green"
        tooltipStatusLabel="Operational"
      />
    </div>
  )
}
