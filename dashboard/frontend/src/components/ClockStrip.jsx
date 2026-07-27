import { useState, useEffect } from 'react'
import { C } from '../constants'

function formatTime(date, tz) {
  return date.toLocaleString('en-GB', {
    timeZone: tz,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

function formatDate(date, tz) {
  return date.toLocaleString('en-GB', {
    timeZone: tz,
    weekday: 'short',
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  })
}

function getMarketStatus(now) {
  const nyDate = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }))
  const dow = nyDate.getDay() // 0=Sun, 6=Sat
  const h   = nyDate.getHours()
  const m   = nyDate.getMinutes()
  const mins = h * 60 + m

  const open  = 9 * 60 + 30
  const close = 16 * 60

  if (dow === 0 || dow === 6) {
    const daysUntilMon = dow === 6 ? 2 : 1
    return { state: 'weekend', daysUntilOpen: daysUntilMon }
  }

  if (mins >= open && mins < close) {
    const remaining = close - mins
    return { state: 'open', remainingMins: remaining }
  }

  if (mins < open) {
    return { state: 'premarket', minsUntilOpen: open - mins }
  }

  return { state: 'afterhours' }
}

function fmtDuration(mins) {
  const h = Math.floor(mins / 60)
  const m = mins % 60
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}

export default function ClockStrip() {
  const [now, setNow] = useState(new Date())

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  const status = getMarketStatus(now)

  let dot, label, labelColor, detail
  switch (status.state) {
    case 'open':
      dot = C.green; label = 'Open'; labelColor = C.green
      detail = `Closes 16:00 EST · ${fmtDuration(status.remainingMins)} remaining`
      break
    case 'premarket':
      dot = C.red; label = 'Closed'; labelColor = C.red
      detail = `Pre-market · opens in ${fmtDuration(status.minsUntilOpen)}`
      break
    case 'afterhours':
      dot = C.red; label = 'Closed'; labelColor = C.red
      detail = 'After hours · opens tomorrow 09:30 EST'
      break
    case 'weekend':
      dot = C.red; label = 'Closed'; labelColor = C.red
      detail = 'Weekend · opens Monday 09:30 EST'
      break
  }

  const col = {
    width: 180,
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
  }

  const timeStyle = {
    fontSize: 15,
    fontWeight: 500,
    color: C.text,
    fontVariantNumeric: 'tabular-nums',
    letterSpacing: '-0.01em',
  }

  const dateLabelStyle = {
    fontSize: 10,
    color: C.muted,
    letterSpacing: '0.04em',
    textTransform: 'uppercase',
  }

  const divider = {
    width: '0.5px',
    background: C.border,
    alignSelf: 'stretch',
    margin: '0 12px',
  }

  return (
    <div style={{
      background: C.surface2,
      borderBottom: `0.5px solid ${C.border}`,
      padding: '7px 16px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    }}>
      <div style={col}>
        <div style={dateLabelStyle}>Singapore (SGT)</div>
        <div style={timeStyle}>{formatTime(now, 'Asia/Singapore')}</div>
        <div style={{ fontSize: 11, color: C.subtle }}>{formatDate(now, 'Asia/Singapore')}</div>
      </div>

      <div style={divider} />

      <div style={col}>
        <div style={dateLabelStyle}>New York (NYSE)</div>
        <div style={timeStyle}>{formatTime(now, 'America/New_York')}</div>
        <div style={{ fontSize: 11, color: C.subtle }}>{formatDate(now, 'America/New_York')}</div>
      </div>

      <div style={divider} />

      <div style={{ ...col, gap: 3 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: dot, flexShrink: 0 }} />
          <span style={{ fontSize: 12, fontWeight: 600, color: labelColor }}>{label}</span>
        </div>
        <div style={{ fontSize: 11, color: C.subtle }}>{detail}</div>
      </div>
    </div>
  )
}
