import { C } from '../constants'

export default function TickerNav({ activeTicker, setActiveTicker, tickers }) {
  return (
    <div style={{
      position: 'sticky',
      top: 52,
      zIndex: 19,
      background: C.surface2,
      borderBottom: `0.5px solid ${C.border}`,
      padding: '10px 16px',
      display: 'flex',
      justifyContent: 'center',
      gap: 8,
    }}>
      {tickers.map(({ symbol, change }) => {
        const isActive  = symbol === activeTicker
        const isPos     = change != null && change >= 0
        const dotColor  = change == null ? C.subtle : isPos ? C.green : C.red
        const changeStr = change == null
          ? ''
          : `${change >= 0 ? '+' : ''}${change.toFixed(1)}%`

        return (
          <button
            key={symbol}
            onClick={() => setActiveTicker(symbol)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              borderRadius: 6,
              border: `0.5px solid ${isActive ? '#1a6644' : C.border3}`,
              padding: '7px 16px',
              fontSize: 13,
              fontWeight: 600,
              color: isActive ? C.bright : '#5a5a57',
              background: isActive ? C.greenBg : '#161614',
              cursor: 'pointer',
              transition: 'background 0.1s, border-color 0.1s, color 0.1s',
              outline: 'none',
            }}
            onMouseEnter={e => {
              if (!isActive) {
                e.currentTarget.style.background = '#1c1c1a'
                e.currentTarget.style.borderColor = '#2f2f2c'
                e.currentTarget.style.color = C.subtle
              }
            }}
            onMouseLeave={e => {
              if (!isActive) {
                e.currentTarget.style.background = '#161614'
                e.currentTarget.style.borderColor = C.border3
                e.currentTarget.style.color = '#5a5a57'
              }
            }}
          >
            <span style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: dotColor,
              opacity: isActive ? 1 : 0.5,
              flexShrink: 0,
            }} />
            {symbol}
            {changeStr && (
              <span style={{
                fontSize: 11,
                fontWeight: 400,
                color: isPos ? C.green : C.red,
                fontVariantNumeric: 'tabular-nums',
                letterSpacing: '-0.01em',
              }}>
                {changeStr}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}
