import { C } from '../constants'

export default function TopBar() {
  return (
    <div style={{
      position: 'sticky',
      top: 0,
      zIndex: 20,
      background: C.surface,
      borderBottom: `0.5px solid ${C.border2}`,
      height: 52,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '0 16px',
    }}>
      <span style={{ fontSize: 18, fontWeight: 700, letterSpacing: '-0.02em' }}>
        <span style={{ color: C.bright }}>Fin</span>
        <span style={{ color: C.green }}>BERT</span>
        <span style={{ color: C.bright }}> Sentinel</span>
      </span>

      <div style={{
        position: 'absolute',
        right: 16,
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        background: C.greenBg,
        border: `0.5px solid ${C.greenBdr}`,
        borderRadius: 20,
        padding: '3px 9px',
        fontSize: 10,
        fontWeight: 500,
        color: C.green,
        letterSpacing: '0.04em',
        textTransform: 'uppercase',
      }}>
        <span style={{
          width: 5,
          height: 5,
          borderRadius: '50%',
          background: C.green,
          animation: 'pulse 2s infinite',
          flexShrink: 0,
        }} />
        Live
        <style>{`
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
          }
        `}</style>
      </div>
    </div>
  )
}
