import { C } from '../constants'

export default function SectionHeader({ title, subtitle, infoTitle, infoBody, infoExample, error, rightSlot }) {
  return (
    <div style={{
      background: C.surface2,
      borderBottom: `0.5px solid ${C.border}`,
      padding: '10px 16px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      position: 'relative',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{
          fontSize: 13,
          fontWeight: 600,
          color: C.subtle,
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
        }}>
          {title}
        </span>
        {error && (
          <span style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: C.red,
            flexShrink: 0,
            display: 'inline-block',
          }} />
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        {rightSlot}
        {subtitle && (
          <span style={{ fontSize: 11, color: C.muted }}>{subtitle}</span>
        )}
        <div style={{ position: 'relative', display: 'inline-flex' }} className="info-wrap">
          <div style={{
            width: 16,
            height: 16,
            borderRadius: '50%',
            border: `0.5px solid #2f2f2c`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'default',
            flexShrink: 0,
          }}>
            <span style={{ fontSize: 9, color: C.muted, lineHeight: 1 }}>i</span>
          </div>
          {/* opacity/pointer-events live in CSS so :hover can override them */}
          <div className="info-tooltip" style={{
            position: 'absolute',
            top: 'calc(100% + 6px)',
            right: 0,
            width: 268,
            background: '#1c1c1a',
            border: '0.5px solid #2f2f2c',
            borderRadius: 6,
            padding: '10px 12px',
            zIndex: 50,
          }}>
            {infoTitle && (
              <div style={{ fontSize: 11, fontWeight: 500, color: C.bright, marginBottom: 5 }}>
                {infoTitle}
              </div>
            )}
            {infoBody && (
              <div style={{ fontSize: 11, color: C.subtle, lineHeight: 1.65 }}>
                {infoBody}
              </div>
            )}
            {infoExample && (
              <div style={{
                fontSize: 11,
                color: C.green,
                background: C.greenBg,
                border: `0.5px solid ${C.greenBdr}`,
                borderRadius: 4,
                padding: '5px 8px',
                marginTop: 6,
                lineHeight: 1.55,
              }}>
                {infoExample}
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        .info-tooltip {
          opacity: 0;
          pointer-events: none;
          transition: opacity 0.15s;
        }
        .info-wrap:hover .info-tooltip {
          opacity: 1;
          pointer-events: auto;
        }
      `}</style>
    </div>
  )
}
