import { C } from '../constants'
import SentimentFeed from './SentimentFeed'
import CorrelationChart from './CorrelationChart'
import SpikeEvents from './SpikeEvents'
import HealthPanel from './HealthPanel'

const section = { borderBottom: `0.5px solid ${C.border}` }

export default function TickerPage({ ticker }) {
  return (
    <div>
      <div style={section}>
        <SentimentFeed ticker={ticker} />
      </div>
      <div style={section}>
        <CorrelationChart ticker={ticker} />
      </div>
      <div style={section}>
        <SpikeEvents ticker={ticker} />
      </div>
      <div style={section}>
        <HealthPanel />
      </div>
    </div>
  )
}
