const BASE = import.meta.env.VITE_API_URL

const get = (path) => fetch(`${BASE}${path}`).then(r => {
  if (!r.ok) throw new Error(`${r.status}`)
  return r.json()
})

export const api = {
  sentiment:   (ticker, limit = 200) => get(`/sentiment?ticker=${ticker}&limit=${limit}`),
  correlation: (ticker, window = 7)  => get(`/correlation?ticker=${ticker}&window=${window}`),
  events:      (ticker)              => get(`/events?ticker=${ticker}`),
  health:      ()                    => get('/health'),
  prices:      ()                    => get('/prices'),
}
