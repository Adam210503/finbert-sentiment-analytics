import { useState, useEffect, useCallback } from 'react'

export function usePolling(fetchFn, deps = [], interval = 120000) {
  const [data, setData]       = useState(null)
  const [error, setError]     = useState(false)
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    try {
      setData(await fetchFn())
      setError(false)
    } catch {
      setError(true)
    } finally {
      setLoading(false)
    }
  }, deps) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    load()
    const id = setInterval(load, interval)
    return () => clearInterval(id)
  }, [load])

  return { data, error, loading }
}
