package manager

import (
	"sync"
	"time"
)

type CircuitState int

const (
	CircuitClosed CircuitState = iota
	CircuitOpen
)

type CircuitBreaker struct {
	mu          sync.Mutex
	failures    map[string]int
	failTimes   map[string]time.Time
	maxFailures int
	resetAfter  time.Duration
}

func NewCircuitBreaker(maxFailures int, resetAfter time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		failures:    make(map[string]int),
		failTimes:   make(map[string]time.Time),
		maxFailures: maxFailures,
		resetAfter:  resetAfter,
	}
}

func (cb *CircuitBreaker) RecordFailure(key string) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.failures[key]++
	cb.failTimes[key] = time.Now()
}

func (cb *CircuitBreaker) RecordSuccess(key string) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	delete(cb.failures, key)
	delete(cb.failTimes, key)
}

func (cb *CircuitBreaker) State(key string) CircuitState {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	if cb.failures[key] >= cb.maxFailures {
		if t, ok := cb.failTimes[key]; ok {
			if time.Since(t) < cb.resetAfter {
				return CircuitOpen
			}
			delete(cb.failures, key)
			delete(cb.failTimes, key)
		}
	}
	return CircuitClosed
}
