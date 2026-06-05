package manager

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

type ModelState int

const (
	StateLoading ModelState = iota
	StateReady
	StateUnloading
	StateError
)

func (s ModelState) String() string {
	switch s {
	case StateLoading:
		return "LOADING"
	case StateReady:
		return "READY"
	case StateUnloading:
		return "UNLOADING"
	case StateError:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

type ModelEntry struct {
	Config   *ModelConfig
	State    ModelState
	LoadedAt time.Time
	LastUsed time.Time
	Version  int
}

type ModelRegistry struct {
	mu     sync.RWMutex
	models map[string]*ModelEntry
}

func NewModelRegistry() *ModelRegistry {
	return &ModelRegistry{
		models: make(map[string]*ModelEntry),
	}
}

func modelKey(name string, version int) string {
	return fmt.Sprintf("%s@v%d", name, version)
}

func (r *ModelRegistry) Set(name string, version int, config *ModelConfig) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := modelKey(name, version)
	r.models[key] = &ModelEntry{
		Config:   config,
		State:    StateLoading,
		Version:  version,
		LoadedAt: time.Now(),
		LastUsed: time.Now(),
	}
}

func (r *ModelRegistry) MarkReady(name string, version int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := modelKey(name, version)
	if entry, ok := r.models[key]; ok {
		entry.State = StateReady
	}
}

func (r *ModelRegistry) MarkError(name string, version int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := modelKey(name, version)
	if entry, ok := r.models[key]; ok {
		entry.State = StateError
	}
}

func (r *ModelRegistry) MarkUnloading(name string, version int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := modelKey(name, version)
	if entry, ok := r.models[key]; ok {
		entry.State = StateUnloading
	}
}

func (r *ModelRegistry) Remove(name string, version int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := modelKey(name, version)
	delete(r.models, key)
}

func (r *ModelRegistry) Get(name string, version int) (*ModelEntry, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	key := modelKey(name, version)
	entry, ok := r.models[key]
	return entry, ok
}

func (r *ModelRegistry) Touch(name string, version int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := modelKey(name, version)
	if entry, ok := r.models[key]; ok {
		entry.LastUsed = time.Now()
	}
}

func (r *ModelRegistry) List() []*ModelEntry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var entries []*ModelEntry
	for _, e := range r.models {
		entries = append(entries, e)
	}
	return entries
}

func (r *ModelRegistry) GetLatestVersion(name string) (*ModelEntry, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var best *ModelEntry
	var bestV int
	for _, e := range r.models {
		if e.Config != nil && e.Config.Name == name && e.State == StateReady {
			if best == nil || e.Version > bestV {
				best = e
				bestV = e.Version
			}
		}
	}
	return best, best != nil
}

func (r *ModelRegistry) GetAvailableVersions(name string) []int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var versions []int
	for _, e := range r.models {
		if e.Config != nil && e.Config.Name == name && e.State == StateReady {
			versions = append(versions, e.Version)
		}
	}
	sort.Ints(versions)
	return versions
}

func (r *ModelRegistry) AllNames() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	seen := make(map[string]bool)
	var names []string
	for _, e := range r.models {
		if e.Config != nil && e.State == StateReady {
			if !seen[e.Config.Name] {
				seen[e.Config.Name] = true
				names = append(names, e.Config.Name)
			}
		}
	}
	sort.Strings(names)
	return names
}
