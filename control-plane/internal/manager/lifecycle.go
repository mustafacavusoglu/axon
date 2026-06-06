package manager

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/mustafacavusoglu/axon/control-plane/internal/client"
	zaplog "github.com/mustafacavusoglu/axon/control-plane/internal/log"
)

type LifecycleManager struct {
	registry *ModelRegistry
	client   *client.InferenceClient
	repoPath string
	breaker  *CircuitBreaker
}

func NewLifecycleManager(registry *ModelRegistry, client *client.InferenceClient, repoPath string) *LifecycleManager {
	return &LifecycleManager{
		registry: registry,
		client:   client,
		repoPath: repoPath,
		breaker:  NewCircuitBreaker(3, 5*time.Minute),
	}
}

func (m *LifecycleManager) LoadModel(name string, version int) error {
	modelPath := m.resolveModelPath(name, version)
	if modelPath == "" {
		return fmt.Errorf("model path not found for %s:%d", name, version)
	}

	configPath := filepath.Join(filepath.Dir(filepath.Dir(modelPath)), "config.pbtxt")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to read config for %s: %w", name, err)
	}

	config, err := ParseModelConfig(configData)
	if err != nil {
		return fmt.Errorf("failed to parse config for %s: %w", name, err)
	}

	m.registry.Set(name, version, config)

	var concurrency uint32 = 1
	if len(config.InstanceGroups) > 0 && config.InstanceGroups[0].Count > 0 {
		concurrency = uint32(config.InstanceGroups[0].Count)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	zaplog.L.Infow("sending LoadModel to engine",
		"model", name, "version", version,
		"path", filepath.Dir(modelPath), "concurrency", concurrency,
	)
	err = m.client.LoadModel(ctx, name, uint32(version), filepath.Dir(modelPath), concurrency)
	if err != nil {
		m.registry.MarkError(name, version)
		return fmt.Errorf("failed to load model %s on engine: %w", name, err)
	}

	m.registry.MarkReady(name, version)
	return nil
}

func (m *LifecycleManager) UnloadModel(name string, version int) error {
	m.registry.MarkUnloading(name, version)

	err := m.client.UnloadModel(context.Background(), name, uint32(version))
	if err != nil {
		return fmt.Errorf("failed to unload model %s from engine: %w", name, err)
	}

	m.registry.Remove(name, version)
	return nil
}

func (m *LifecycleManager) LoadAllFromRepo(repoPath string) error {
	entries, err := os.ReadDir(repoPath)
	if err != nil {
		return fmt.Errorf("failed to read model repo: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		modelName := entry.Name()
		modelDir := filepath.Join(repoPath, modelName)

		configPath := filepath.Join(modelDir, "config.pbtxt")
		if _, err := os.Stat(configPath); os.IsNotExist(err) {
			continue
		}

		versions, err := filepath.Glob(filepath.Join(modelDir, "*", "model.onnx"))
		if err != nil {
			continue
		}

		for _, onnxPath := range versions {
			verDir := filepath.Dir(onnxPath)
			verStr := filepath.Base(verDir)
			version, err := strconv.Atoi(verStr)
			if err != nil {
				continue
			}

			circuitKey := fmt.Sprintf("%s@v%d", modelName, version)
			if m.breaker.State(circuitKey) == CircuitOpen {
				zaplog.L.Warnw("skipping model — circuit open",
					"model", modelName, "version", version,
				)
				continue
			}

			var loadErr error
			for retry := 0; retry < 3; retry++ {
				if retry > 0 {
					time.Sleep(2 * time.Second)
				}
				loadErr = m.LoadModel(modelName, version)
				if loadErr == nil {
					m.breaker.RecordSuccess(circuitKey)
					break
				}
				zaplog.L.Warnw("model load failed, retrying",
					"model", modelName, "version", version,
					"attempt", retry+1, "error", loadErr,
				)
			}
			if loadErr != nil {
				m.breaker.RecordFailure(circuitKey)
				zaplog.L.Errorw("model load failed after retries",
					"model", modelName, "version", version, "error", loadErr,
				)
			}
		}
	}

	return nil
}

func (m *LifecycleManager) resolveModelPath(name string, version int) string {
	base := filepath.Join(m.repoPath, name, strconv.Itoa(version), "model.onnx")
	if _, err := os.Stat(base); err == nil {
		return base
	}

	entries, err := filepath.Glob(filepath.Join(m.repoPath, name, strconv.Itoa(version), "*.onnx"))
	if err == nil && len(entries) > 0 {
		return entries[0]
	}

	return ""
}
