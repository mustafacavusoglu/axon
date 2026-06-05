package manager

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/mustafacavusoglu/axon/control-plane/internal/client"
)

type LifecycleManager struct {
	registry *ModelRegistry
	client   *client.InferenceClient
}

func NewLifecycleManager(registry *ModelRegistry, client *client.InferenceClient) *LifecycleManager {
	return &LifecycleManager{
		registry: registry,
		client:   client,
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

	err = m.client.LoadModel(context.Background(), name, uint32(version), filepath.Dir(modelPath))
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

			if err := m.LoadModel(modelName, version); err != nil {
				fmt.Printf("warning: failed to load %s:%d - %v\n", modelName, version, err)
			}
		}
	}

	return nil
}

func (m *LifecycleManager) resolveModelPath(name string, version int) string {
	base := filepath.Join("/models", name, strconv.Itoa(version), "model.onnx")
	if _, err := os.Stat(base); err == nil {
		return base
	}

	entries, err := filepath.Glob(filepath.Join("/models", name, strconv.Itoa(version), "*.onnx"))
	if err == nil && len(entries) > 0 {
		return entries[0]
	}

	return ""
}
