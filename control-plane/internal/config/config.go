package config

import (
	"os"
	"strconv"
	"time"
)

type Config struct {
	HTTPPort        int           `json:"http_port"`
	GRPCPort        int           `json:"grpc_port"`
	InferenceSocket string        `json:"inference_socket"`
	ModelRepoPath   string        `json:"model_repo_path"`
	MaxModelMemory  int64         `json:"max_model_memory_bytes"`
	DrainTimeout    time.Duration `json:"drain_timeout"`
	LogLevel        string        `json:"log_level"`
}

func Load() (*Config, error) {
	cfg := &Config{
		HTTPPort:        8080,
		GRPCPort:        8001,
		InferenceSocket: "/run/inference.sock",
		ModelRepoPath:   "/models",
		MaxModelMemory:  8 * 1024 * 1024 * 1024,
		DrainTimeout:    30 * time.Second,
		LogLevel:        "info",
	}

	if v := os.Getenv("HTTP_PORT"); v != "" {
		cfg.HTTPPort, _ = strconv.Atoi(v)
	}
	if v := os.Getenv("GRPC_PORT"); v != "" {
		cfg.GRPCPort, _ = strconv.Atoi(v)
	}
	if v := os.Getenv("INFERENCE_SOCKET"); v != "" {
		cfg.InferenceSocket = v
	}
	if v := os.Getenv("MODEL_REPO_PATH"); v != "" {
		cfg.ModelRepoPath = v
	}
	if v := os.Getenv("MAX_MODEL_MEMORY_GB"); v != "" {
		gb, _ := strconv.ParseInt(v, 10, 64)
		cfg.MaxModelMemory = gb * 1024 * 1024 * 1024
	}
	if v := os.Getenv("DRAIN_TIMEOUT"); v != "" {
		d, _ := time.ParseDuration(v)
		if d > 0 {
			cfg.DrainTimeout = d
		}
	}

	return cfg, nil
}
