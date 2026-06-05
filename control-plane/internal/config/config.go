package config

import (
	"time"

	"github.com/spf13/viper"
)

type Config struct {
	HTTPPort       int           `mapstructure:"http_port"`
	GRPCPort       int           `mapstructure:"grpc_port"`
	InferenceSocket string       `mapstructure:"inference_socket"`
	ModelRepoPath  string        `mapstructure:"model_repo_path"`
	MaxModelMemory int64         `mapstructure:"max_model_memory_bytes"`
	DrainTimeout   time.Duration `mapstructure:"drain_timeout"`
	LogLevel       string        `mapstructure:"log_level"`
}

func Load() (*Config, error) {
	v := viper.New()

	v.SetDefault("http_port", 8080)
	v.SetDefault("grpc_port", 8001)
	v.SetDefault("inference_socket", "/run/inference.sock")
	v.SetDefault("model_repo_path", "/models")
	v.SetDefault("max_model_memory_bytes", 8*1024*1024*1024)
	v.SetDefault("drain_timeout", 30*time.Second)
	v.SetDefault("log_level", "info")

	v.SetConfigName("config")
	v.SetConfigType("yaml")
	v.AddConfigPath(".")

	v.AutomaticEnv()
	v.SetEnvPrefix("AXON")
	v.BindEnv("inference_socket", "INFERENCE_SOCKET")
	v.BindEnv("model_repo_path", "MODEL_REPO_PATH")
	v.BindEnv("max_model_memory_bytes", "MAX_MODEL_MEMORY_GB")
	v.BindEnv("http_port", "HTTP_PORT")
	v.BindEnv("grpc_port", "GRPC_PORT")
	v.BindEnv("drain_timeout", "DRAIN_TIMEOUT")

	_ = v.ReadInConfig()

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}
