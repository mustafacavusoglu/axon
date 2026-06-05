package manager

import (
	"fmt"
	"strconv"
	"strings"
)

type DataType int32

const (
	DTInvalid DataType = iota
	DTFP32
	DTFP64
	DTINT32
	DTINT64
	DTINT8
	DTUINT8
	DTBOOL
	DTSTRING
)

var dtypeNames = map[string]DataType{
	"TYPE_FP32":   DTFP32,
	"TYPE_FP64":   DTFP64,
	"TYPE_INT32":  DTINT32,
	"TYPE_INT64":  DTINT64,
	"TYPE_INT8":   DTINT8,
	"TYPE_UINT8":  DTUINT8,
	"TYPE_BOOL":   DTBOOL,
	"TYPE_STRING": DTSTRING,
}

type TensorDef struct {
	Name     string
	DataType DataType
	Dims     []int64
}

type VersionPolicy struct {
	Type        string
	NumVersions int32
	Versions    []int32
}

type InstanceGroup struct {
	Count int32
	Kind  string
	CpuSet string
}

type DynamicBatching struct {
	PreferredBatchSize []int32
	MaxQueueDelayUs    int64
}

type ModelConfig struct {
	Name            string
	Platform        string
	MaxBatchSize    int32
	Inputs          []TensorDef
	Outputs         []TensorDef
	InstanceGroups  []InstanceGroup
	DynamicBatching *DynamicBatching
	VersionPolicy   *VersionPolicy
}

func ParseModelConfig(content []byte) (*ModelConfig, error) {
	cfg := &ModelConfig{
		Platform:     "onnxruntime_onnx",
		MaxBatchSize: 1,
	}
	lines := strings.Split(string(content), "\n")

	var currentSection string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		line = strings.TrimRight(line, "{")
		line = strings.TrimSpace(line)

		switch {
		case strings.HasPrefix(line, "name:"):
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				cfg.Name = strings.Trim(strings.TrimSpace(parts[1]), "\"")
			}

		case strings.HasPrefix(line, "platform:"):
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				cfg.Platform = strings.Trim(strings.TrimSpace(parts[1]), "\"")
			}

		case strings.HasPrefix(line, "max_batch_size:"):
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				v, _ := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 32)
				cfg.MaxBatchSize = int32(v)
			}

		case strings.HasPrefix(line, "input"):
			currentSection = "input"
		case strings.HasPrefix(line, "output"):
			currentSection = "output"
		case strings.HasPrefix(line, "instance_group"):
			currentSection = "instance_group"
		case strings.HasPrefix(line, "dynamic_batching"):
			currentSection = "dynamic_batching"
		case strings.HasPrefix(line, "version_policy"):
			currentSection = "version_policy"

		case currentSection == "input" && strings.HasPrefix(line, "name:"):
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				name := strings.Trim(strings.TrimSpace(parts[1]), "\"")
				cfg.Inputs = append(cfg.Inputs, TensorDef{Name: name})
			}

		case currentSection == "input" && strings.HasPrefix(line, "data_type:"):
			if len(cfg.Inputs) > 0 {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 {
					dt := strings.TrimSpace(parts[1])
					if t, ok := dtypeNames[dt]; ok {
						cfg.Inputs[len(cfg.Inputs)-1].DataType = t
					}
				}
			}

		case currentSection == "input" && strings.HasPrefix(line, "dims:"):
			if len(cfg.Inputs) > 0 {
				cfg.Inputs[len(cfg.Inputs)-1].Dims = parseDims(line)
			}

		case currentSection == "output" && strings.HasPrefix(line, "name:"):
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				name := strings.Trim(strings.TrimSpace(parts[1]), "\"")
				cfg.Outputs = append(cfg.Outputs, TensorDef{Name: name})
			}

		case currentSection == "output" && strings.HasPrefix(line, "data_type:"):
			if len(cfg.Outputs) > 0 {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 {
					dt := strings.TrimSpace(parts[1])
					if t, ok := dtypeNames[dt]; ok {
						cfg.Outputs[len(cfg.Outputs)-1].DataType = t
					}
				}
			}

		case currentSection == "output" && strings.HasPrefix(line, "dims:"):
			if len(cfg.Outputs) > 0 {
				cfg.Outputs[len(cfg.Outputs)-1].Dims = parseDims(line)
			}

		case currentSection == "instance_group" && strings.HasPrefix(line, "count:"):
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				v, _ := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 32)
				cfg.InstanceGroups = append(cfg.InstanceGroups, InstanceGroup{Count: int32(v), Kind: "KIND_CPU"})
			}

		case currentSection == "instance_group" && strings.HasPrefix(line, "kind:"):
			if len(cfg.InstanceGroups) > 0 {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 {
					cfg.InstanceGroups[len(cfg.InstanceGroups)-1].Kind = strings.Trim(strings.TrimSpace(parts[1]), "\"")
				}
			}

		case currentSection == "dynamic_batching" && strings.HasPrefix(line, "preferred_batch_size:"):
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				cfg.DynamicBatching = &DynamicBatching{}
				nums := parseNumList(parts[1])
				for _, n := range nums {
					cfg.DynamicBatching.PreferredBatchSize = append(cfg.DynamicBatching.PreferredBatchSize, int32(n))
				}
			}

		case currentSection == "dynamic_batching" && strings.HasPrefix(line, "max_queue_delay_microseconds:"):
			parts := strings.SplitN(line, ":", 2)
			if cfg.DynamicBatching != nil && len(parts) == 2 {
				v, _ := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
				cfg.DynamicBatching.MaxQueueDelayUs = v
			}

		case currentSection == "version_policy" && strings.Contains(line, "latest"):
			cfg.VersionPolicy = &VersionPolicy{Type: "latest", NumVersions: 1}
		case currentSection == "version_policy" && strings.Contains(line, "all"):
			cfg.VersionPolicy = &VersionPolicy{Type: "all"}
		case currentSection == "version_policy" && strings.HasPrefix(line, "num_versions:"):
			parts := strings.SplitN(line, ":", 2)
			if cfg.VersionPolicy != nil && len(parts) == 2 {
				v, _ := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 32)
				cfg.VersionPolicy.NumVersions = int32(v)
			}

		case strings.HasPrefix(line, "}"):
			currentSection = ""
		}
	}

	if cfg.Name == "" {
		return nil, fmt.Errorf("model config missing 'name' field")
	}

	return cfg, nil
}

func parseDims(line string) []int64 {
	parts := strings.SplitN(line, ":", 2)
	if len(parts) < 2 {
		return nil
	}
	return parseNumList(parts[1])
}

func parseNumList(s string) []int64 {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "[")
	s = strings.TrimSuffix(s, "]")
	var nums []int64
	for _, item := range strings.Split(s, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		n, err := strconv.ParseInt(item, 10, 64)
		if err != nil {
			continue
		}
		nums = append(nums, n)
	}
	return nums
}
