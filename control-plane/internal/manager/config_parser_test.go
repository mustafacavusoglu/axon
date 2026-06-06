package manager

import (
	"os"
	"testing"
)

func TestParseLgbmBreastCancerConfig(t *testing.T) {
	data, err := os.ReadFile("../../../local_models/model_repository/lgbm_breast_cancer/config.pbtxt")
	if err != nil {
		t.Skipf("config.pbtxt not found: %v", err)
	}

	cfg, err := ParseModelConfig(data)
	if err != nil {
		t.Fatalf("ParseModelConfig failed: %v", err)
	}

	if cfg.Name != "lgbm_breast_cancer" {
		t.Errorf("name = %q, want %q", cfg.Name, "lgbm_breast_cancer")
	}
	if cfg.Platform != "onnxruntime_onnx" {
		t.Errorf("platform = %q", cfg.Platform)
	}
	if cfg.MaxBatchSize != 32 {
		t.Errorf("max_batch_size = %d, want 32", cfg.MaxBatchSize)
	}
	if len(cfg.Inputs) != 1 {
		t.Fatalf("inputs count = %d, want 1", len(cfg.Inputs))
	}
	if cfg.Inputs[0].Name != "input" {
		t.Errorf("input name = %q, want %q", cfg.Inputs[0].Name, "input")
	}
	if cfg.Inputs[0].DataType != DTFP32 {
		t.Errorf("input dtype = %d, want TYPE_FP32", cfg.Inputs[0].DataType)
	}
	if len(cfg.Inputs[0].Dims) != 1 || cfg.Inputs[0].Dims[0] != 30 {
		t.Errorf("input dims = %v, want [30]", cfg.Inputs[0].Dims)
	}

	if len(cfg.Outputs) != 2 {
		t.Fatalf("outputs count = %d, want 2", len(cfg.Outputs))
	}
	if cfg.Outputs[0].Name != "output_label" {
		t.Errorf("output[0] name = %q", cfg.Outputs[0].Name)
	}
	if cfg.Outputs[0].DataType != DTINT64 {
		t.Errorf("output[0] dtype = %d, want TYPE_INT64", cfg.Outputs[0].DataType)
	}
	if cfg.Outputs[1].Name != "output_probability" {
		t.Errorf("output[1] name = %q", cfg.Outputs[1].Name)
	}
	if cfg.Outputs[1].DataType != DTFP32 {
		t.Errorf("output[1] dtype = %d, want TYPE_FP32", cfg.Outputs[1].DataType)
	}
	if len(cfg.Outputs[1].Dims) != 1 || cfg.Outputs[1].Dims[0] != 2 {
		t.Errorf("output[1] dims = %v, want [2]", cfg.Outputs[1].Dims)
	}

	if len(cfg.InstanceGroups) != 1 {
		t.Errorf("instance_groups = %d, want 1", len(cfg.InstanceGroups))
	}
	if cfg.InstanceGroups[0].Count != 2 {
		t.Errorf("instance_group count = %d, want 2", cfg.InstanceGroups[0].Count)
	}

	if cfg.DynamicBatching == nil {
		t.Error("dynamic_batching should not be nil")
	} else if cfg.DynamicBatching.MaxQueueDelayUs != 100 {
		t.Errorf("max_queue_delay = %d, want 100", cfg.DynamicBatching.MaxQueueDelayUs)
	}
}

func TestParseMultiInputConfig(t *testing.T) {
	data, err := os.ReadFile("../../../local_models/model_repository/multi_input_example/config.pbtxt")
	if err != nil {
		t.Skipf("config.pbtxt not found: %v", err)
	}

	cfg, err := ParseModelConfig(data)
	if err != nil {
		t.Fatalf("ParseModelConfig failed: %v", err)
	}

	if cfg.Name != "multi_input_example" {
		t.Errorf("name = %q, want %q", cfg.Name, "multi_input_example")
	}
	if cfg.MaxBatchSize != 8 {
		t.Errorf("max_batch_size = %d, want 8", cfg.MaxBatchSize)
	}
	if len(cfg.Inputs) != 5 {
		t.Fatalf("inputs count = %d, want 5", len(cfg.Inputs))
	}

	expected := []struct {
		name string
		dt   DataType
	}{
		{"customer_age", DTINT32},
		{"account_balance", DTFP32},
		{"credit_score", DTINT32},
		{"is_active_member", DTINT32},
		{"customer_country", DTSTRING},
	}
	for i, exp := range expected {
		if cfg.Inputs[i].Name != exp.name {
			t.Errorf("input[%d] name = %q, want %q", i, cfg.Inputs[i].Name, exp.name)
		}
		if cfg.Inputs[i].DataType != exp.dt {
			t.Errorf("input[%d] dtype = %d, want %d", i, cfg.Inputs[i].DataType, exp.dt)
		}
	}

	if len(cfg.Outputs) != 1 {
		t.Fatalf("outputs count = %d, want 1", len(cfg.Outputs))
	}
	if cfg.Outputs[0].Name != "prediction_probabilities" {
		t.Errorf("output name = %q", cfg.Outputs[0].Name)
	}
	if len(cfg.Outputs[0].Dims) != 1 || cfg.Outputs[0].Dims[0] != 2 {
		t.Errorf("output dims = %v, want [2]", cfg.Outputs[0].Dims)
	}
}
