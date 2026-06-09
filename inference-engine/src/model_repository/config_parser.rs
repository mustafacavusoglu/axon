use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Invalid = 0,
    Fp32 = 1,
    Fp64 = 2,
    Int32 = 3,
    Int64 = 4,
    Int8 = 5,
    Uint8 = 6,
    Bool = 7,
    String = 8,
}

impl DataType {
    pub fn from_str_name(s: &str) -> Self {
        match s {
            "TYPE_FP32" => Self::Fp32,
            "TYPE_FP64" => Self::Fp64,
            "TYPE_INT32" => Self::Int32,
            "TYPE_INT64" => Self::Int64,
            "TYPE_INT8" => Self::Int8,
            "TYPE_UINT8" => Self::Uint8,
            "TYPE_BOOL" => Self::Bool,
            "TYPE_STRING" => Self::String,
            _ => Self::Invalid,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fp32 => "FP32",
            Self::Fp64 => "FP64",
            Self::Int32 => "INT32",
            Self::Int64 => "INT64",
            Self::Int8 => "INT8",
            Self::Uint8 => "UINT8",
            Self::Bool => "BOOL",
            Self::String => "BYTES",
            Self::Invalid => "INVALID",
        }
    }
}

impl<'de> Deserialize<'de> for DataType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(DataType::from_str_name(&s))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorDef {
    pub name: String,
    pub data_type: DataType,
    pub dims: Vec<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InstanceGroup {
    #[serde(default = "default_count")]
    pub count: i32,
    pub kind: String,
}

fn default_count() -> i32 {
    2
}

#[derive(Debug, Clone, Deserialize)]
pub struct DynamicBatching {
    pub preferred_batch_size: Vec<i32>,
    #[serde(rename = "max_queue_delay_microseconds")]
    pub max_queue_delay_us: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KeyValue {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EnsembleStep {
    pub model_name: String,
    #[serde(default = "default_model_version")]
    pub model_version: i32,
    #[serde(default)]
    pub input_map: Vec<KeyValue>,
    #[serde(default)]
    pub output_map: Vec<KeyValue>,
}

fn default_model_version() -> i32 {
    -1
}

#[derive(Debug, Clone, Deserialize)]
pub struct EnsembleScheduling {
    pub steps: Vec<EnsembleStep>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    #[serde(default = "default_platform")]
    pub platform: String,
    #[serde(default = "default_batch_size")]
    pub max_batch_size: i32,
    #[serde(default)]
    pub inputs: Vec<TensorDef>,
    #[serde(default)]
    pub outputs: Vec<TensorDef>,
    #[serde(default)]
    pub instance_groups: Vec<InstanceGroup>,
    pub dynamic_batching: Option<DynamicBatching>,
    pub ensemble_scheduling: Option<EnsembleScheduling>,
}

fn default_platform() -> String {
    "onnxruntime_onnx".to_string()
}
fn default_batch_size() -> i32 {
    1
}

pub fn parse_model_config(content: &[u8]) -> Result<ModelConfig> {
    let text = std::str::from_utf8(content)?;

    let mut cfg = ModelConfig {
        name: String::new(),
        platform: "onnxruntime_onnx".to_string(),
        max_batch_size: 1,
        inputs: Vec::new(),
        outputs: Vec::new(),
        instance_groups: Vec::new(),
        dynamic_batching: None,
        ensemble_scheduling: None,
    };

    let mut section_stack: Vec<String> = Vec::new();
    let mut section_depth: Vec<i32> = Vec::new();
    let mut in_list_stack: Vec<bool> = Vec::new();
    let mut current_section = "";
    let mut brace_depth: i32 = 0;
    let mut in_list = false;

    let mut current_step: Option<EnsembleStep> = None;
    let mut current_map_key = String::new();
    let mut current_map_value = String::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line == "{" {
            brace_depth += 1;
            continue;
        }
        if line == "}" {
            brace_depth -= 1;
            if !section_stack.is_empty() {
                let expected_depth = *section_depth.last().unwrap();
                if brace_depth == expected_depth {
                    let section = section_stack.last().unwrap().as_str();
                    match section {
                        "input_map" => {
                            if let Some(ref mut step) = current_step {
                                step.input_map.push(KeyValue {
                                    key: std::mem::take(&mut current_map_key),
                                    value: std::mem::take(&mut current_map_value),
                                });
                            }
                        }
                        "output_map" => {
                            if let Some(ref mut step) = current_step {
                                step.output_map.push(KeyValue {
                                    key: std::mem::take(&mut current_map_key),
                                    value: std::mem::take(&mut current_map_value),
                                });
                            }
                        }
                        "step" if in_list => {
                            if let Some(step) = current_step.take() {
                                let ens = cfg
                                    .ensemble_scheduling
                                    .get_or_insert(EnsembleScheduling { steps: Vec::new() });
                                ens.steps.push(step);
                            }
                            current_step = Some(EnsembleStep {
                                model_name: String::new(),
                                model_version: -1,
                                input_map: Vec::new(),
                                output_map: Vec::new(),
                            });
                        }
                        _ => {}
                    }
                    if !in_list {
                        section_depth.pop();
                        in_list_stack.pop();
                        let popped = section_stack.pop().unwrap();
                        if popped == "step" {
                            if let Some(step) = current_step.take() {
                                let ens = cfg
                                    .ensemble_scheduling
                                    .get_or_insert(EnsembleScheduling { steps: Vec::new() });
                                ens.steps.push(step);
                            }
                        }
                    }
                    current_section = section_stack.last().map(|s| s.as_str()).unwrap_or("");
                    in_list = in_list_stack.last().copied().unwrap_or(false);
                }
            }
            continue;
        }

        let line = line.trim_end_matches('{').trim();

        if line == "]" {
            if in_list {
                if !section_stack.is_empty() {
                    section_stack.pop();
                    section_depth.pop();
                    in_list_stack.pop();
                }
                current_section = section_stack.last().map(|s| s.as_str()).unwrap_or("");
                in_list = in_list_stack.last().copied().unwrap_or(false);
            }
            continue;
        }

        let mut section_start = false;
        for prefix in &[
            "input",
            "output",
            "instance_group",
            "dynamic_batching",
            "version_policy",
            "ensemble_scheduling",
        ] {
            if line.starts_with(prefix)
                && (line.len() == prefix.len()
                    || line.as_bytes().get(prefix.len()) == Some(&b' ')
                    || line.as_bytes().get(prefix.len()) == Some(&b'['))
            {
                section_stack.push(prefix.to_string());
                section_depth.push(brace_depth);
                current_section = prefix;
                in_list = line.ends_with('[');
                in_list_stack.push(in_list);
                section_start = true;
                break;
            }
        }

        if !section_start {
            if current_section == "ensemble_scheduling" {
                if let Some(prefix) = ["step"].iter().find(|p| {
                    line.starts_with(*p)
                        && (line.len() == p.len()
                            || line.as_bytes().get(p.len()) == Some(&b' ')
                            || line.as_bytes().get(p.len()) == Some(&b'['))
                }) {
                    section_stack.push(prefix.to_string());
                    section_depth.push(brace_depth);
                    current_section = prefix;
                    in_list = line.ends_with('[');
                    in_list_stack.push(in_list);
                    current_step = Some(EnsembleStep {
                        model_name: String::new(),
                        model_version: -1,
                        input_map: Vec::new(),
                        output_map: Vec::new(),
                    });
                    section_start = true;
                }
            } else if current_section == "step" {
                for map_prefix in &["input_map", "output_map"] {
                    if line.starts_with(map_prefix)
                        && (line.len() == map_prefix.len()
                            || line.as_bytes().get(map_prefix.len()) == Some(&b' ')
                            || line.as_bytes().get(map_prefix.len()) == Some(&b'['))
                    {
                        section_stack.push(map_prefix.to_string());
                        section_depth.push(brace_depth);
                        current_section = map_prefix;
                        in_list = line.ends_with('[');
                        in_list_stack.push(in_list);
                        current_map_key.clear();
                        current_map_value.clear();
                        section_start = true;
                        break;
                    }
                }
            }
        }

        if section_start {
            if raw_line.contains('{') {
                brace_depth += 1;
            }
            continue;
        }

        match current_section {
            "input" => {
                if let Some(val) = strip_field(line, "name:") {
                    cfg.inputs.push(TensorDef {
                        name: unquote(val),
                        data_type: DataType::Invalid,
                        dims: Vec::new(),
                    });
                } else if let Some(val) = strip_field(line, "data_type:") {
                    if let Some(last) = cfg.inputs.last_mut() {
                        last.data_type = DataType::from_str_name(val.trim());
                    }
                } else if let Some(val) = strip_field(line, "dims:") {
                    if let Some(last) = cfg.inputs.last_mut() {
                        last.dims = parse_num_list(val);
                    }
                }
            }
            "output" => {
                if let Some(val) = strip_field(line, "name:") {
                    cfg.outputs.push(TensorDef {
                        name: unquote(val),
                        data_type: DataType::Invalid,
                        dims: Vec::new(),
                    });
                } else if let Some(val) = strip_field(line, "data_type:") {
                    if let Some(last) = cfg.outputs.last_mut() {
                        last.data_type = DataType::from_str_name(val.trim());
                    }
                } else if let Some(val) = strip_field(line, "dims:") {
                    if let Some(last) = cfg.outputs.last_mut() {
                        last.dims = parse_num_list(val);
                    }
                }
            }
            "instance_group" => {
                if let Some(val) = strip_field(line, "count:") {
                    if let Ok(v) = val.trim().parse::<i32>() {
                        cfg.instance_groups.push(InstanceGroup {
                            count: v.max(1),
                            kind: "KIND_CPU".to_string(),
                        });
                    }
                } else if let Some(val) = strip_field(line, "kind:") {
                    if let Some(last) = cfg.instance_groups.last_mut() {
                        last.kind = unquote(val);
                    }
                }
            }
            "dynamic_batching" => {
                let db = cfg.dynamic_batching.get_or_insert(DynamicBatching {
                    preferred_batch_size: Vec::new(),
                    max_queue_delay_us: 0,
                });
                if let Some(val) = strip_field(line, "preferred_batch_size:") {
                    for n in parse_num_list(val) {
                        db.preferred_batch_size.push(n as i32);
                    }
                } else if let Some(val) = strip_field(line, "max_queue_delay_microseconds:") {
                    if let Ok(v) = val.trim().parse::<i64>() {
                        db.max_queue_delay_us = v;
                    }
                }
            }
            "ensemble_scheduling" => {}
            "step" => {
                if let Some(ref mut step) = current_step {
                    if let Some(val) = strip_field(line, "model_name:") {
                        step.model_name = unquote(val);
                    } else if let Some(val) = strip_field(line, "model_version:") {
                        step.model_version = val.trim().parse::<i32>().unwrap_or(-1);
                    }
                }
            }
            "input_map" | "output_map" => {
                if let Some(val) = strip_field(line, "key:") {
                    current_map_key = unquote(val);
                } else if let Some(val) = strip_field(line, "value:") {
                    current_map_value = unquote(val);
                }
            }
            "" => {
                if let Some(val) = strip_field(line, "name:") {
                    cfg.name = unquote(val);
                } else if let Some(val) = strip_field(line, "platform:") {
                    cfg.platform = unquote(val);
                } else if let Some(val) = strip_field(line, "max_batch_size:") {
                    if let Ok(v) = val.trim().parse::<i32>() {
                        cfg.max_batch_size = v.max(1);
                    }
                }
            }
            _ => {}
        }
    }

    if cfg.name.is_empty() {
        anyhow::bail!("model config missing 'name' field");
    }

    Ok(cfg)
}

pub fn parse_model_config_yaml(content: &[u8]) -> Result<ModelConfig> {
    let cfg: ModelConfig = serde_yaml::from_slice(content)?;
    if cfg.name.is_empty() {
        anyhow::bail!("model config missing 'name' field");
    }
    Ok(cfg)
}

fn strip_field<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    line.strip_prefix(prefix).map(|s| s.trim())
}

fn unquote(s: &str) -> String {
    s.trim().trim_matches('"').to_string()
}

fn parse_num_list(s: &str) -> Vec<i64> {
    let s = s.trim().trim_start_matches('[').trim_end_matches(']');
    s.split(',')
        .filter_map(|item| {
            let item = item.trim();
            if item.is_empty() {
                None
            } else {
                item.parse::<i64>().ok()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_config() {
        let content = br#"
name: "test_model"
platform: "onnxruntime_onnx"
max_batch_size: 8

input {
  name: "input_0"
  data_type: TYPE_FP32
  dims: [1, 30]
}

output {
  name: "output_0"
  data_type: TYPE_FP32
  dims: [1, 2]
}

instance_group {
  count: 2
  kind: "KIND_CPU"
}
"#;
        let cfg = parse_model_config(content).unwrap();
        assert_eq!(cfg.name, "test_model");
        assert_eq!(cfg.max_batch_size, 8);
        assert_eq!(cfg.inputs.len(), 1);
        assert_eq!(cfg.inputs[0].name, "input_0");
        assert_eq!(cfg.inputs[0].data_type, DataType::Fp32);
        assert_eq!(cfg.inputs[0].dims, vec![1, 30]);
        assert_eq!(cfg.outputs.len(), 1);
        assert_eq!(cfg.instance_groups[0].count, 2);
    }

    #[test]
    fn test_missing_name_errors() {
        let content = b"platform: \"onnxruntime_onnx\"\n";
        assert!(parse_model_config(content).is_err());
    }

    #[test]
    fn test_parse_list_format() {
        let content = br#"
name: "list_model"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {
    name: "input_0"
    data_type: TYPE_FP32
    dims: [1, 30]
  }
  {
    name: "input_1"
    data_type: TYPE_INT64
    dims: [1, -1]
  }
  {
    name: "input_2"
    data_type: TYPE_STRING
    dims: [1]
  }
]

output [
  {
    name: "output_0"
    data_type: TYPE_FP32
    dims: [1, 2]
  }
  {
    name: "output_1"
    data_type: TYPE_INT64
    dims: [1]
  }
]

instance_group {
  count: 2
  kind: KIND_CPU
}
"#;
        let cfg = parse_model_config(content).unwrap();
        assert_eq!(cfg.name, "list_model");
        assert_eq!(cfg.max_batch_size, 8);
        assert_eq!(cfg.inputs.len(), 3);
        assert_eq!(cfg.inputs[0].name, "input_0");
        assert_eq!(cfg.inputs[0].data_type, DataType::Fp32);
        assert_eq!(cfg.inputs[0].dims, vec![1, 30]);
        assert_eq!(cfg.inputs[1].name, "input_1");
        assert_eq!(cfg.inputs[1].data_type, DataType::Int64);
        assert_eq!(cfg.inputs[1].dims, vec![1, -1]);
        assert_eq!(cfg.inputs[2].name, "input_2");
        assert_eq!(cfg.inputs[2].data_type, DataType::String);
        assert_eq!(cfg.inputs[2].dims, vec![1]);
        assert_eq!(cfg.outputs.len(), 2);
        assert_eq!(cfg.outputs[0].name, "output_0");
        assert_eq!(cfg.outputs[0].data_type, DataType::Fp32);
        assert_eq!(cfg.outputs[0].dims, vec![1, 2]);
        assert_eq!(cfg.outputs[1].name, "output_1");
        assert_eq!(cfg.outputs[1].data_type, DataType::Int64);
        assert_eq!(cfg.outputs[1].dims, vec![1]);
        assert_eq!(cfg.instance_groups.len(), 1);
        assert_eq!(cfg.instance_groups[0].count, 2);
    }

    #[test]
    fn test_parse_ensemble_config() {
        let content = br#"
name: "test_ensemble"
platform: "ensemble"
max_batch_size: 1

input {
  name: "raw_text"
  data_type: TYPE_STRING
  dims: [1]
}

output {
  name: "result"
  data_type: TYPE_FP32
  dims: [1, 2]
}

ensemble_scheduling {
  step {
    model_name: "tokenizer"
    model_version: -1
    input_map {
      key: "text"
      value: "raw_text"
    }
    output_map {
      key: "input_ids"
      value: "ids"
    }
    output_map {
      key: "attention_mask"
      value: "mask"
    }
  }
  step {
    model_name: "classifier"
    model_version: 1
    input_map {
      key: "input_ids"
      value: "ids"
    }
    input_map {
      key: "attention_mask"
      value: "mask"
    }
    output_map {
      key: "logits"
      value: "result"
    }
  }
}

instance_group {
  count: 2
  kind: KIND_CPU
}
"#;
        let cfg = parse_model_config(content).unwrap();
        assert_eq!(cfg.name, "test_ensemble");
        assert_eq!(cfg.platform, "ensemble");
        assert_eq!(cfg.inputs.len(), 1);
        assert_eq!(cfg.inputs[0].name, "raw_text");
        assert_eq!(cfg.outputs.len(), 1);
        assert_eq!(cfg.outputs[0].name, "result");

        let ens = cfg
            .ensemble_scheduling
            .as_ref()
            .expect("ensemble_scheduling should be present");
        assert_eq!(ens.steps.len(), 2);

        let s0 = &ens.steps[0];
        assert_eq!(s0.model_name, "tokenizer");
        assert_eq!(s0.model_version, -1);
        assert_eq!(s0.input_map.len(), 1);
        assert_eq!(s0.input_map[0].key, "text");
        assert_eq!(s0.input_map[0].value, "raw_text");
        assert_eq!(s0.output_map.len(), 2);
        assert_eq!(s0.output_map[0].key, "input_ids");
        assert_eq!(s0.output_map[0].value, "ids");
        assert_eq!(s0.output_map[1].key, "attention_mask");
        assert_eq!(s0.output_map[1].value, "mask");

        let s1 = &ens.steps[1];
        assert_eq!(s1.model_name, "classifier");
        assert_eq!(s1.model_version, 1);
        assert_eq!(s1.input_map.len(), 2);
        assert_eq!(s1.input_map[0].key, "input_ids");
        assert_eq!(s1.input_map[0].value, "ids");
        assert_eq!(s1.input_map[1].key, "attention_mask");
        assert_eq!(s1.input_map[1].value, "mask");
        assert_eq!(s1.output_map.len(), 1);
        assert_eq!(s1.output_map[0].key, "logits");
        assert_eq!(s1.output_map[0].value, "result");
    }

    #[test]
    fn test_parse_ensemble_list_format() {
        let content = br#"
name: "ensemble_list"
platform: "ensemble"
max_batch_size: 1

input {
  name: "raw_text"
  data_type: TYPE_STRING
  dims: [1]
}

output {
  name: "result"
  data_type: TYPE_FP32
  dims: [1, 2]
}

ensemble_scheduling {
  step {
    model_name: "tokenizer"
    model_version: -1
    input_map [
      {
        key: "text"
        value: "raw_text"
      }
    ]
    output_map [
      {
        key: "input_ids"
        value: "ids"
      }
      {
        key: "attention_mask"
        value: "mask"
      }
    ]
  }
  step {
    model_name: "classifier"
    model_version: 1
    input_map [
      {
        key: "input_ids"
        value: "ids"
      }
      {
        key: "attention_mask"
        value: "mask"
      }
    ]
    output_map [
      {
        key: "logits"
        value: "result"
      }
    ]
  }
}

instance_group {
  count: 2
  kind: KIND_CPU
}
"#;
        let cfg = parse_model_config(content).unwrap();
        assert_eq!(cfg.name, "ensemble_list");
        assert_eq!(cfg.platform, "ensemble");

        let ens = cfg
            .ensemble_scheduling
            .as_ref()
            .expect("ensemble_scheduling should be present");
        assert_eq!(ens.steps.len(), 2);

        let s0 = &ens.steps[0];
        assert_eq!(s0.model_name, "tokenizer");
        assert_eq!(s0.model_version, -1);
        assert_eq!(s0.input_map.len(), 1);
        assert_eq!(s0.input_map[0].key, "text");
        assert_eq!(s0.input_map[0].value, "raw_text");
        assert_eq!(s0.output_map.len(), 2);
        assert_eq!(s0.output_map[0].key, "input_ids");
        assert_eq!(s0.output_map[0].value, "ids");
        assert_eq!(s0.output_map[1].key, "attention_mask");
        assert_eq!(s0.output_map[1].value, "mask");

        let s1 = &ens.steps[1];
        assert_eq!(s1.model_name, "classifier");
        assert_eq!(s1.model_version, 1);
        assert_eq!(s1.input_map.len(), 2);
        assert_eq!(s1.input_map[0].key, "input_ids");
        assert_eq!(s1.input_map[0].value, "ids");
        assert_eq!(s1.input_map[1].key, "attention_mask");
        assert_eq!(s1.input_map[1].value, "mask");
        assert_eq!(s1.output_map.len(), 1);
        assert_eq!(s1.output_map[0].key, "logits");
        assert_eq!(s1.output_map[0].value, "result");
    }

    #[test]
    fn test_parse_yaml_config() {
        let content = br#"
name: yaml_model
platform: onnxruntime_onnx
max_batch_size: 8

inputs:
  - name: input_0
    data_type: TYPE_FP32
    dims: [1, 30]
  - name: input_1
    data_type: TYPE_INT64
    dims: [1, -1]
  - name: input_2
    data_type: TYPE_STRING
    dims: [1]

outputs:
  - name: output_0
    data_type: TYPE_FP32
    dims: [1, 2]
  - name: output_1
    data_type: TYPE_INT64
    dims: [1]

instance_groups:
  - count: 2
    kind: KIND_CPU
"#;
        let cfg = parse_model_config_yaml(content).unwrap();
        assert_eq!(cfg.name, "yaml_model");
        assert_eq!(cfg.max_batch_size, 8);
        assert_eq!(cfg.inputs.len(), 3);
        assert_eq!(cfg.inputs[0].name, "input_0");
        assert_eq!(cfg.inputs[0].data_type, DataType::Fp32);
        assert_eq!(cfg.inputs[0].dims, vec![1, 30]);
        assert_eq!(cfg.inputs[1].name, "input_1");
        assert_eq!(cfg.inputs[1].data_type, DataType::Int64);
        assert_eq!(cfg.inputs[1].dims, vec![1, -1]);
        assert_eq!(cfg.outputs.len(), 2);
        assert_eq!(cfg.outputs[0].name, "output_0");
        assert_eq!(cfg.outputs[0].data_type, DataType::Fp32);
        assert_eq!(cfg.outputs[0].dims, vec![1, 2]);
        assert_eq!(cfg.instance_groups.len(), 1);
        assert_eq!(cfg.instance_groups[0].count, 2);
    }

    #[test]
    fn test_parse_yaml_ensemble_config() {
        let content = br#"
name: ensemble_yaml
platform: ensemble
max_batch_size: 1

inputs:
  - name: raw_text
    data_type: TYPE_STRING
    dims: [1]

outputs:
  - name: result
    data_type: TYPE_FP32
    dims: [1, 2]

ensemble_scheduling:
  steps:
    - model_name: tokenizer
      model_version: -1
      input_map:
        - key: text
          value: raw_text
      output_map:
        - key: input_ids
          value: ids
        - key: attention_mask
          value: mask
    - model_name: classifier
      model_version: 1
      input_map:
        - key: input_ids
          value: ids
        - key: attention_mask
          value: mask
      output_map:
        - key: logits
          value: result

instance_groups:
  - count: 2
    kind: KIND_CPU
"#;
        let cfg = parse_model_config_yaml(content).unwrap();
        assert_eq!(cfg.name, "ensemble_yaml");
        assert_eq!(cfg.platform, "ensemble");

        let ens = cfg
            .ensemble_scheduling
            .as_ref()
            .expect("ensemble_scheduling should be present");
        assert_eq!(ens.steps.len(), 2);

        let s0 = &ens.steps[0];
        assert_eq!(s0.model_name, "tokenizer");
        assert_eq!(s0.model_version, -1);
        assert_eq!(s0.input_map.len(), 1);
        assert_eq!(s0.input_map[0].key, "text");
        assert_eq!(s0.input_map[0].value, "raw_text");
        assert_eq!(s0.output_map.len(), 2);
        assert_eq!(s0.output_map[0].key, "input_ids");
        assert_eq!(s0.output_map[0].value, "ids");

        let s1 = &ens.steps[1];
        assert_eq!(s1.model_name, "classifier");
        assert_eq!(s1.model_version, 1);
        assert_eq!(s1.input_map.len(), 2);
        assert_eq!(s1.output_map.len(), 1);
        assert_eq!(s1.output_map[0].key, "logits");
        assert_eq!(s1.output_map[0].value, "result");
    }

    #[test]
    fn test_parse_ensemble_step_list_format() {
        let content = br#"
name: "ensemble_step_list"
platform: "ensemble"
max_batch_size: 1

input {
  name: "raw_text"
  data_type: TYPE_STRING
  dims: [1]
}

output {
  name: "sentiment"
  data_type: TYPE_FP32
  dims: [1, 3]
}
output {
  name: "confidence"
  data_type: TYPE_FP32
  dims: [1]
}

ensemble_scheduling {
  step [
    {
      model_name: "tokenizer"
      model_version: -1
      input_map [
        {
          key: "text"
          value: "raw_text"
        }
      ]
      output_map [
        {
          key: "input_ids"
          value: "ids"
        }
        {
          key: "attention_mask"
          value: "mask"
        }
      ]
    }
    {
      model_name: "classifier"
      model_version: 2
      input_map [
        {
          key: "input_ids"
          value: "ids"
        }
        {
          key: "attention_mask"
          value: "mask"
        }
      ]
      output_map [
        {
          key: "logits"
          value: "sentiment"
        }
        {
          key: "probability"
          value: "confidence"
        }
      ]
    }
  ]
}

instance_group {
  count: 1
  kind: KIND_CPU
}
"#;
        let cfg = parse_model_config(content).unwrap();
        assert_eq!(cfg.name, "ensemble_step_list");

        let ens = cfg
            .ensemble_scheduling
            .as_ref()
            .expect("ensemble_scheduling should be present");
        assert_eq!(ens.steps.len(), 2);

        let s0 = &ens.steps[0];
        assert_eq!(s0.model_name, "tokenizer");
        assert_eq!(s0.model_version, -1);
        assert_eq!(s0.input_map.len(), 1);
        assert_eq!(s0.input_map[0].key, "text");
        assert_eq!(s0.input_map[0].value, "raw_text");
        assert_eq!(s0.output_map.len(), 2);
        assert_eq!(s0.output_map[0].key, "input_ids");
        assert_eq!(s0.output_map[0].value, "ids");
        assert_eq!(s0.output_map[1].key, "attention_mask");
        assert_eq!(s0.output_map[1].value, "mask");

        let s1 = &ens.steps[1];
        assert_eq!(s1.model_name, "classifier");
        assert_eq!(s1.model_version, 2);
        assert_eq!(s1.input_map.len(), 2);
        assert_eq!(s1.input_map[0].key, "input_ids");
        assert_eq!(s1.input_map[0].value, "ids");
        assert_eq!(s1.output_map.len(), 2);
        assert_eq!(s1.output_map[0].key, "logits");
        assert_eq!(s1.output_map[0].value, "sentiment");
        assert_eq!(s1.output_map[1].key, "probability");
        assert_eq!(s1.output_map[1].value, "confidence");
    }

    #[test]
    fn test_parse_step_list_simple() {
        let content = br#"name: "simple"
platform: "ensemble"
max_batch_size: 1
input { name: "in" data_type: TYPE_STRING dims: [1] }
output { name: "out" data_type: TYPE_FP32 dims: [1] }
ensemble_scheduling {
  step [
    {
      model_name: "s1"
    }
    {
      model_name: "s2"
    }
  ]
}
"#;
        let cfg = parse_model_config(content).unwrap();
        let ens = cfg.ensemble_scheduling.as_ref().unwrap();
        assert_eq!(ens.steps.len(), 2);
        assert_eq!(ens.steps[0].model_name, "s1");
        assert_eq!(ens.steps[1].model_name, "s2");
    }

    #[test]
    fn test_parse_step_list_with_map() {
        let content = br#"name: "with_map"
platform: "ensemble"
max_batch_size: 1
input { name: "in" data_type: TYPE_STRING dims: [1] }
output { name: "out" data_type: TYPE_FP32 dims: [1] }
ensemble_scheduling {
  step [
    {
      model_name: "s1"
      input_map [
        {
          key: "text"
          value: "raw"
        }
      ]
      output_map [
        {
          key: "result"
          value: "out"
        }
      ]
    }
  ]
}
"#;
        let cfg = parse_model_config(content).unwrap();
        let ens = cfg.ensemble_scheduling.as_ref().unwrap();
        assert_eq!(ens.steps.len(), 1);
        assert_eq!(ens.steps[0].model_name, "s1");
        assert_eq!(ens.steps[0].input_map.len(), 1);
        assert_eq!(ens.steps[0].input_map[0].key, "text");
        assert_eq!(ens.steps[0].output_map.len(), 1);
    }

    #[test]
    fn test_parse_step_list_two_with_maps() {
        let content = br#"name: "two_maps"
platform: "ensemble"
max_batch_size: 1
input { name: "in" data_type: TYPE_STRING dims: [1] }
output { name: "out" data_type: TYPE_FP32 dims: [1] }
ensemble_scheduling {
  step [
    {
      model_name: "s1"
      output_map [
        {
          key: "from1"
          value: "mid"
        }
      ]
    }
    {
      model_name: "s2"
      input_map [
        {
          key: "to2"
          value: "mid"
        }
      ]
    }
  ]
}
"#;
        let cfg = parse_model_config(content).unwrap();
        let ens = cfg.ensemble_scheduling.as_ref().unwrap();
        assert_eq!(ens.steps.len(), 2);
        assert_eq!(ens.steps[0].model_name, "s1");
        assert_eq!(ens.steps[0].output_map[0].key, "from1");
        assert_eq!(ens.steps[1].model_name, "s2");
        assert_eq!(ens.steps[1].input_map[0].key, "to2");
    }

    #[test]
    fn test_parse_step_list_both_maps() {
        let content = br#"name: "both"
platform: "ensemble"
max_batch_size: 1
input { name: "in" data_type: TYPE_STRING dims: [1] }
output { name: "out" data_type: TYPE_FP32 dims: [1] }
ensemble_scheduling {
  step [
    {
      model_name: "s1"
      input_map [
        {
          key: "a"
          value: "in"
        }
      ]
      output_map [
        {
          key: "b"
          value: "mid"
        }
      ]
    }
    {
      model_name: "s2"
      input_map [
        {
          key: "c"
          value: "mid"
        }
      ]
      output_map [
        {
          key: "d"
          value: "out"
        }
      ]
    }
  ]
}
"#;
        let cfg = parse_model_config(content).unwrap();
        let ens = cfg.ensemble_scheduling.as_ref().unwrap();
        assert_eq!(ens.steps.len(), 2);
        assert_eq!(ens.steps[0].model_name, "s1");
        assert_eq!(ens.steps[0].input_map[0].key, "a");
        assert_eq!(ens.steps[0].output_map[0].key, "b");
        assert_eq!(ens.steps[1].model_name, "s2");
        assert_eq!(ens.steps[1].input_map[0].key, "c");
        assert_eq!(ens.steps[1].output_map[0].key, "d");
    }

    #[test]
    fn test_parse_step_list_multi_map_entries() {
        let content = br#"name: "multi_entries"
platform: "ensemble"
max_batch_size: 1
input { name: "in" data_type: TYPE_STRING dims: [1] }
output { name: "out_a" data_type: TYPE_FP32 dims: [1] }
output { name: "out_b" data_type: TYPE_FP32 dims: [1] }
ensemble_scheduling {
  step [
    {
      model_name: "s1"
      input_map [
        {
          key: "text"
          value: "in"
        }
      ]
      output_map [
        {
          key: "input_ids"
          value: "ids"
        }
        {
          key: "attention_mask"
          value: "mask"
        }
      ]
    }
    {
      model_name: "s2"
      input_map [
        {
          key: "input_ids"
          value: "ids"
        }
        {
          key: "attention_mask"
          value: "mask"
        }
      ]
      output_map [
        {
          key: "logits"
          value: "out_a"
        }
        {
          key: "probability"
          value: "out_b"
        }
      ]
    }
  ]
}
"#;
        let cfg = parse_model_config(content).unwrap();
        let ens = cfg.ensemble_scheduling.as_ref().unwrap();
        assert_eq!(ens.steps.len(), 2);
        assert_eq!(ens.steps[0].output_map.len(), 2);
        assert_eq!(ens.steps[0].output_map[0].key, "input_ids");
        assert_eq!(ens.steps[0].output_map[1].key, "attention_mask");
        assert_eq!(ens.steps[1].input_map.len(), 2);
        assert_eq!(ens.steps[1].output_map.len(), 2);
    }
}
