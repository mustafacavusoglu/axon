use anyhow::Result;

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

#[derive(Debug, Clone)]
pub struct TensorDef {
    pub name: String,
    pub data_type: DataType,
    pub dims: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct InstanceGroup {
    pub count: i32,
    pub kind: String,
}

#[derive(Debug, Clone)]
pub struct DynamicBatching {
    pub preferred_batch_size: Vec<i32>,
    pub max_queue_delay_us: i64,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub platform: String,
    pub max_batch_size: i32,
    pub inputs: Vec<TensorDef>,
    pub outputs: Vec<TensorDef>,
    pub instance_groups: Vec<InstanceGroup>,
    pub dynamic_batching: Option<DynamicBatching>,
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
    };

    let mut current_section = "";
    let mut brace_depth: i32 = 0;
    let mut in_list = false;

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
            if brace_depth == 0 && !current_section.is_empty() && !in_list {
                current_section = "";
            }
            continue;
        }

        let line = line.trim_end_matches('{').trim();

        if line == "]" {
            if in_list {
                current_section = "";
                in_list = false;
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
        ] {
            if line.starts_with(prefix)
                && (line.len() == prefix.len()
                    || line.as_bytes().get(prefix.len()) == Some(&b' ')
                    || line.as_bytes().get(prefix.len()) == Some(&b'['))
            {
                current_section = prefix;
                brace_depth = 0;
                in_list = line.ends_with('[');
                section_start = true;
                break;
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

fn strip_field<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    if line.starts_with(prefix) {
        Some(line[prefix.len()..].trim())
    } else {
        None
    }
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
}
