use std::collections::HashMap;

#[derive(Clone)]
pub enum InputTensor {
    F32(Vec<f32>, Vec<usize>),
    I32(Vec<i32>, Vec<usize>),
    I64(Vec<i64>, Vec<usize>),
    String(Vec<String>, Vec<usize>),
}

#[derive(Clone)]
pub enum TensorData {
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    String(Vec<String>),
}

#[allow(dead_code)]
impl TensorData {
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            TensorData::F32(data) => {
                let mut buf = Vec::with_capacity(data.len() * 4);
                for f in data {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
                buf
            }
            TensorData::I32(data) => {
                let mut buf = Vec::with_capacity(data.len() * 4);
                for i in data {
                    buf.extend_from_slice(&i.to_le_bytes());
                }
                buf
            }
            TensorData::I64(data) => {
                let mut buf = Vec::with_capacity(data.len() * 8);
                for i in data {
                    buf.extend_from_slice(&i.to_le_bytes());
                }
                buf
            }
            TensorData::String(data) => data.join("").into_bytes(),
        }
    }

    pub fn dtype_str(&self) -> &'static str {
        match self {
            TensorData::F32(_) => "FP32",
            TensorData::I32(_) => "INT32",
            TensorData::I64(_) => "INT64",
            TensorData::String(_) => "BYTES",
        }
    }

    pub fn as_f64_array(&self) -> Vec<f64> {
        match self {
            TensorData::F32(d) => d.iter().map(|&v| v as f64).collect(),
            TensorData::I32(d) => d.iter().map(|&v| v as f64).collect(),
            TensorData::I64(d) => d.iter().map(|&v| v as f64).collect(),
            TensorData::String(_) => vec![],
        }
    }

    pub fn as_i64_array(&self) -> Vec<i64> {
        match self {
            TensorData::F32(d) => d.iter().map(|&v| v as i64).collect(),
            TensorData::I32(d) => d.iter().map(|&v| v as i64).collect(),
            TensorData::I64(d) => d.clone(),
            TensorData::String(_) => vec![],
        }
    }
}

pub type InferenceOutput = Vec<(String, Vec<i64>, TensorData)>;

#[allow(dead_code)]
impl InputTensor {
    pub fn name(&self) -> &str {
        ""
    }

    pub fn into_f64_array(self) -> Vec<f64> {
        match self {
            InputTensor::F32(d, _) => d.into_iter().map(|v| v as f64).collect(),
            InputTensor::I32(d, _) => d.into_iter().map(|v| v as f64).collect(),
            InputTensor::I64(d, _) => d.into_iter().map(|v| v as f64).collect(),
            InputTensor::String(_, _) => vec![],
        }
    }

    pub fn into_i64_array(self) -> Vec<i64> {
        match self {
            InputTensor::F32(d, _) => d.into_iter().map(|v| v as i64).collect(),
            InputTensor::I32(d, _) => d.into_iter().map(|v| v as i64).collect(),
            InputTensor::I64(d, _) => d,
            InputTensor::String(_, _) => vec![],
        }
    }

    pub fn into_string(self) -> String {
        match self {
            InputTensor::String(d, _) => d.into_iter().next().unwrap_or_default(),
            _ => String::new(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            InputTensor::F32(d, _) => d.len(),
            InputTensor::I32(d, _) => d.len(),
            InputTensor::I64(d, _) => d.len(),
            InputTensor::String(d, _) => d.len(),
        }
    }
}

#[allow(dead_code)]
pub fn input_tensors_to_map(inputs: Vec<(String, InputTensor)>) -> HashMap<String, InputTensor> {
    let mut map = HashMap::new();
    for (name, tensor) in inputs {
        map.insert(name, tensor);
    }
    map
}

#[allow(dead_code)]
pub fn map_to_input_tensors(map: HashMap<String, InputTensor>) -> Vec<(String, InputTensor)> {
    map.into_iter().collect()
}
