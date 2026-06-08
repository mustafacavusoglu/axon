use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use rhai::{Dynamic, Engine, Scope, AST};
use tokio::sync::Semaphore;

use crate::session::pool::SessionPool;
use crate::tokenizer::Tokenizer;
use super::types::{InferenceOutput, InputTensor, TensorData};

#[derive(Clone)]
struct RhaiTensor {
    name: String,
    shape: Vec<i64>,
    datatype: String,
    data: RhaiTensorData,
}

#[derive(Clone)]
enum RhaiTensorData {
    F64(Vec<f64>),
    I64(Vec<i64>),
    String(Vec<String>),
}

impl RhaiTensor {
    fn from_input(name: String, tensor: InputTensor) -> Self {
        let shape: Vec<i64> = match &tensor {
            InputTensor::F32(_, s) => s.iter().map(|&x| x as i64).collect(),
            InputTensor::I32(_, s) => s.iter().map(|&x| x as i64).collect(),
            InputTensor::I64(_, s) => s.iter().map(|&x| x as i64).collect(),
            InputTensor::String(_, s) => s.iter().map(|&x| x as i64).collect(),
        };
        let datatype = match &tensor {
            InputTensor::F32(_, _) => "FP32",
            InputTensor::I32(_, _) => "INT32",
            InputTensor::I64(_, _) => "INT64",
            InputTensor::String(_, _) => "BYTES",
        };
        let data = match tensor {
            InputTensor::F32(d, _) => RhaiTensorData::F64(d.into_iter().map(|v| v as f64).collect()),
            InputTensor::I32(d, _) => RhaiTensorData::I64(d.into_iter().map(|v| v as i64).collect()),
            InputTensor::I64(d, _) => RhaiTensorData::I64(d),
            InputTensor::String(d, _) => RhaiTensorData::String(d),
        };
        RhaiTensor {
            name,
            shape,
            datatype: datatype.to_string(),
            data,
        }
    }

    fn into_output(self) -> (Vec<i64>, TensorData) {
        match self.data {
            RhaiTensorData::F64(d) => (self.shape, TensorData::F32(d.into_iter().map(|v| v as f32).collect())),
            RhaiTensorData::I64(d) => (self.shape, TensorData::I64(d)),
            RhaiTensorData::String(d) => {
                let len = d.len() as i64;
                (vec![len], TensorData::I32(vec![d.len() as i32]))
            }
        }
    }

    fn into_input(self) -> InputTensor {
        let shape: Vec<usize> = self.shape.iter().map(|&x| x as usize).collect();
        match self.data {
            RhaiTensorData::F64(d) => InputTensor::F32(d.into_iter().map(|v| v as f32).collect(), shape),
            RhaiTensorData::I64(d) => InputTensor::I64(d, shape),
            RhaiTensorData::String(d) => InputTensor::String(d, shape),
        }
    }

    fn from_output(name: String, shape: Vec<i64>, data: TensorData) -> Self {
        let datatype = data.dtype_str().to_string();
        let rhai_data = match data {
            TensorData::F32(d) => RhaiTensorData::F64(d.into_iter().map(|v| v as f64).collect()),
            TensorData::I32(d) => RhaiTensorData::I64(d.into_iter().map(|v| v as i64).collect()),
            TensorData::I64(d) => RhaiTensorData::I64(d),
        };
        RhaiTensor {
            name,
            shape,
            datatype,
            data: rhai_data,
        }
    }
}

pub struct RhaiRunner {
    engine: Arc<Engine>,
    ast: AST,
    semaphore: Arc<Semaphore>,
    script_path: PathBuf,
    tokenizer: Arc<Mutex<Option<Tokenizer>>>,
}

impl RhaiRunner {
    pub fn load(
        script_path: &Path,
        pool: SessionPool,
        concurrency: usize,
    ) -> anyhow::Result<Self> {
        let script_content = std::fs::read_to_string(script_path)
            .map_err(|e| anyhow::anyhow!("failed to read script {}: {}", script_path.display(), e))?;

        let script_dir = script_path.parent().unwrap_or(Path::new(".")).to_path_buf();

        let tokenizer: Arc<Mutex<Option<Tokenizer>>> = Arc::new(Mutex::new(None));
        let vocab_path = script_dir.join("vocab.txt");
        let tokenizer_json_path = script_dir.join("tokenizer.json");
        if vocab_path.exists() && tokenizer_json_path.exists() {
            let tok = Tokenizer::load(&vocab_path, &tokenizer_json_path)?;
            *tokenizer.lock() = Some(tok);
        }

        let mut engine = Engine::new();

        engine.register_type::<RhaiTensor>();

        engine.register_get("name", |t: &mut RhaiTensor| t.name.clone());
        engine.register_get("shape", |t: &mut RhaiTensor| t.shape.clone());
        engine.register_get("datatype", |t: &mut RhaiTensor| t.datatype.clone());

        engine.register_fn("as_f64", |t: &mut RhaiTensor| -> Result<Vec<Dynamic>, Box<rhai::EvalAltResult>> {
            match &t.data {
                RhaiTensorData::F64(d) => Ok(d.iter().map(|&v| Dynamic::from(v)).collect()),
                _ => Err("tensor is not FP32/FP64".into()),
            }
        });

        engine.register_fn("as_i64", |t: &mut RhaiTensor| -> Result<Vec<Dynamic>, Box<rhai::EvalAltResult>> {
            match &t.data {
                RhaiTensorData::I64(d) => Ok(d.iter().map(|&v| Dynamic::from(v)).collect()),
                _ => Err("tensor is not INT64".into()),
            }
        });

        engine.register_fn("as_string", |t: &mut RhaiTensor| -> String {
            match &t.data {
                RhaiTensorData::String(d) => d.first().cloned().unwrap_or_default(),
                _ => String::new(),
            }
        });

        engine.register_fn("as_f64_array", |t: &mut RhaiTensor| -> Vec<Dynamic> {
            match &t.data {
                RhaiTensorData::F64(d) => d.iter().map(|&v| Dynamic::from(v)).collect(),
                _ => vec![],
            }
        });

        engine.register_fn("as_i64_array", |t: &mut RhaiTensor| -> Vec<Dynamic> {
            match &t.data {
                RhaiTensorData::I64(d) => d.iter().map(|&v| Dynamic::from(v)).collect(),
                _ => vec![],
            }
        });

        engine.register_fn("create_tensor_f64",
            |name: &str, shape: Dynamic, data: rhai::Array| -> RhaiTensor {
                let shape_i64: Vec<i64> = if shape.is::<Vec<i64>>() {
                    shape.cast::<Vec<i64>>()
                } else if shape.is::<rhai::Array>() {
                    let arr = shape.cast::<rhai::Array>();
                    arr.iter().map(|v| v.as_int().unwrap_or(1)).collect()
                } else if shape.is_int() {
                    vec![shape.as_int().unwrap_or(1)]
                } else {
                    vec![1]
                };
                let data_f64: Vec<f64> = data.iter().map(|v| v.as_float().unwrap_or(0.0)).collect();
                RhaiTensor {
                    name: name.to_string(),
                    shape: shape_i64,
                    datatype: "FP32".to_string(),
                    data: RhaiTensorData::F64(data_f64),
                }
            }
        );

        engine.register_fn("create_tensor_i64",
            |name: &str, shape: Dynamic, data: rhai::Array| -> RhaiTensor {
                let shape_i64: Vec<i64> = if shape.is::<Vec<i64>>() {
                    shape.cast::<Vec<i64>>()
                } else if shape.is::<rhai::Array>() {
                    let arr = shape.cast::<rhai::Array>();
                    arr.iter().map(|v| v.as_int().unwrap_or(1)).collect()
                } else if shape.is_int() {
                    vec![shape.as_int().unwrap_or(1)]
                } else {
                    vec![1]
                };
                let data_i64: Vec<i64> = data.iter().map(|v| v.as_int().unwrap_or(0)).collect();
                RhaiTensor {
                    name: name.to_string(),
                    shape: shape_i64,
                    datatype: "INT64".to_string(),
                    data: RhaiTensorData::I64(data_i64),
                }
            }
        );

        let tokenizer_ref = tokenizer.clone();
        engine.register_fn("tokenize", move |text: &str| -> Result<rhai::Map, Box<rhai::EvalAltResult>> {
            let tok_guard = tokenizer_ref.lock();
            let tok = tok_guard.as_ref()
                .ok_or_else(|| "tokenizer not loaded (vocab.txt/tokenizer.json not found)".to_string())?;
            let (input_ids, attention_mask, token_type_ids) = tok.encode(text);
            drop(tok_guard);

            let seq_len = input_ids.len() as i64;

            let ids_tensor = RhaiTensor {
                name: "input_ids".to_string(),
                shape: vec![1, seq_len],
                datatype: "INT64".to_string(),
                data: RhaiTensorData::I64(input_ids),
            };
            let mask_tensor = RhaiTensor {
                name: "attention_mask".to_string(),
                shape: vec![1, seq_len],
                datatype: "INT64".to_string(),
                data: RhaiTensorData::I64(attention_mask),
            };
            let type_tensor = RhaiTensor {
                name: "token_type_ids".to_string(),
                shape: vec![1, seq_len],
                datatype: "INT64".to_string(),
                data: RhaiTensorData::I64(token_type_ids),
            };

            let mut result = rhai::Map::new();
            result.insert("input_ids".into(), Dynamic::from(ids_tensor));
            result.insert("attention_mask".into(), Dynamic::from(mask_tensor));
            result.insert("token_type_ids".into(), Dynamic::from(type_tensor));
            Ok(result)
        });

        let bls_pool = pool.clone();
        engine.register_fn("infer", move |model_name: &str, inputs: rhai::Map| -> Result<rhai::Map, Box<rhai::EvalAltResult>> {
            let mut input_tensors: Vec<(String, InputTensor)> = Vec::new();
            for (key, value) in inputs {
                let tensor: RhaiTensor = if value.is::<RhaiTensor>() {
                value.cast::<RhaiTensor>()
            } else {
                return Err(format!("expected Tensor for input '{}', got {:?}", key, value.type_name()).into());
            };
                input_tensors.push((key.to_string(), tensor.into_input()));
            }

            let session = bls_pool.get_latest(model_name)
                .ok_or_else(|| format!("model '{}' not found or not ready", model_name))?;

            let outputs = session.runner.run(input_tensors)
                .map_err(|e| format!("BLS inference failed for '{}': {}", model_name, e))?;

            let mut output_map = rhai::Map::new();
            for (name, shape, data) in outputs {
                let tensor = RhaiTensor::from_output(name.clone(), shape, data);
                output_map.insert(name.into(), Dynamic::from(tensor));
            }
            Ok(output_map)
        });

        let ast = engine.compile(script_content)
            .map_err(|e| anyhow::anyhow!("failed to compile script {}: {}", script_path.display(), e))?;

        tracing::info!(
            path = %script_path.display(),
            instances = concurrency,
            "Rhai script compiled"
        );

        Ok(Self {
            engine: Arc::new(engine),
            ast,
            semaphore: Arc::new(Semaphore::new(concurrency.max(1))),
            script_path: script_path.to_path_buf(),
            tokenizer,
        })
    }

    pub fn concurrency_semaphore(&self) -> &Arc<Semaphore> {
        &self.semaphore
    }

    pub fn run(&self, inputs: Vec<(String, InputTensor)>) -> anyhow::Result<InferenceOutput> {
        let mut scope = Scope::new();

        let mut input_map = rhai::Map::new();
        for (name, tensor) in inputs {
            let rhai_tensor = RhaiTensor::from_input(name.clone(), tensor);
            input_map.insert(name.into(), Dynamic::from(rhai_tensor));
        }

        let outputs_map: rhai::Map = self
            .engine
            .call_fn(&mut scope, &self.ast, "execute", (input_map,))
            .map_err(|e| {
                anyhow::anyhow!(
                    "Rhai script '{}' execution failed: {}",
                    self.script_path.display(),
                    e
                )
            })?;

        let mut outputs: InferenceOutput = Vec::new();
        for (key, value) in outputs_map {
            let tensor: RhaiTensor = if value.is::<RhaiTensor>() {
                value.cast::<RhaiTensor>()
            } else {
                return Err(anyhow::anyhow!(
                    "Rhai script '{}' returned non-Tensor value for output '{}'",
                    self.script_path.display(),
                    key
                ).into());
            };
            let (shape, data) = tensor.into_output();
            outputs.push((key.to_string(), shape, data));
        }

        Ok(outputs)
    }
}
