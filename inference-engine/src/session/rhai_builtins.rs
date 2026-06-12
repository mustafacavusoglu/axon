use std::sync::Arc;

use parking_lot::Mutex as PLMutex;
use rhai::{Dynamic, Engine};
use tokenizers::Tokenizer;

fn to_f64(v: &Dynamic) -> f64 {
    if let Ok(f) = v.as_float() {
        f
    } else if let Ok(i) = v.as_int() {
        i as f64
    } else {
        0.0
    }
}

pub fn register_all(engine: &mut Engine, tokenizer: Arc<PLMutex<Option<Tokenizer>>>) {
    register_math_functions(engine);
    register_nlp_functions(engine, tokenizer);
    register_tabular_functions(engine);
    register_cv_functions(engine);
}

fn register_math_functions(engine: &mut Engine) {
    engine.register_fn("softmax", |arr: rhai::Array| -> rhai::Array {
        let values: Vec<f64> = arr.iter().map(|v| to_f64(v)).collect();
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = values.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| Dynamic::from(e / sum)).collect()
    });

    engine.register_fn("sigmoid", |arr: rhai::Array| -> rhai::Array {
        arr.iter()
            .map(|v| Dynamic::from(1.0 / (1.0 + (-to_f64(v)).exp())))
            .collect()
    });

    engine.register_fn("argmax", |arr: rhai::Array| -> i64 {
        let mut max_idx = 0i64;
        let mut max_val = f64::NEG_INFINITY;
        for (i, v) in arr.iter().enumerate() {
            let val = to_f64(v);
            if val > max_val {
                max_val = val;
                max_idx = i as i64;
            }
        }
        max_idx
    });

    engine.register_fn("argmin", |arr: rhai::Array| -> i64 {
        let mut min_idx = 0i64;
        let mut min_val = f64::INFINITY;
        for (i, v) in arr.iter().enumerate() {
            let val = to_f64(v);
            if val < min_val {
                min_val = val;
                min_idx = i as i64;
            }
        }
        min_idx
    });

    engine.register_fn("topk", |arr: rhai::Array, k: i64| -> rhai::Array {
        let mut indexed: Vec<(usize, f64)> = arr
            .iter()
            .enumerate()
            .map(|(i, v)| (i, to_f64(v)))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed
            .into_iter()
            .take(k.max(0) as usize)
            .map(|(idx, val)| {
                let mut map = rhai::Map::new();
                map.insert("index".into(), Dynamic::from(idx as i64));
                map.insert("value".into(), Dynamic::from(val));
                Dynamic::from(map)
            })
            .collect()
    });

    engine.register_fn(
        "threshold",
        |arr: rhai::Array, val: f64| -> rhai::Array {
            arr.iter()
                .map(|v| {
                    Dynamic::from(if to_f64(v) >= val { 1.0_f64 } else { 0.0_f64 })
                })
                .collect()
        },
    );

    engine.register_fn(
        "clip",
        |arr: rhai::Array, min_val: f64, max_val: f64| -> rhai::Array {
            arr.iter()
                .map(|v| Dynamic::from(to_f64(v).clamp(min_val, max_val)))
                .collect()
        },
    );
}

fn register_nlp_functions(engine: &mut Engine, tokenizer: Arc<PLMutex<Option<Tokenizer>>>) {
    engine.register_fn(
        "pad_sequence",
        |arr: rhai::Array, target_len: i64, pad_value: i64| -> rhai::Array {
            let len = target_len.max(0) as usize;
            if arr.len() >= len {
                arr[..len].to_vec()
            } else {
                let mut result = arr;
                result.resize(len, Dynamic::from(pad_value));
                result
            }
        },
    );

    engine.register_fn("text_lower", |text: &str| -> String {
        text.to_lowercase()
    });

    engine.register_fn(
        "regex_replace",
        |text: &str,
         pattern: &str,
         replacement: &str|
         -> Result<String, Box<rhai::EvalAltResult>> {
            let re = regex::Regex::new(pattern)
                .map_err(|e| format!("invalid regex '{}': {e}", pattern))?;
            Ok(re.replace_all(text, replacement).to_string())
        },
    );

    engine.register_fn(
        "decode_tokens",
        move |ids: rhai::Array| -> Result<String, Box<rhai::EvalAltResult>> {
            let tok = tokenizer.lock();
            let t = tok
                .as_ref()
                .ok_or_else(|| "tokenizer.json not loaded".to_string())?;
            let token_ids: Vec<u32> = ids
                .iter()
                .map(|v| v.as_int().unwrap_or(0) as u32)
                .collect();
            t.decode(&token_ids, true)
                .map_err(|e| format!("decode failed: {e}").into())
        },
    );
}

fn register_tabular_functions(engine: &mut Engine) {
    engine.register_fn(
        "normalize",
        |arr: rhai::Array, method: &str| -> rhai::Array {
            let values: Vec<f64> = arr.iter().map(|v| to_f64(v)).collect();
            match method {
                "minmax" => {
                    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let range = max - min;
                    if range == 0.0 {
                        values.iter().map(|_| Dynamic::from(0.0_f64)).collect()
                    } else {
                        values
                            .iter()
                            .map(|&v| Dynamic::from((v - min) / range))
                            .collect()
                    }
                }
                "l2" => {
                    let norm: f64 = values.iter().map(|&v| v * v).sum::<f64>().sqrt();
                    if norm == 0.0 {
                        values.iter().map(|_| Dynamic::from(0.0_f64)).collect()
                    } else {
                        values
                            .iter()
                            .map(|&v| Dynamic::from(v / norm))
                            .collect()
                    }
                }
                _ => arr,
            }
        },
    );

    engine.register_fn(
        "standardize",
        |arr: rhai::Array, mean: f64, std_dev: f64| -> rhai::Array {
            arr.iter()
                .map(|v| {
                    let x = to_f64(v);
                    Dynamic::from(if std_dev != 0.0 {
                        (x - mean) / std_dev
                    } else {
                        0.0
                    })
                })
                .collect()
        },
    );

    engine.register_fn(
        "one_hot",
        |index: i64, num_classes: i64| -> rhai::Array {
            (0..num_classes)
                .map(|i| Dynamic::from(if i == index { 1.0_f64 } else { 0.0_f64 }))
                .collect()
        },
    );

    engine.register_fn(
        "label_encode",
        |value: &str, mapping: rhai::Map| -> i64 {
            mapping
                .get(value)
                .and_then(|v| v.as_int().ok())
                .unwrap_or(-1)
        },
    );

    engine.register_fn(
        "fill_missing",
        |arr: rhai::Array, strategy: &str| -> rhai::Array {
            let values: Vec<f64> = arr.iter().map(|v| to_f64(v)).collect();
            let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).cloned().collect();
            let fill_val = match strategy {
                "zero" => 0.0,
                "mean" => {
                    if valid.is_empty() {
                        0.0
                    } else {
                        valid.iter().sum::<f64>() / valid.len() as f64
                    }
                }
                "median" => {
                    if valid.is_empty() {
                        0.0
                    } else {
                        let mut sorted = valid.clone();
                        sorted.sort_by(|a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let mid = sorted.len() / 2;
                        if sorted.len() % 2 == 0 {
                            (sorted[mid - 1] + sorted[mid]) / 2.0
                        } else {
                            sorted[mid]
                        }
                    }
                }
                _ => 0.0,
            };
            values
                .iter()
                .map(|&v| Dynamic::from(if v.is_nan() { fill_val } else { v }))
                .collect()
        },
    );
}

fn register_cv_functions(_engine: &mut Engine) {}
