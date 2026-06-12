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

fn register_nlp_functions(_engine: &mut Engine, _tokenizer: Arc<PLMutex<Option<Tokenizer>>>) {}

fn register_tabular_functions(_engine: &mut Engine) {}

fn register_cv_functions(_engine: &mut Engine) {}
