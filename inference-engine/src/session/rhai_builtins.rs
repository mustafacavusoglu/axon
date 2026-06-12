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

fn register_cv_functions(engine: &mut Engine) {
    engine.register_fn(
        "decode_image",
        |data: &str| -> Result<rhai::Map, Box<rhai::EvalAltResult>> {
            use base64::Engine as B64Engine;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|e| format!("base64 decode failed: {e}"))?;
            let img = image::load_from_memory(&bytes)
                .map_err(|e| format!("image decode failed: {e}"))?;
            let rgb = img.to_rgb8();
            let (w, h) = rgb.dimensions();
            let pixels: rhai::Array = rgb
                .as_raw()
                .iter()
                .map(|&p| Dynamic::from(p as f64))
                .collect();
            let mut map = rhai::Map::new();
            map.insert("pixels".into(), Dynamic::from(pixels));
            map.insert("width".into(), Dynamic::from(w as i64));
            map.insert("height".into(), Dynamic::from(h as i64));
            map.insert("channels".into(), Dynamic::from(3_i64));
            Ok(map)
        },
    );

    engine.register_fn(
        "resize_image",
        |pixels: rhai::Array,
         src_h: i64,
         src_w: i64,
         dst_h: i64,
         dst_w: i64,
         channels: i64|
         -> rhai::Array {
            let sh = src_h as usize;
            let sw = src_w as usize;
            let dh = dst_h as usize;
            let dw = dst_w as usize;
            let ch = channels as usize;
            let src: Vec<f64> = pixels.iter().map(|v| to_f64(v)).collect();
            let mut dst = vec![0.0f64; dh * dw * ch];
            let scale_y = if dh > 1 {
                (sh as f64 - 1.0) / (dh as f64 - 1.0)
            } else {
                0.0
            };
            let scale_x = if dw > 1 {
                (sw as f64 - 1.0) / (dw as f64 - 1.0)
            } else {
                0.0
            };
            for y in 0..dh {
                for x in 0..dw {
                    let sy = y as f64 * scale_y;
                    let sx = x as f64 * scale_x;
                    let y0 = sy.floor() as usize;
                    let x0 = sx.floor() as usize;
                    let y1 = (y0 + 1).min(sh - 1);
                    let x1 = (x0 + 1).min(sw - 1);
                    let fy = sy - y0 as f64;
                    let fx = sx - x0 as f64;
                    for c in 0..ch {
                        let v00 = src[(y0 * sw + x0) * ch + c];
                        let v01 = src[(y0 * sw + x1) * ch + c];
                        let v10 = src[(y1 * sw + x0) * ch + c];
                        let v11 = src[(y1 * sw + x1) * ch + c];
                        dst[(y * dw + x) * ch + c] = v00 * (1.0 - fx) * (1.0 - fy)
                            + v01 * fx * (1.0 - fy)
                            + v10 * (1.0 - fx) * fy
                            + v11 * fx * fy;
                    }
                }
            }
            dst.into_iter().map(|v| Dynamic::from(v)).collect()
        },
    );

    engine.register_fn(
        "normalize_image",
        |pixels: rhai::Array, mean: rhai::Array, std: rhai::Array| -> rhai::Array {
            let mean_vals: Vec<f64> = mean.iter().map(|v| to_f64(v)).collect();
            let std_vals: Vec<f64> = std.iter().map(|v| to_f64(v)).collect();
            let ch = mean_vals.len();
            pixels
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let c = i % ch;
                    let s = if std_vals[c] != 0.0 {
                        std_vals[c]
                    } else {
                        1.0
                    };
                    Dynamic::from((to_f64(v) - mean_vals[c]) / s)
                })
                .collect()
        },
    );

    engine.register_fn(
        "image_to_chw",
        |pixels: rhai::Array, h: i64, w: i64, c: i64| -> rhai::Array {
            let height = h as usize;
            let width = w as usize;
            let channels = c as usize;
            let src: Vec<f64> = pixels.iter().map(|v| to_f64(v)).collect();
            let mut dst = vec![0.0f64; height * width * channels];
            for y in 0..height {
                for x in 0..width {
                    for ch in 0..channels {
                        dst[ch * height * width + y * width + x] =
                            src[(y * width + x) * channels + ch];
                    }
                }
            }
            dst.into_iter().map(|v| Dynamic::from(v)).collect()
        },
    );

    engine.register_fn(
        "center_crop",
        |pixels: rhai::Array,
         src_h: i64,
         src_w: i64,
         crop_h: i64,
         crop_w: i64,
         channels: i64|
         -> rhai::Array {
            let sh = src_h as usize;
            let sw = src_w as usize;
            let ch_val = crop_h as usize;
            let cw = crop_w as usize;
            let c = channels as usize;
            let start_y = (sh.saturating_sub(ch_val)) / 2;
            let start_x = (sw.saturating_sub(cw)) / 2;
            let src: Vec<f64> = pixels.iter().map(|v| to_f64(v)).collect();
            let mut dst = Vec::with_capacity(ch_val * cw * c);
            for y in 0..ch_val {
                for x in 0..cw {
                    let sy = start_y + y;
                    let sx = start_x + x;
                    for ch in 0..c {
                        dst.push(src[(sy * sw + sx) * c + ch]);
                    }
                }
            }
            dst.into_iter().map(|v| Dynamic::from(v)).collect()
        },
    );

    engine.register_fn(
        "grayscale",
        |pixels: rhai::Array, h: i64, w: i64| -> rhai::Array {
            let height = h as usize;
            let width = w as usize;
            let src: Vec<f64> = pixels.iter().map(|v| to_f64(v)).collect();
            let mut dst = Vec::with_capacity(height * width);
            for i in 0..(height * width) {
                let r = src[i * 3];
                let g = src[i * 3 + 1];
                let b = src[i * 3 + 2];
                dst.push(0.2989 * r + 0.5870 * g + 0.1140 * b);
            }
            dst.into_iter().map(|v| Dynamic::from(v)).collect()
        },
    );

    engine.register_fn(
        "nms",
        |boxes: rhai::Array,
         scores: rhai::Array,
         iou_threshold: f64|
         -> rhai::Array {
            let n = scores.len();
            let sc: Vec<f64> = scores.iter().map(|v| to_f64(v)).collect();
            let bx: Vec<[f64; 4]> = boxes
                .iter()
                .map(|v| {
                    let arr = v.clone().into_typed_array::<Dynamic>().unwrap_or_default();
                    let mut b = [0.0f64; 4];
                    for (i, val) in arr.iter().take(4).enumerate() {
                        b[i] = to_f64(val);
                    }
                    b
                })
                .collect();

            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| {
                sc[b].partial_cmp(&sc[a]).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut keep: Vec<i64> = Vec::new();
            let mut suppressed = vec![false; n];

            for &i in &indices {
                if suppressed[i] {
                    continue;
                }
                keep.push(i as i64);
                for &j in &indices {
                    if suppressed[j] || j == i {
                        continue;
                    }
                    let iou = compute_iou(&bx[i], &bx[j]);
                    if iou >= iou_threshold {
                        suppressed[j] = true;
                    }
                }
            }
            keep.into_iter().map(|v| Dynamic::from(v)).collect()
        },
    );
}

fn compute_iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}
