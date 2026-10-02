use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::Mutex as PLMutex;
use rhai::{Dynamic, Engine, EvalAltResult};
use tokenizers::Tokenizer;

type RhaiResult<T> = Result<T, Box<EvalAltResult>>;

static REGEX_CACHE: Lazy<PLMutex<std::collections::HashMap<String, Arc<regex::Regex>>>> =
    Lazy::new(|| PLMutex::new(std::collections::HashMap::new()));

const MAX_IMAGE_BYTES: usize = 32 * 1024 * 1024;
const MAX_IMAGE_PIXELS: u32 = 4096;
/// Upper bound on arrays produced by builtins. Checked *before* allocating,
/// since Rhai's own array-size limit is only enforced after a native function
/// returns (too late to prevent an OOM abort).
pub const MAX_BUILTIN_ELEMENTS: usize = 64 * 1024 * 1024;
const MAX_NMS_BOXES: usize = 100_000;

fn check_len(what: &str, n: usize) -> RhaiResult<()> {
    if n > MAX_BUILTIN_ELEMENTS {
        return Err(format!("{what}: {n} elements exceeds limit {MAX_BUILTIN_ELEMENTS}").into());
    }
    Ok(())
}

/// Converts non-negative `i64` dimensions to `usize` and returns their
/// product, rejecting negative values and overflow.
fn dims(what: &str, values: &[i64]) -> RhaiResult<(Vec<usize>, usize)> {
    let mut out = Vec::with_capacity(values.len());
    let mut product: usize = 1;
    for &v in values {
        let d = usize::try_from(v).map_err(|_| format!("{what}: negative dimension {v}"))?;
        product = product
            .checked_mul(d)
            .ok_or_else(|| format!("{what}: dimensions overflow"))?;
        out.push(d);
    }
    check_len(what, product)?;
    Ok((out, product))
}

fn expect_len(what: &str, actual: usize, expected: usize) -> RhaiResult<()> {
    if actual != expected {
        return Err(format!("{what}: expected {expected} values, got {actual}").into());
    }
    Ok(())
}

fn to_f64(v: &Dynamic) -> f64 {
    if let Ok(f) = v.as_float() {
        f
    } else if let Ok(i) = v.as_int() {
        i as f64
    } else {
        0.0
    }
}

pub fn register_all(engine: &mut Engine, tokenizer: Arc<Option<Tokenizer>>) {
    register_math_functions(engine);
    register_nlp_functions(engine, tokenizer);
    register_tabular_functions(engine);
    register_cv_functions(engine);
}

fn register_math_functions(engine: &mut Engine) {
    engine.register_fn("softmax", |arr: rhai::Array| -> rhai::Array {
        let values: Vec<f64> = arr.iter().map(to_f64).collect();
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
        // Partial selection would be faster, but arrays here are small.
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

    engine.register_fn("threshold", |arr: rhai::Array, val: f64| -> rhai::Array {
        arr.iter()
            .map(|v| Dynamic::from(if to_f64(v) >= val { 1.0_f64 } else { 0.0_f64 }))
            .collect()
    });

    engine.register_fn(
        "clip",
        |arr: rhai::Array, min_val: f64, max_val: f64| -> rhai::Array {
            arr.iter()
                .map(|v| Dynamic::from(to_f64(v).clamp(min_val, max_val)))
                .collect()
        },
    );
}

fn register_nlp_functions(engine: &mut Engine, tokenizer: Arc<Option<Tokenizer>>) {
    engine.register_fn(
        "pad_sequence",
        |arr: rhai::Array, target_len: i64, pad_value: i64| -> RhaiResult<rhai::Array> {
            let len = target_len.max(0) as usize;
            check_len("pad_sequence", len)?;
            let mut result = arr;
            result.resize(len, Dynamic::from(pad_value));
            Ok(result)
        },
    );

    engine.register_fn("text_lower", |text: &str| -> String { text.to_lowercase() });

    engine.register_fn(
        "regex_replace",
        |text: &str,
         pattern: &str,
         replacement: &str|
         -> Result<String, Box<rhai::EvalAltResult>> {
            let re = {
                let mut cache = REGEX_CACHE.lock();
                if let Some(r) = cache.get(pattern) {
                    r.clone()
                } else if cache.len() >= 256 {
                    return Err(format!("regex cache full, pattern rejected: {pattern}").into());
                } else {
                    let r = Arc::new(
                        regex::Regex::new(pattern)
                            .map_err(|e| format!("invalid regex '{pattern}': {e}"))?,
                    );
                    cache.insert(pattern.to_string(), r.clone());
                    r
                }
            };
            Ok(re.replace_all(text, replacement).to_string())
        },
    );

    engine.register_fn(
        "decode_tokens",
        move |ids: rhai::Array| -> Result<String, Box<rhai::EvalAltResult>> {
            let t = tokenizer
                .as_ref()
                .as_ref()
                .ok_or_else(|| "tokenizer.json not loaded".to_string())?;
            let token_ids: Vec<u32> = ids
                .iter()
                .map(|v| {
                    v.as_int()
                        .ok()
                        .and_then(|i| u32::try_from(i).ok())
                        .ok_or_else(|| format!("decode_tokens: invalid token id {v}"))
                })
                .collect::<Result<_, _>>()?;
            t.decode(&token_ids, true)
                .map_err(|e| format!("decode failed: {e}").into())
        },
    );
}

fn register_tabular_functions(engine: &mut Engine) {
    engine.register_fn(
        "normalize",
        |arr: rhai::Array, method: &str| -> rhai::Array {
            let values: Vec<f64> = arr.iter().map(to_f64).collect();
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
                        values.iter().map(|&v| Dynamic::from(v / norm)).collect()
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
        |index: i64, num_classes: i64| -> RhaiResult<rhai::Array> {
            check_len("one_hot", num_classes.max(0) as usize)?;
            Ok((0..num_classes)
                .map(|i| Dynamic::from(if i == index { 1.0_f64 } else { 0.0_f64 }))
                .collect())
        },
    );

    engine.register_fn("label_encode", |value: &str, mapping: rhai::Map| -> i64 {
        mapping
            .get(value)
            .and_then(|v| v.as_int().ok())
            .unwrap_or(-1)
    });

    engine.register_fn(
        "fill_missing",
        |arr: rhai::Array, strategy: &str| -> rhai::Array {
            let values: Vec<f64> = arr.iter().map(to_f64).collect();
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
                        sorted
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let mid = sorted.len() / 2;
                        if sorted.len().is_multiple_of(2) {
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
    engine.register_fn("decode_image", |data: &str| -> RhaiResult<rhai::Map> {
        use base64::Engine as B64Engine;
        // base64 expands by 4/3: reject before decoding anything.
        if data.len() / 4 * 3 > MAX_IMAGE_BYTES {
            return Err(format!("image too large (max {MAX_IMAGE_BYTES} bytes)").into());
        }
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data)
            .map_err(|e| format!("base64 decode failed: {e}"))?;

        // Enforce dimension/allocation limits *during* decoding so a small
        // compressed "image bomb" cannot allocate gigabytes.
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(MAX_IMAGE_PIXELS);
        limits.max_image_height = Some(MAX_IMAGE_PIXELS);
        limits.max_alloc = Some(256 * 1024 * 1024);
        let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|e| format!("image decode failed: {e}"))?;
        reader.limits(limits);
        let img = reader
            .decode()
            .map_err(|e| format!("image decode failed: {e}"))?;

        let (w, h) = (img.width(), img.height());
        let rgb = img.to_rgb8();
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
    });

    engine.register_fn(
        "resize_image",
        |pixels: rhai::Array,
         src_h: i64,
         src_w: i64,
         dst_h: i64,
         dst_w: i64,
         channels: i64|
         -> RhaiResult<rhai::Array> {
            let (src, src_len) = dims("resize_image", &[src_h, src_w, channels])?;
            let (dst_dims, dst_len) = dims("resize_image", &[dst_h, dst_w, channels])?;
            expect_len("resize_image", pixels.len(), src_len)?;
            let (sh, sw, ch) = (src[0], src[1], src[2]);
            let (dh, dw) = (dst_dims[0], dst_dims[1]);
            if src_len == 0 {
                return Ok(vec![Dynamic::from(0.0_f64); dst_len]);
            }
            let src: Vec<f64> = pixels.iter().map(to_f64).collect();
            let mut dst = vec![0.0f64; dst_len];
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
                    let y0 = (sy.floor() as usize).min(sh - 1);
                    let x0 = (sx.floor() as usize).min(sw - 1);
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
            Ok(dst.into_iter().map(Dynamic::from).collect())
        },
    );

    engine.register_fn(
        "normalize_image",
        |pixels: rhai::Array, mean: rhai::Array, std: rhai::Array| -> RhaiResult<rhai::Array> {
            let mean_vals: Vec<f64> = mean.iter().map(to_f64).collect();
            let std_vals: Vec<f64> = std.iter().map(to_f64).collect();
            let ch = mean_vals.len();
            if ch == 0 || std_vals.len() != ch {
                return Err(
                    "normalize_image: mean and std must be non-empty and equal length".into(),
                );
            }
            Ok(pixels
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let c = i % ch;
                    let s = if std_vals[c] != 0.0 { std_vals[c] } else { 1.0 };
                    Dynamic::from((to_f64(v) - mean_vals[c]) / s)
                })
                .collect())
        },
    );

    engine.register_fn(
        "image_to_chw",
        |pixels: rhai::Array, h: i64, w: i64, c: i64| -> RhaiResult<rhai::Array> {
            let (d, total) = dims("image_to_chw", &[h, w, c])?;
            expect_len("image_to_chw", pixels.len(), total)?;
            let (height, width, channels) = (d[0], d[1], d[2]);
            let src: Vec<f64> = pixels.iter().map(to_f64).collect();
            let mut dst = vec![0.0f64; total];
            for y in 0..height {
                for x in 0..width {
                    for ch in 0..channels {
                        dst[ch * height * width + y * width + x] =
                            src[(y * width + x) * channels + ch];
                    }
                }
            }
            Ok(dst.into_iter().map(Dynamic::from).collect())
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
         -> RhaiResult<rhai::Array> {
            let (src, src_len) = dims("center_crop", &[src_h, src_w, channels])?;
            let (crop, crop_len) = dims("center_crop", &[crop_h, crop_w, channels])?;
            expect_len("center_crop", pixels.len(), src_len)?;
            let (sh, sw, c) = (src[0], src[1], src[2]);
            let (ch_val, cw) = (crop[0], crop[1]);
            if ch_val > sh || cw > sw {
                return Err(format!(
                    "center_crop: crop {ch_val}x{cw} larger than source {sh}x{sw}"
                )
                .into());
            }
            let start_y = (sh - ch_val) / 2;
            let start_x = (sw - cw) / 2;
            let mut dst = Vec::with_capacity(crop_len);
            for y in 0..ch_val {
                for x in 0..cw {
                    let base = ((start_y + y) * sw + start_x + x) * c;
                    dst.extend(pixels[base..base + c].iter().cloned());
                }
            }
            Ok(dst)
        },
    );

    engine.register_fn(
        "grayscale",
        |pixels: rhai::Array, h: i64, w: i64| -> RhaiResult<rhai::Array> {
            let (_, total) = dims("grayscale", &[h, w])?;
            expect_len("grayscale", pixels.len(), total.saturating_mul(3))?;
            let src: Vec<f64> = pixels.iter().map(to_f64).collect();
            Ok(src
                .chunks_exact(3)
                .map(|p| Dynamic::from(0.2989 * p[0] + 0.5870 * p[1] + 0.1140 * p[2]))
                .collect())
        },
    );

    engine.register_fn(
        "nms",
        |boxes: rhai::Array, scores: rhai::Array, iou_threshold: f64| -> RhaiResult<rhai::Array> {
            let n = scores.len();
            expect_len("nms", boxes.len(), n)?;
            if n > MAX_NMS_BOXES {
                return Err(format!("nms: {n} boxes exceeds limit {MAX_NMS_BOXES}").into());
            }
            let sc: Vec<f64> = scores.iter().map(to_f64).collect();
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
                sc[b]
                    .partial_cmp(&sc[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut keep: Vec<i64> = Vec::new();
            let mut suppressed = vec![false; n];

            for (pos, &i) in indices.iter().enumerate() {
                if suppressed[i] {
                    continue;
                }
                keep.push(i as i64);
                // Only lower-scored boxes can be suppressed by box i.
                for &j in &indices[pos + 1..] {
                    if !suppressed[j] && compute_iou(&bx[i], &bx[j]) >= iou_threshold {
                        suppressed[j] = true;
                    }
                }
            }
            Ok(keep.into_iter().map(Dynamic::from).collect())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> Engine {
        let mut engine = Engine::new();
        let tokenizer = Arc::new(None);
        register_all(&mut engine, tokenizer);
        engine
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn test_softmax() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>("softmax([1.0, 2.0, 3.0])")
            .unwrap();
        let vals: Vec<f64> = result.iter().map(|v| v.as_float().unwrap()).collect();
        let sum: f64 = vals.iter().sum();
        assert!(approx_eq(sum, 1.0));
        assert!(vals[2] > vals[1]);
        assert!(vals[1] > vals[0]);
    }

    #[test]
    fn test_softmax_integers() {
        let engine = make_engine();
        let result = engine.eval::<rhai::Array>("softmax([1, 2, 3])").unwrap();
        let vals: Vec<f64> = result.iter().map(|v| v.as_float().unwrap()).collect();
        assert!(approx_eq(vals.iter().sum::<f64>(), 1.0));
    }

    #[test]
    fn test_sigmoid() {
        let engine = make_engine();
        let result = engine.eval::<rhai::Array>("sigmoid([0.0])").unwrap();
        assert!(approx_eq(result[0].as_float().unwrap(), 0.5));

        let result2 = engine
            .eval::<rhai::Array>("sigmoid([100.0, -100.0])")
            .unwrap();
        assert!(result2[0].as_float().unwrap() > 0.99);
        assert!(result2[1].as_float().unwrap() < 0.01);
    }

    #[test]
    fn test_argmax() {
        let engine = make_engine();
        assert_eq!(engine.eval::<i64>("argmax([1.0, 3.0, 2.0])").unwrap(), 1);
        assert_eq!(engine.eval::<i64>("argmax([5, 1, 3])").unwrap(), 0);
    }

    #[test]
    fn test_argmin() {
        let engine = make_engine();
        assert_eq!(engine.eval::<i64>("argmin([3.0, 1.0, 2.0])").unwrap(), 1);
        assert_eq!(engine.eval::<i64>("argmin([5, 1, 3])").unwrap(), 1);
    }

    #[test]
    fn test_topk() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>("topk([1.0, 5.0, 3.0, 4.0, 2.0], 2)")
            .unwrap();
        assert_eq!(result.len(), 2);
        let first = result[0].clone().cast::<rhai::Map>();
        let second = result[1].clone().cast::<rhai::Map>();
        assert_eq!(first.get("index").unwrap().as_int().unwrap(), 1);
        assert!(approx_eq(
            first.get("value").unwrap().as_float().unwrap(),
            5.0
        ));
        assert_eq!(second.get("index").unwrap().as_int().unwrap(), 3);
    }

    #[test]
    fn test_threshold() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>("threshold([0.3, 0.7, 0.5], 0.5)")
            .unwrap();
        let vals: Vec<f64> = result.iter().map(|v| v.as_float().unwrap()).collect();
        assert_eq!(vals, vec![0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_clip() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>("clip([-1.0, 0.5, 2.0], 0.0, 1.0)")
            .unwrap();
        let vals: Vec<f64> = result.iter().map(|v| v.as_float().unwrap()).collect();
        assert_eq!(vals, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_pad_sequence_padding() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>("pad_sequence([1, 2, 3], 5, 0)")
            .unwrap();
        let vals: Vec<i64> = result.iter().map(|v| v.as_int().unwrap()).collect();
        assert_eq!(vals, vec![1, 2, 3, 0, 0]);
    }

    #[test]
    fn test_pad_sequence_truncation() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>("pad_sequence([1, 2, 3, 4, 5], 3, 0)")
            .unwrap();
        let vals: Vec<i64> = result.iter().map(|v| v.as_int().unwrap()).collect();
        assert_eq!(vals, vec![1, 2, 3]);
    }

    #[test]
    fn test_text_lower() {
        let engine = make_engine();
        let result = engine
            .eval::<String>(r#"text_lower("Hello WORLD")"#)
            .unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_regex_replace() {
        let engine = make_engine();
        let result = engine
            .eval::<String>(r#"regex_replace("foo123bar456", "[0-9]+", "")"#)
            .unwrap();
        assert_eq!(result, "foobar");
    }

    #[test]
    fn test_decode_tokens_no_tokenizer() {
        let engine = make_engine();
        let result = engine.eval::<String>("decode_tokens([101, 2023])");
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_minmax() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>(r#"normalize([2.0, 4.0, 6.0, 8.0, 10.0], "minmax")"#)
            .unwrap();
        let vals: Vec<f64> = result.iter().map(|v| v.as_float().unwrap()).collect();
        assert!(approx_eq(vals[0], 0.0));
        assert!(approx_eq(vals[4], 1.0));
        assert!(approx_eq(vals[2], 0.5));
    }

    #[test]
    fn test_normalize_l2() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>(r#"normalize([3.0, 4.0], "l2")"#)
            .unwrap();
        let vals: Vec<f64> = result.iter().map(|v| v.as_float().unwrap()).collect();
        assert!(approx_eq(vals[0], 0.6));
        assert!(approx_eq(vals[1], 0.8));
    }

    #[test]
    fn test_standardize() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>("standardize([10.0, 20.0, 30.0], 20.0, 10.0)")
            .unwrap();
        let vals: Vec<f64> = result.iter().map(|v| v.as_float().unwrap()).collect();
        assert!(approx_eq(vals[0], -1.0));
        assert!(approx_eq(vals[1], 0.0));
        assert!(approx_eq(vals[2], 1.0));
    }

    #[test]
    fn test_one_hot() {
        let engine = make_engine();
        let result = engine.eval::<rhai::Array>("one_hot(2, 4)").unwrap();
        let vals: Vec<f64> = result.iter().map(|v| v.as_float().unwrap()).collect();
        assert_eq!(vals, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_label_encode() {
        let engine = make_engine();
        let result = engine
            .eval::<i64>(r#"let m = #{"cat": 0, "dog": 1, "bird": 2}; label_encode("dog", m)"#)
            .unwrap();
        assert_eq!(result, 1);

        let unknown = engine
            .eval::<i64>(r#"let m = #{"cat": 0}; label_encode("fish", m)"#)
            .unwrap();
        assert_eq!(unknown, -1);
    }

    #[test]
    fn test_fill_missing_zero() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>(r#"fill_missing([1.0, 0.0/0.0, 3.0], "zero")"#)
            .unwrap();
        let vals: Vec<f64> = result.iter().map(|v| v.as_float().unwrap()).collect();
        assert!(approx_eq(vals[0], 1.0));
        assert!(approx_eq(vals[1], 0.0));
        assert!(approx_eq(vals[2], 3.0));
    }

    #[test]
    fn test_fill_missing_mean() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>(r#"fill_missing([2.0, 0.0/0.0, 4.0], "mean")"#)
            .unwrap();
        let vals: Vec<f64> = result.iter().map(|v| v.as_float().unwrap()).collect();
        assert!(approx_eq(vals[1], 3.0));
    }

    #[test]
    fn test_fill_missing_median() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>(r#"fill_missing([1.0, 0.0/0.0, 5.0, 3.0], "median")"#)
            .unwrap();
        let vals: Vec<f64> = result.iter().map(|v| v.as_float().unwrap()).collect();
        assert!(approx_eq(vals[1], 3.0));
    }

    #[test]
    fn test_decode_image() {
        let engine = make_engine();
        use image::{ImageBuffer, Rgb};
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(2, 2, |x, y| {
            if x == 0 && y == 0 {
                Rgb([255, 0, 0])
            } else {
                Rgb([0, 0, 0])
            }
        });
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        use base64::Engine as B64Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());

        let script = format!(r#"decode_image("{b64}")"#);
        let result = engine.eval::<rhai::Map>(&script).unwrap();
        assert_eq!(result.get("width").unwrap().as_int().unwrap(), 2);
        assert_eq!(result.get("height").unwrap().as_int().unwrap(), 2);
        assert_eq!(result.get("channels").unwrap().as_int().unwrap(), 3);
        let pixels = result
            .get("pixels")
            .unwrap()
            .clone()
            .into_typed_array::<Dynamic>()
            .unwrap();
        assert_eq!(pixels.len(), 12);
        assert!(approx_eq(to_f64(&pixels[0]), 255.0));
        assert!(approx_eq(to_f64(&pixels[1]), 0.0));
    }

    #[test]
    fn test_resize_image() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>(
                "resize_image([255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, 2, 1, 1, 3)",
            )
            .unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_normalize_image() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>(
                "normalize_image([100.0, 150.0, 200.0, 100.0, 150.0, 200.0], [100.0, 100.0, 100.0], [50.0, 50.0, 50.0])",
            )
            .unwrap();
        let vals: Vec<f64> = result.iter().map(|v| v.as_float().unwrap()).collect();
        assert!(approx_eq(vals[0], 0.0));
        assert!(approx_eq(vals[1], 1.0));
        assert!(approx_eq(vals[2], 2.0));
    }

    #[test]
    fn test_image_to_chw() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>("image_to_chw([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1, 2, 3)")
            .unwrap();
        let vals: Vec<f64> = result.iter().map(|v| v.as_float().unwrap()).collect();
        assert_eq!(vals, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_center_crop() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>(
                "center_crop([\
                    1.0, 2.0, 3.0,  4.0, 5.0, 6.0,  7.0, 8.0, 9.0,\
                    10.0,11.0,12.0, 13.0,14.0,15.0, 16.0,17.0,18.0,\
                    19.0,20.0,21.0, 22.0,23.0,24.0, 25.0,26.0,27.0\
                ], 3, 3, 1, 1, 3)",
            )
            .unwrap();
        let vals: Vec<f64> = result.iter().map(|v| v.as_float().unwrap()).collect();
        assert_eq!(vals, vec![13.0, 14.0, 15.0]);
    }

    #[test]
    fn test_grayscale() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>("grayscale([255.0, 0.0, 0.0, 0.0, 255.0, 0.0], 1, 2)")
            .unwrap();
        let vals: Vec<f64> = result.iter().map(|v| v.as_float().unwrap()).collect();
        assert!(approx_eq(vals[0], 0.2989 * 255.0));
        assert!(approx_eq(vals[1], 0.5870 * 255.0));
    }

    #[test]
    fn test_nms() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>(
                r#"nms(
                    [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]],
                    [0.9, 0.8, 0.7],
                    0.5
                )"#,
            )
            .unwrap();
        let vals: Vec<i64> = result.iter().map(|v| v.as_int().unwrap()).collect();
        assert_eq!(vals[0], 0);
        assert!(vals.contains(&2));
        assert!(!vals.contains(&1));
    }

    #[test]
    fn test_builtins_reject_bad_arguments() {
        let engine = make_engine();
        for script in [
            "resize_image([1.0, 2.0], 2, 2, 4, 4, 1)",
            "resize_image([1.0], 1, 1, -4, 4, 1)",
            "image_to_chw([1.0, 2.0, 3.0], 2, 2, 3)",
            "center_crop([1.0, 2.0, 3.0, 4.0], 2, 2, 3, 3, 1)",
            "grayscale([1.0, 2.0], 1, 1)",
            "normalize_image([1.0], [], [])",
            "nms([[0.0, 0.0, 1.0, 1.0]], [0.9, 0.8], 0.5)",
            "one_hot(1, 999999999999)",
            "pad_sequence([1], 999999999999, 0)",
            "decode_tokens([1])",
        ] {
            assert!(
                engine.eval::<Dynamic>(script).is_err(),
                "{script} should fail"
            );
        }
    }

    #[test]
    fn test_center_crop_odd_size() {
        let engine = make_engine();
        let result = engine
            .eval::<rhai::Array>("center_crop([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 3, 1, 1, 1)")
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_int().unwrap(), 5);
    }

    #[test]
    fn test_decode_image_rejects_oversized_dimensions() {
        use base64::Engine as B64Engine;
        let img = image::RgbImage::new(MAX_IMAGE_PIXELS + 1, 1);
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
        let engine = make_engine();
        assert!(engine
            .eval::<rhai::Map>(&format!(r#"decode_image("{b64}")"#))
            .is_err());
    }
}
