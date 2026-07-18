# Builtin Functions

Axon provides 23 builtin functions available in all Rhai BLS scripts. These cover common preprocessing and postprocessing operations for ML, NLP, CV, and tabular data workloads.

All functions work with Rhai's native types (`Array`, `int`, `float`, `string`, `Map`). Integer and float values are automatically handled — you can pass `[1, 2, 3]` or `[1.0, 2.0, 3.0]`.

---

## ML / Math Functions

### `softmax(arr) → Array`

Converts logits to probability distribution. Numerically stable (subtracts max before exp).

```rhai
let probs = softmax([2.0, 1.0, 0.1]);
// → [0.659, 0.242, 0.099] (sums to 1.0)
```

### `sigmoid(arr) → Array`

Applies sigmoid activation to each element: `1 / (1 + exp(-x))`.

```rhai
let out = sigmoid([0.0, 2.0, -2.0]);
// → [0.5, 0.881, 0.119]
```

### `argmax(arr) → int`

Returns index of maximum value.

```rhai
let idx = argmax([0.1, 0.7, 0.2]);
// → 1
```

### `argmin(arr) → int`

Returns index of minimum value.

```rhai
let idx = argmin([0.5, 0.1, 0.4]);
// → 1
```

### `topk(arr, k) → Array`

Returns top-K elements as array of maps with `index` and `value`.

```rhai
let top = topk([0.1, 0.9, 0.5, 0.8], 2);
// → [#{index: 1, value: 0.9}, #{index: 3, value: 0.8}]
```

### `threshold(arr, val) → Array`

Binary thresholding. Returns 1.0 for values >= threshold, 0.0 otherwise.

```rhai
let binary = threshold([0.3, 0.7, 0.5], 0.5);
// → [0.0, 1.0, 1.0]
```

### `clip(arr, min, max) → Array`

Clamps all values to [min, max] range.

```rhai
let clamped = clip([-1.0, 0.5, 2.0], 0.0, 1.0);
// → [0.0, 0.5, 1.0]
```

---

## NLP Functions

### `tokenize(text) → Map`

Tokenizes text using HuggingFace tokenizer. Requires `tokenizer.json` in the script directory.

Returns a map with `input_ids` and `attention_mask` as `RhaiTensor` objects.

```rhai
let result = tokenize("Hello world");
// result.input_ids → RhaiTensor [1, N] INT64
// result.attention_mask → RhaiTensor [1, N] INT64
```

### `decode_tokens(ids) → string`

Converts token IDs back to text using the loaded tokenizer.

```rhai
let text = decode_tokens([101, 7592, 2088, 102]);
// → "hello world"
```

### `pad_sequence(arr, target_len, pad_value) → Array`

Pads array to `target_len` with `pad_value`, or truncates if longer.

```rhai
let padded = pad_sequence([1, 2, 3], 5, 0);
// → [1, 2, 3, 0, 0]

let truncated = pad_sequence([1, 2, 3, 4, 5], 3, 0);
// → [1, 2, 3]
```

### `text_lower(text) → string`

Converts text to lowercase.

```rhai
let lower = text_lower("Hello WORLD");
// → "hello world"
```

### `regex_replace(text, pattern, replacement) → string`

Replaces all regex matches in text.

```rhai
let clean = regex_replace("foo123bar456", "[0-9]+", "");
// → "foobar"

let normalized = regex_replace("  extra   spaces  ", "\\s+", " ");
// → " extra spaces "
```

---

## Tabular / Normalization Functions

### `normalize(arr, method) → Array`

Normalizes array. Methods: `"minmax"` (0-1 range) or `"l2"` (unit vector).

```rhai
let mm = normalize([2.0, 4.0, 6.0, 8.0, 10.0], "minmax");
// → [0.0, 0.25, 0.5, 0.75, 1.0]

let l2 = normalize([3.0, 4.0], "l2");
// → [0.6, 0.8]
```

### `standardize(arr, mean, std) → Array`

Z-score standardization: `(x - mean) / std`.

```rhai
let z = standardize([10.0, 20.0, 30.0], 20.0, 10.0);
// → [-1.0, 0.0, 1.0]
```

### `one_hot(index, num_classes) → Array`

Creates one-hot encoded vector.

```rhai
let encoded = one_hot(2, 4);
// → [0.0, 0.0, 1.0, 0.0]
```

### `label_encode(value, mapping) → int`

Maps categorical string to integer. Returns -1 if not found.

```rhai
let mapping = #{"cat": 0, "dog": 1, "bird": 2};
let encoded = label_encode("dog", mapping);
// → 1

let unknown = label_encode("fish", mapping);
// → -1
```

### `fill_missing(arr, strategy) → Array`

Fills NaN values using specified strategy: `"zero"`, `"mean"`, or `"median"`.

```rhai
let filled = fill_missing([1.0, 0.0/0.0, 5.0, 3.0], "mean");
// → [1.0, 3.0, 5.0, 3.0]  (mean of valid = 3.0)

let median_filled = fill_missing([1.0, 0.0/0.0, 5.0, 3.0], "median");
// → [1.0, 3.0, 5.0, 3.0]  (median of [1,3,5] = 3.0)
```

---

## Computer Vision Functions

### `decode_image(base64_string) → Map`

Decodes a base64-encoded JPEG/PNG image to pixels.

Returns: `#{pixels: Array<f64>, width: int, height: int, channels: 3}`

Pixels are in HWC format with values 0-255.

```rhai
let img = decode_image(image_data);
// img.pixels → [255.0, 0.0, 0.0, ...]  (RGB flat array)
// img.width → 224
// img.height → 224
// img.channels → 3
```

### `resize_image(pixels, src_h, src_w, dst_h, dst_w, channels) → Array`

Bilinear interpolation resize. Input/output in HWC format.

```rhai
let resized = resize_image(img.pixels, img.height, img.width, 224, 224, 3);
// → 224*224*3 = 150528 elements
```

### `normalize_image(pixels, mean, std) → Array`

Per-channel normalization: `(pixel - mean[c]) / std[c]`.

```rhai
// ImageNet normalization (assuming pixels already /255.0)
let normalized = normalize_image(pixels,
    [0.485, 0.456, 0.406],  // mean
    [0.229, 0.224, 0.225]   // std
);
```

### `image_to_chw(pixels, h, w, c) → Array`

Converts HWC layout to CHW layout (required by most ONNX models).

```rhai
// Input: [R,G,B, R,G,B, ...] (HWC)
// Output: [R,R,..., G,G,..., B,B,...] (CHW)
let chw = image_to_chw(pixels, 224, 224, 3);
```

### `center_crop(pixels, src_h, src_w, crop_h, crop_w, channels) → Array`

Crops the center region of an image.

```rhai
let cropped = center_crop(pixels, 256, 256, 224, 224, 3);
// Crops center 224x224 from a 256x256 image
```

### `grayscale(pixels, h, w) → Array`

Converts RGB to grayscale using luminance formula: `0.2989*R + 0.587*G + 0.114*B`.

```rhai
let gray = grayscale(rgb_pixels, 224, 224);
// → 224*224 = 50176 elements (single channel)
```

### `nms(boxes, scores, iou_threshold) → Array`

Non-Maximum Suppression for object detection postprocessing.

- `boxes`: Array of `[x1, y1, x2, y2]` arrays
- `scores`: Array of confidence scores
- Returns: Array of kept indices

```rhai
let boxes = [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]];
let scores = [0.9, 0.8, 0.7];
let kept = nms(boxes, scores, 0.5);
// → [0, 2]  (box 1 suppressed by box 0 due to high IoU)
```

---

## Complete Pipeline Example

### Image Classification Preprocessor
```rhai
fn execute(inputs) {
    let raw = inputs.get("image").as_string();

    let img = decode_image(raw);
    let resized = resize_image(img.pixels, img.height, img.width, 224, 224, 3);

    // Scale to 0-1 and apply ImageNet normalization
    let scaled = clip(resized.map(|x| x / 255.0), 0.0, 1.0);
    let normalized = normalize_image(scaled, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    let chw = image_to_chw(normalized, 224, 224, 3);

    return #{
        "input": create_tensor_f64("input", [1, 3, 224, 224], chw),
    };
}
```

### Sentiment Analysis Pipeline
```rhai
fn execute(inputs) {
    let text = inputs.get("text").as_string();
    let clean = text_lower(regex_replace(text, "[^a-zA-Z\\s]", ""));

    let tokens = tokenize(clean);
    let model_out = infer("sentiment_model", tokens);

    let logits = model_out.get("logits").as_f64();
    let probs = softmax(logits);
    let label = argmax(probs);

    return #{
        "label": create_tensor_i64("label", [1], [label]),
        "confidence": create_tensor_f64("confidence", [1], [probs[label]]),
    };
}
```
