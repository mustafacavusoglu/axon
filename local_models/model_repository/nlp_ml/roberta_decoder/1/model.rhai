fn execute(inputs) {
    let logits = inputs.get("logits").as_f64();
    let probs = softmax(logits);
    let label = argmax(probs);

    return #{
        "label": create_tensor_i64("label", [1], [label]),
        "probs": create_tensor_f64("probs", [3], probs),
    };
}
