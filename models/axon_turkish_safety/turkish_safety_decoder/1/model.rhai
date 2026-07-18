fn execute(inputs) {
    let logits_tensor = inputs.get("logits");
    let logits = logits_tensor.as_f64();

    let probs = softmax(logits);
    let idx = argmax(probs);
    let score = probs[idx];

    let labels = ["SAFE", "OFFENSIVE", "SELF_HARM", "GROOMING", "BULLYING", "SEXUAL", "THREAT"];
    let label = labels[idx];

    return #{
        "label": create_tensor_string("label", [1], [label]),
        "score": create_tensor_f64("score", [1], [score]),
    };
}
