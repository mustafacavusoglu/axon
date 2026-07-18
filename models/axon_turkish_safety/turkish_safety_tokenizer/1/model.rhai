fn execute(inputs) {
    let text_tensor = inputs.get("text");
    let text = text_tensor.as_string();

    let enc = tokenize(text);

    return #{
        "input_ids":      enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    };
}
