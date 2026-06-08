fn execute(inputs) {
    let text_tensor = inputs.get("text");
    let text = text_tensor.as_string();

    let tokens = tokenize(text);

    let model_inputs = #{
        "input_ids": tokens.get("input_ids"),
        "attention_mask": tokens.get("attention_mask"),
        "token_type_ids": tokens.get("token_type_ids"),
    };

    return infer("ner_model", model_inputs);
}
