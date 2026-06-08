fn tokenize(text) {
    let words = split_text(text);

    let input_ids = [];
    input_ids.push(cls_id());

    for word in words {
        if word == "" { continue; }
        if vocab_has(word) {
            input_ids.push(vocab_id(word));
        } else {
            input_ids.push(unk_id());
        }
    }

    input_ids.push(sep_id());

    let n = input_ids.len();
    let attention_mask = [];
    let token_type_ids = [];
    let i = 0;
    while i < n {
        attention_mask.push(1);
        token_type_ids.push(0);
        i += 1;
    }

    return #{
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    };
}

fn execute(inputs) {
    let text_tensor = inputs.get("text");
    let text = text_tensor.as_string();

    let tokens = tokenize(text);

    let seq_len = tokens["input_ids"].len();
    let ids_tensor = create_tensor_i64("input_ids", [1, seq_len], tokens["input_ids"]);
    let mask_tensor = create_tensor_i64("attention_mask", [1, seq_len], tokens["attention_mask"]);
    let type_tensor = create_tensor_i64("token_type_ids", [1, seq_len], tokens["token_type_ids"]);

    let model_inputs = #{
        "input_ids": ids_tensor,
        "attention_mask": mask_tensor,
        "token_type_ids": type_tensor,
    };

    return infer("ner_model", model_inputs);
}
