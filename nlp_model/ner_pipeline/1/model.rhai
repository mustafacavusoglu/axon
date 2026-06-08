fn execute(inputs) {
    let ids = inputs.get("input_ids");

    let n = ids.shape[1];
    let mask_data = [];
    let type_data = [];
    let i = 0;
    while i < n {
        mask_data.push(1);
        type_data.push(0);
        i += 1;
    }

    let mask_tensor = create_tensor_i64("attention_mask", [1, n], mask_data);
    let type_tensor = create_tensor_i64("token_type_ids", [1, n], type_data);

    return infer("ner_model", #{
        "input_ids": ids,
        "attention_mask": mask_tensor,
        "token_type_ids": type_tensor,
    });
}
