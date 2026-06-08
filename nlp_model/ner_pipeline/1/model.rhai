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

    let result = infer("ner_model", #{
        "input_ids": ids,
        "attention_mask": mask_tensor,
        "token_type_ids": type_tensor,
    });

    let logits_tensor = result.get("logits");
    let logits = logits_tensor.as_f64();

    let labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"];
    let num_labels = 7;

    let nums = ids.as_i64();
    let ids_len = nums.len();

    let output = "";
    let i = 1;
    while i < ids_len - 1 {
        let base = i * num_labels;
        let best_label = 0;
        let best_score = logits[base];
        let j = 1;
        let id_val = nums[i];
        while j < num_labels {
            if logits[base + j] > best_score {
                best_label = j;
                best_score = logits[base + j];
            }
            j += 1;
        }

        let entity = labels[best_label];
        if entity != "O" {
            if output != "" { output += "; "; }
            output += "token " + i + " (id=" + id_val + "): " + entity;
        }
        i += 1;
    }

    return #{ "entities": create_tensor_string("entities", [1], [output]) };
}
