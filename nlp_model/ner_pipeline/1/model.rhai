fn tokenize(text) {
    let lines = read_lines("vocab.txt");
    let vocab = #{};
    let i = 0;
    while i < lines.len() {
        vocab[lines[i]] = i;
        i += 1;
    }

    let cls = if "[CLS]" in vocab { vocab["[CLS]"] } else { 0 };
    let sep = if "[SEP]" in vocab { vocab["[SEP]"] } else { 0 };
    let unk = if "[UNK]" in vocab { vocab["[UNK]"] } else { 0 };

    let words = split_words(text);

    let input_ids = [];
    input_ids.push(cls);

    for word in words {
        if word == "" { continue; }
        if word in vocab {
            input_ids.push(vocab[word]);
        } else {
            input_ids.push(unk);
        }
    }

    input_ids.push(sep);

    let n = input_ids.len();
    let attention_mask = [];
    let token_type_ids = [];
    let j = 0;
    while j < n {
        attention_mask.push(1);
        token_type_ids.push(0);
        j += 1;
    }

    return #{ "ids": input_ids, "mask": attention_mask, "types": token_type_ids, "words": words };
}

fn execute(inputs) {
    let text_tensor = inputs.get("text");
    let text = text_tensor.as_string();

    let tok = tokenize(text);

    let seq_len = tok["ids"].len();
    let ids_tensor = create_tensor_i64("input_ids", [1, seq_len], tok["ids"]);
    let mask_tensor = create_tensor_i64("attention_mask", [1, seq_len], tok["mask"]);
    let type_tensor = create_tensor_i64("token_type_ids", [1, seq_len], tok["types"]);

    let result = infer("ner_model", #{
        "input_ids": ids_tensor,
        "attention_mask": mask_tensor,
        "token_type_ids": type_tensor,
    });

    let logits_tensor = result.get("logits");
    let logits = logits_tensor.as_f64();

    let labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"];
    let num_labels = 7;
    let words = tok["words"];

    let output = "";
    for i in 0..words.len() {
        let base = (i + 1) * num_labels;
        let best_label = 0;
        let best_score = logits[base];
        let j = 1;
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
            output += words[i] + ": " + entity;
        }
    }

    return #{ "entities": create_tensor_string("entities", [1], [output]) };
}
