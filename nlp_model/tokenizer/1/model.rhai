fn execute(inputs) {
    let text_tensor = inputs.get("text");
    let text = text_lower(text_tensor.as_string());

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
        let lower_word = text_lower(word);
        if lower_word in vocab {
            input_ids.push(vocab[lower_word]);
        } else {
            input_ids.push(unk);
        }
    }

    input_ids.push(sep);

    let max_len = 128;
    let input_ids = pad_sequence(input_ids, max_len, 0);

    let n = input_ids.len();
    let attention_mask = [];
    let token_type_ids = [];
    let j = 0;
    while j < n {
        if input_ids[j] != 0 {
            attention_mask.push(1);
        } else {
            attention_mask.push(0);
        }
        token_type_ids.push(0);
        j += 1;
    }

    let word_str = "";
    let k = 0;
    while k < words.len() {
        if k > 0 { word_str += "|"; }
        word_str += words[k];
        k += 1;
    }

    return #{
        "input_ids": create_tensor_i64("input_ids", [1, n], input_ids),
        "attention_mask": create_tensor_i64("attention_mask", [1, n], attention_mask),
        "token_type_ids": create_tensor_i64("token_type_ids", [1, n], token_type_ids),
        "words": create_tensor_string("words", [1], [word_str]),
    };
}
