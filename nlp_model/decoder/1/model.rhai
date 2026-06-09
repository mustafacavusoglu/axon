fn execute(inputs) {
    let logits = inputs.get("logits").as_f64();
    let dims = inputs.get("logits").shape;
    let seq_len = dims[1];
    let num_labels = dims[2];
    let labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"];

    let output = "";
    let i = 0;
    while i < seq_len {
        let base = i * num_labels;
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
            output += "token " + i + ": " + entity;
        }
        i += 1;
    }

    return #{ "entities": create_tensor_string("entities", [1], [output]) };
}
