"""Download BERT Turkish NER model, convert to ONNX, export tokenizer."""
import os
import json
from pathlib import Path

MODEL_NAME = "akdeniz27/bert-base-turkish-cased-ner"
OUTPUT_DIR = Path(__file__).parent / "ner_pipeline" / "1"

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch

    print(f"Downloading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Exporting tokenizer...")
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    vocab_path = OUTPUT_DIR / "vocab.txt"
    if not vocab_path.exists():
        vocab = tokenizer.get_vocab()
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        vocab_path.write_text("\n".join(k for k, _ in sorted_vocab))

    print("Converting to ONNX...")
    dummy_input = tokenizer("Mustafa İstanbul'da yaşıyor.", return_tensors="pt")
    
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    }
    
    if "token_type_ids" in dummy_input:
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch", 1: "sequence"}

    torch.onnx.export(
        model,
        tuple(dummy_input[n] for n in input_names),
        str(OUTPUT_DIR / "model.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    id2label = model.config.id2label
    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    print(f"Done! Files in {OUTPUT_DIR}:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
