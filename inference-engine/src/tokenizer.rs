use std::collections::HashMap;
use std::path::Path;

pub struct Tokenizer {
    vocab: HashMap<String, i64>,
    unk_token_id: i64,
    pad_token_id: i64,
    cls_token_id: i64,
    sep_token_id: i64,
    max_len: usize,
}

impl Tokenizer {
    pub fn load(vocab_path: &Path, tokenizer_json_path: &Path) -> anyhow::Result<Self> {
        let mut vocab = HashMap::new();

        let content = std::fs::read_to_string(vocab_path)?;
        for (idx, line) in content.lines().enumerate() {
            let token = line.trim();
            if !token.is_empty() {
                vocab.insert(token.to_string(), idx as i64);
            }
        }

        let extra: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(tokenizer_json_path)?)?;

        let unk_token = extra["model"]["unk_token"]
            .as_str()
            .unwrap_or("[UNK]");
        let pad_token = extra["model"]["pad_token"]
            .as_str()
            .unwrap_or("[PAD]");
        let cls_token = extra["model"]["cls_token"]
            .as_str()
            .unwrap_or("[CLS]");
        let sep_token = extra["model"]["sep_token"]
            .as_str()
            .unwrap_or("[SEP]");
        let max_len = extra["model"]["max_length"]
            .as_u64()
            .unwrap_or(512) as usize;

        let unk_token_id = *vocab.get(unk_token).unwrap_or(&0);
        let pad_token_id = *vocab.get(pad_token).unwrap_or(&0);
        let cls_token_id = *vocab.get(cls_token).unwrap_or(&0);
        let sep_token_id = *vocab.get(sep_token).unwrap_or(&0);

        tracing::info!(
            vocab_size = vocab.len(),
            unk_token,
            pad_token,
            cls_token,
            sep_token,
            "tokenizer loaded"
        );

        Ok(Self {
            vocab,
            unk_token_id,
            pad_token_id,
            cls_token_id,
            sep_token_id,
            max_len,
        })
    }

    pub fn encode(&self, text: &str) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let tokens = self.tokenize(text);
        let seq_len = tokens.len().min(self.max_len - 2);

        let mut input_ids = Vec::with_capacity(seq_len + 2);
        let mut attention_mask = Vec::with_capacity(seq_len + 2);

        input_ids.push(self.cls_token_id);
        attention_mask.push(1);

        for token in tokens.iter().take(seq_len) {
            let id = self.vocab.get(token).copied().unwrap_or(self.unk_token_id);
            input_ids.push(id);
            attention_mask.push(1);
        }

        input_ids.push(self.sep_token_id);
        attention_mask.push(1);

        let token_type_ids = vec![0i64; input_ids.len()];

        (input_ids, attention_mask, token_type_ids)
    }

    pub fn decode(&self, ids: &[i64]) -> String {
        let inv_vocab: HashMap<i64, &str> =
            self.vocab.iter().map(|(k, v)| (*v, k.as_str())).collect();

        let mut tokens: Vec<String> = Vec::new();
        for &id in ids {
            if let Some(&token) = inv_vocab.get(&id) {
                if token.starts_with("##") {
                    if let Some(last) = tokens.last_mut() {
                        last.push_str(&token[2..]);
                    }
                } else {
                    tokens.push(token.to_string());
                }
            }
        }

        tokens.join(" ").replace(" ##", "")
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let text = text.trim();

        for word in split_words(text) {
            if word.is_empty() {
                continue;
            }

            if self.vocab.contains_key(&word) {
                tokens.push(word);
            } else {
                tokens.extend(self.wordpiece(&word));
            }
        }

        tokens
    }

    fn wordpiece(&self, word: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let mut start = 0;

        while start < chars.len() {
            let mut end = chars.len();
            let mut found = false;

            while end > start {
                let sub: String = chars[start..end].iter().collect();
                let lookup = if start == 0 {
                    sub.clone()
                } else {
                    format!("##{}", sub)
                };

                if self.vocab.contains_key(&lookup) {
                    tokens.push(lookup);
                    found = true;
                    start = end;
                    break;
                }
                end -= 1;
            }

            if !found {
                tokens.push("[UNK]".to_string());
                break;
            }
        }

        tokens
    }
}

fn split_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_whitespace() {
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
        } else if ch.is_ascii_punctuation() && ch != '-' && ch != '#' {
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
            words.push(ch.to_string());
        } else {
            current.push(ch);
        }
    }

    if !current.is_empty() {
        words.push(current);
    }

    words
}
