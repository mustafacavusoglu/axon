use std::collections::HashMap;
use std::time::{Duration, Instant};

const MAX_FAILURES: u32 = 3;
const RESET_TIMEOUT: Duration = Duration::from_secs(300);

pub struct CircuitBreaker {
    failures: HashMap<String, (u32, Instant)>,
}

impl CircuitBreaker {
    pub fn new() -> Self {
        Self {
            failures: HashMap::new(),
        }
    }

    pub fn is_open(&self, key: &str) -> bool {
        if let Some((count, last_fail)) = self.failures.get(key) {
            if *count >= MAX_FAILURES {
                if last_fail.elapsed() < RESET_TIMEOUT {
                    return true;
                }
            }
        }
        false
    }

    pub fn record_failure(&mut self, key: &str) {
        let entry = self.failures.entry(key.to_string()).or_insert((0, Instant::now()));
        entry.0 += 1;
        entry.1 = Instant::now();
    }

    pub fn record_success(&mut self, key: &str) {
        self.failures.remove(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_opens_after_max_failures() {
        let mut cb = CircuitBreaker::new();
        let key = "model_a@v1";

        assert!(!cb.is_open(key));
        cb.record_failure(key);
        cb.record_failure(key);
        assert!(!cb.is_open(key));
        cb.record_failure(key);
        assert!(cb.is_open(key));
    }

    #[test]
    fn test_success_resets() {
        let mut cb = CircuitBreaker::new();
        let key = "model_b@v1";

        cb.record_failure(key);
        cb.record_failure(key);
        cb.record_failure(key);
        assert!(cb.is_open(key));

        cb.record_success(key);
        assert!(!cb.is_open(key));
    }
}
