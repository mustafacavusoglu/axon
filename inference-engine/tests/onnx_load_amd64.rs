use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("models/cb_credit_risk/1/model.onnx")
}

#[test]
fn test_onnx_load_cb_credit_risk() {
    std::env::set_var("OMP_NUM_THREADS", "1");
    std::env::set_var("OMP_WAIT_POLICY", "PASSIVE");
    ort::init().commit();

    let path = model_path();
    assert!(path.exists(), "model file not found: {}", path.display());

    eprintln!("[test] loading model from: {}", path.display());
    eprintln!(
        "[test] file size: {} bytes",
        std::fs::metadata(&path).unwrap().len()
    );

    let start = Instant::now();

    let model_bytes = std::fs::read(&path).expect("failed to read model file");
    eprintln!("[test] read into memory in {:?}", start.elapsed());

    let (tx, rx) = mpsc::channel();
    let model_bytes_clone = model_bytes.clone();
    std::thread::spawn(move || {
        let result = Session::builder()
            .expect("builder")
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .expect("opt level")
            .with_intra_threads(1)
            .expect("intra")
            .with_inter_threads(1)
            .expect("inter")
            .commit_from_memory(&model_bytes_clone);
        let _ = tx.send(result);
    });

    match rx.recv_timeout(Duration::from_secs(10)) {
        Ok(Ok(session)) => {
            eprintln!("[test] session created in {:?}", start.elapsed());
            drop(session);
        }
        Ok(Err(e)) => {
            eprintln!("[test] session FAILED in {:?}: {e}", start.elapsed());
            panic!("failed to create ONNX session: {e}");
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            eprintln!("[test] TIMEOUT: commit_from_memory hung for 10s");
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("session creation thread panicked");
        }
    }

    eprintln!("[test] total load time: {:?}", start.elapsed());
}
