use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

fn repo_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn try_load_model(path: &Path) -> Result<Duration, String> {
    eprintln!("[test] model: {}", path.display());
    eprintln!(
        "[test] size: {} bytes",
        std::fs::metadata(path).unwrap().len()
    );

    let start = Instant::now();
    let model_bytes = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
    eprintln!("[test] read into memory in {:?}", start.elapsed());

    let (tx, rx) = mpsc::channel();
    let mb = model_bytes.clone();
    let label = path.display().to_string();
    std::thread::spawn(move || {
        let result = Session::builder()
            .expect("builder")
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .expect("opt level")
            .with_intra_threads(1)
            .expect("intra")
            .with_inter_threads(1)
            .expect("inter")
            .commit_from_memory(&mb);
        let _ = tx.send((label, result));
    });

    match rx.recv_timeout(Duration::from_secs(10)) {
        Ok((_, Ok(session))) => {
            drop(session);
            let elapsed = start.elapsed();
            eprintln!("[test] OK in {:?}", elapsed);
            Ok(elapsed)
        }
        Ok((label, Err(e))) => {
            eprintln!("[test] FAILED in {:?}: {e}", start.elapsed());
            Err(format!("{label}: {e}"))
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            eprintln!("[test] TIMEOUT after 10s");
            Err(format!("{}: timeout after 10s", path.display()))
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            Err(format!("{}: thread panicked", path.display()))
        }
    }
}

#[test]
fn test_onnx_simple_model() {
    std::env::set_var("OMP_NUM_THREADS", "1");
    std::env::set_var("OMP_WAIT_POLICY", "PASSIVE");
    ort::init().commit();

    let path = repo_dir().join("models/test_model/1/model.onnx");
    assert!(path.exists(), "model file not found: {}", path.display());

    match try_load_model(&path) {
        Ok(d) => eprintln!("[test] simple model loaded in {d:?}"),
        Err(e) => panic!("simple model failed: {e}"),
    }
}

#[test]
fn test_onnx_cb_credit_risk() {
    std::env::set_var("OMP_NUM_THREADS", "1");
    std::env::set_var("OMP_WAIT_POLICY", "PASSIVE");
    ort::init().commit();

    let path = repo_dir().join("models/cb_credit_risk/1/model.onnx");
    assert!(path.exists(), "model file not found: {}", path.display());

    match try_load_model(&path) {
        Ok(d) => eprintln!("[test] cb_credit_risk loaded in {d:?}"),
        Err(e) => eprintln!("[test] cb_credit_risk FAILED (expected on some CPUs): {e}"),
    }
}
