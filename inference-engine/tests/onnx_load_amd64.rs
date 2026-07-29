use std::path::PathBuf;
use std::time::Instant;

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

    let thread_opts = ort::environment::GlobalThreadPoolOptions::default()
        .with_inter_threads(1)
        .unwrap()
        .with_intra_threads(1)
        .unwrap()
        .with_spin_control(false)
        .unwrap();
    ort::init().with_global_thread_pool(thread_opts).commit();

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

    let session_start = Instant::now();
    let result = Session::builder()
        .expect("builder")
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .expect("opt level")
        .with_intra_threads(1)
        .expect("intra")
        .with_inter_threads(1)
        .expect("inter")
        .commit_from_memory(&model_bytes);

    match &result {
        Ok(_) => eprintln!("[test] session created in {:?}", session_start.elapsed()),
        Err(e) => eprintln!(
            "[test] session FAILED in {:?}: {e}",
            session_start.elapsed()
        ),
    }

    let _session = result.expect("failed to create ONNX session");
    eprintln!("[test] total load time: {:?}", start.elapsed());
}
