use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ort::session::Session;

fn try_load(path: &Path, label: &str) -> bool {
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    println!("[{label}] file={} size={size} bytes", path.display());

    let start = Instant::now();
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            println!("[{label}] READ ERROR: {e}");
            return false;
        }
    };
    println!("[{label}] read in {:?}", start.elapsed());

    let (tx, rx) = mpsc::channel();
    let b = bytes.clone();
    let name = label.to_string();
    std::thread::spawn(move || {
        let r = Session::builder()
            .expect("builder")
            .commit_from_memory(&b);
        let _ = tx.send((name, r));
    });

    match rx.recv_timeout(Duration::from_secs(15)) {
        Ok((_, Ok(session))) => {
            drop(session);
            println!("[{label}] ✅ OK in {:?}", start.elapsed());
            true
        }
        Ok((_, Err(e))) => {
            println!("[{label}] ❌ ERROR: {e}");
            false
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            println!("[{label}] ⏱ TIMEOUT — commit_from_memory hung >15s");
            false
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            println!("[{label}] 💥 PANIC");
            false
        }
    }
}

fn main() {
    println!("=== ONNX Runtime Smoke Test ===");
    println!();

    let models_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../axon/models".to_string());
    let root = Path::new(&models_dir);

    // 1) ort::init
    std::env::set_var("OMP_NUM_THREADS", "1");
    std::env::set_var("OMP_WAIT_POLICY", "PASSIVE");

    let start = Instant::now();
    print!("[init] ort::init()... ");
    let ok = ort::init().commit();
    println!("result={ok} ({:?})", start.elapsed());
    println!();

    // 2) Simple model (y = 2*x + 1, 132 bytes)
    let simple = root.join("test_model/1/model.onnx");
    if simple.exists() {
        try_load(&simple, "simple");
    } else {
        println!("[simple] SKIP — file not found: {}", simple.display());
    }
    println!();

    // 3) CatBoost credit risk (2 MB)
    let cb = root.join("cb_credit_risk/1/model.onnx");
    if cb.exists() {
        try_load(&cb, "cb_credit_risk");
    } else {
        println!("[cb_credit_risk] SKIP — file not found: {}", cb.display());
    }
    println!();

    // 4) BERT safety model (422 MB) — skip by default, pass as 2nd arg
    let safety = std::env::args().nth(2).unwrap_or_default();
    if !safety.is_empty() {
        let p = Path::new(&safety);
        if p.exists() {
            try_load(p, "safety_model");
        }
    }

    println!("=== Done ===");
}
