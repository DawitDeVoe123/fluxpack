use std::time::Instant;
use fluxpack::{Encoder, Decoder, decode_zero_copy};
use serde_json::json;

fn main() {
    let large = json!({
        "experiment_id": "exp_2024_001",
        "model": "transformer_v3",
        "config": {
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 6,
            "lr": 0.0001,
            "warmup_steps": 4000,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "batch_size": 64,
            "max_seq_len": 2048
        },
        "training_history": [
            {"epoch": 1, "train_loss": 2.5, "val_loss": 2.4, "acc": 0.15},
            {"epoch": 2, "train_loss": 1.8, "val_loss": 1.9, "acc": 0.35},
            {"epoch": 3, "train_loss": 1.2, "val_loss": 1.4, "acc": 0.55},
            {"epoch": 4, "train_loss": 0.8, "val_loss": 1.0, "acc": 0.72},
            {"epoch": 5, "train_loss": 0.5, "val_loss": 0.7, "acc": 0.85}
        ],
        "final_metrics": {
            "accuracy": 0.94,
            "f1": 0.93,
            "precision": 0.92,
            "recall": 0.95
        },
        "status": "completed"
    });

    // Pre-encode
    let mut encoder = Encoder::new();
    let fp_encoded = encoder.encode(&large).unwrap().to_vec();
    let json_encoded = serde_json::to_vec(&large).unwrap();

    println!("=== LARGE PAYLOAD DECODE BENCHMARK ===");
    println!("JSON size: {} bytes", json_encoded.len());
    println!("FluxPack size: {} bytes ({:.1}% smaller)", fp_encoded.len(),
        (1.0 - fp_encoded.len() as f64 / json_encoded.len() as f64) * 100.0);
    println!();

    // Warmup
    for _ in 0..1000 {
        let _: serde_json::Value = serde_json::from_slice(&json_encoded).unwrap();
        let _ = Decoder::new().decode(&fp_encoded).unwrap();
    }

    // JSON decode benchmark
    let iterations = 10000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _: serde_json::Value = serde_json::from_slice(&json_encoded).unwrap();
    }
    let json_time = start.elapsed();
    let json_ns = json_time.as_nanos() / iterations;

    // FluxPack standard decode benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let mut decoder = Decoder::new();
        let _ = decoder.decode(&fp_encoded).unwrap();
    }
    let fp_std_time = start.elapsed();
    let fp_std_ns = fp_std_time.as_nanos() / iterations;

    // FluxPack zero-copy decode benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = decode_zero_copy(&fp_encoded).unwrap();
    }
    let fp_zc_time = start.elapsed();
    let fp_zc_ns = fp_zc_time.as_nanos() / iterations;

    // FluxPack standard decode with reused decoder
    let mut decoder = Decoder::new();
    let start = Instant::now();
    for _ in 0..iterations {
        decoder.reset();
        let _ = decoder.decode(&fp_encoded).unwrap();
    }
    let fp_reuse_time = start.elapsed();
    let fp_reuse_ns = fp_reuse_time.as_nanos() / iterations;

    println!("=== RESULTS ({} iterations) ===", iterations);
    println!("JSON decode:         {:>8} ns", json_ns);
    println!("FluxPack standard:   {:>8} ns ({:.1}% vs JSON)", fp_std_ns,
        (1.0 - fp_std_ns as f64 / json_ns as f64) * 100.0);
    println!("FluxPack zero-copy:  {:>8} ns ({:.1}% vs JSON)", fp_zc_ns,
        (1.0 - fp_zc_ns as f64 / json_ns as f64) * 100.0);
    println!("FluxPack reuse dec:  {:>8} ns ({:.1}% vs JSON)", fp_reuse_ns,
        (1.0 - fp_reuse_ns as f64 / json_ns as f64) * 100.0);
}
