use criterion::{criterion_group, criterion_main, Criterion};
use fluxpack::{Encoder, Decoder, FastDecoder, decode_zero_copy};
use serde_json::json;

fn bench_ml_workloads(c: &mut Criterion) {
    // === WORKLOAD 1: Batch of 1000 identical-schema training metrics ===
    let training_metrics = json!({
        "epoch": 1,
        "train_loss": 2.4521,
        "val_loss": 2.3891,
        "accuracy": 0.1523,
        "learning_rate": 0.001,
        "batch_size": 32,
        "step": 100,
        "timestamp": 1700000000000u64,
        "status": "training"
    });

    println!("\n=== ML WORKLOAD BENCHMARKS ===\n");

    // --- Encode 1000 messages ---
    c.bench_function("ml_json_encode_1000", |b| {
        b.iter(|| {
            let mut total = 0;
            for _ in 0..1000 {
                total += serde_json::to_vec(&training_metrics).unwrap().len();
            }
            total
        })
    });

    c.bench_function("ml_fluxpack_encode_1000", |b| {
        let mut encoder = Encoder::new();
        b.iter(|| {
            let mut total = 0;
            for _ in 0..1000 {
                total += encoder.encode(&training_metrics).unwrap().len();
            }
            total
        })
    });

    // --- Decode 1000 messages ---
    let json_bytes: Vec<Vec<u8>> = (0..1000)
        .map(|_| serde_json::to_vec(&training_metrics).unwrap())
        .collect();

    let mut enc = Encoder::new();
    let fp_bytes: Vec<Vec<u8>> = (0..1000)
        .map(|_| enc.encode(&training_metrics).unwrap().to_vec())
        .collect();

    c.bench_function("ml_json_decode_1000", |b| {
        let data = json_bytes.clone();
        b.iter(|| {
            for d in &data {
                let _: serde_json::Value = serde_json::from_slice(d).unwrap();
            }
        })
    });

    c.bench_function("ml_fluxpack_decode_1000", |b| {
        let data = fp_bytes.clone();
        b.iter(|| {
            let mut decoder = Decoder::new();
            for d in &data {
                decoder.decode(d).unwrap();
            }
        })
    });

    c.bench_function("ml_fluxpack_fast_decode_1000", |b| {
        let data = fp_bytes.clone();
        b.iter(|| {
            let mut decoder = FastDecoder::new();
            for d in &data {
                decoder.decode(d).unwrap();
            }
        })
    });

    c.bench_function("ml_fluxpack_zerocopy_decode_1000", |b| {
        let data = fp_bytes.clone();
        b.iter(|| {
            for d in &data {
                decode_zero_copy(d).unwrap();
            }
        })
    });

    // --- Size comparison for batch ---
    println!("--- Batch of 1000 Training Metrics ---");
    let json_total: usize = json_bytes.iter().map(|b| b.len()).sum();
    let fp_total: usize = fp_bytes.iter().map(|b| b.len()).sum();
    println!("  JSON:       {} bytes ({} bytes/msg)", json_total, json_total / 1000);
    println!("  FluxPack:   {} bytes ({} bytes/msg, {:.1}% smaller)",
        fp_total, fp_total / 1000,
        (1.0 - fp_total as f64 / json_total as f64) * 100.0);

    // === WORKLOAD 2: Columnar training log (8 epochs) ===
    let training_log = json!({
        "training_log": [
            {"epoch": 1, "lr": 0.001, "loss": 2.5, "acc": 0.15},
            {"epoch": 2, "lr": 0.001, "loss": 1.8, "acc": 0.35},
            {"epoch": 3, "lr": 0.001, "loss": 1.2, "acc": 0.55},
            {"epoch": 4, "lr": 0.0005, "loss": 0.8, "acc": 0.72},
            {"epoch": 5, "lr": 0.0005, "loss": 0.5, "acc": 0.85},
            {"epoch": 6, "lr": 0.0005, "loss": 0.3, "acc": 0.90},
            {"epoch": 7, "lr": 0.0001, "loss": 0.2, "acc": 0.93},
            {"epoch": 8, "lr": 0.0001, "loss": 0.1, "acc": 0.95},
        ]
    });

    println!("\n--- Columnar Training Log (8 epochs) ---");

    let json_log_bytes = serde_json::to_vec(&training_log).unwrap();
    let mut encoder = Encoder::new();
    let fp_log_bytes = encoder.encode_with_columnar(&training_log).unwrap().to_vec();

    println!("  JSON:       {} bytes", json_log_bytes.len());
    println!("  FluxPack:   {} bytes ({:.1}% smaller)",
        fp_log_bytes.len(),
        (1.0 - fp_log_bytes.len() as f64 / json_log_bytes.len() as f64) * 100.0);

    c.bench_function("ml_json_encode_columnar", |b| {
        b.iter(|| serde_json::to_vec(&training_log).unwrap())
    });

    c.bench_function("ml_fluxpack_encode_columnar", |b| {
        let mut encoder = Encoder::new();
        b.iter(|| encoder.encode_with_columnar(&training_log).unwrap().to_vec())
    });

    c.bench_function("ml_json_decode_columnar", |b| {
        let data = json_log_bytes.clone();
        b.iter(|| {
            let _: serde_json::Value = serde_json::from_slice(&data).unwrap();
        })
    });

    c.bench_function("ml_fluxpack_decode_columnar", |b| {
        let data = fp_log_bytes.clone();
        b.iter(|| {
            let mut decoder = Decoder::new();
            decoder.decode(&data).unwrap();
        })
    });

    // === WORKLOAD 3: Tensor data (model embeddings) ===
    let tensor_data = json!({
        "embeddings": vec![0.1f64; 128], // 128-dimensional embedding
        "layer": "encoder",
        "head": 0
    });

    println!("\n--- Tensor Data (128-dim embedding) ---");

    let json_tensor_bytes = serde_json::to_vec(&tensor_data).unwrap();
    let mut encoder = Encoder::new();
    let fp_tensor_bytes = encoder.encode(&tensor_data).unwrap().to_vec();

    println!("  JSON:       {} bytes", json_tensor_bytes.len());
    println!("  FluxPack:   {} bytes ({:.1}% smaller)",
        fp_tensor_bytes.len(),
        (1.0 - fp_tensor_bytes.len() as f64 / json_tensor_bytes.len() as f64) * 100.0);

    // === WORKLOAD 4: Large config with nested objects ===
    let large_config = json!({
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

    println!("\n--- Large Config (600B) ---");

    let json_large_bytes = serde_json::to_vec(&large_config).unwrap();
    let mut encoder = Encoder::new();
    let fp_large_bytes = encoder.encode(&large_config).unwrap().to_vec();

    println!("  JSON:       {} bytes", json_large_bytes.len());
    println!("  FluxPack:   {} bytes ({:.1}% smaller)",
        fp_large_bytes.len(),
        (1.0 - fp_large_bytes.len() as f64 / json_large_bytes.len() as f64) * 100.0);

    c.bench_function("ml_json_decode_large", |b| {
        let data = json_large_bytes.clone();
        b.iter(|| {
            let _: serde_json::Value = serde_json::from_slice(&data).unwrap();
        })
    });

    c.bench_function("ml_fluxpack_decode_large", |b| {
        let data = fp_large_bytes.clone();
        b.iter(|| {
            let mut decoder = Decoder::new();
            decoder.decode(&data).unwrap();
        })
    });

    c.bench_function("ml_fluxpack_fast_decode_large", |b| {
        let data = fp_large_bytes.clone();
        b.iter(|| {
            let mut decoder = FastDecoder::new();
            decoder.decode(&data).unwrap();
        })
    });

    c.bench_function("ml_fluxpack_zerocopy_decode_large", |b| {
        let data = fp_large_bytes.clone();
        b.iter(|| {
            decode_zero_copy(&data).unwrap();
        })
    });
}

criterion_group!(benches, bench_ml_workloads);
criterion_main!(benches);
