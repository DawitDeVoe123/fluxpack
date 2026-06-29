use criterion::{criterion_group, criterion_main, Criterion};
use fluxpack::{Encoder, Decoder};
use serde_json::json;

fn bench_fluxpack_vs_json(c: &mut Criterion) {
    // Small payload: typical ML inference request
    let small = json!({
        "user_id": 8821,
        "email": "user@example.com",
        "active": true
    });

    // Medium payload: training configuration
    let medium = json!({
        "job_id": "training_job_001",
        "model_type": "random_forest",
        "training_config": {
            "n_estimators": 100,
            "max_depth": 10,
            "learning_rate": 0.1,
            "batch_size": 32,
            "epochs": 100,
            "validation_split": 0.2,
            "random_state": 42
        },
        "metrics": {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.97,
            "f1_score": 0.95
        },
        "environment": "production",
        "status": "running"
    });

    // Large payload: training metrics history
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

    // === ENCODE BENCHMARKS ===

    // Small encode
    c.bench_function("fluxpack_encode_small", |b| {
        let mut encoder = Encoder::new();
        b.iter(|| {
            let encoded = encoder.encode(&small).unwrap();
            criterion::black_box(encoded.to_vec())
        })
    });

    c.bench_function("json_encode_small", |b| {
        b.iter(|| {
            let bytes = serde_json::to_vec(&small).unwrap();
            criterion::black_box(bytes)
        })
    });

    // Medium encode
    c.bench_function("fluxpack_encode_medium", |b| {
        let mut encoder = Encoder::new();
        b.iter(|| {
            let encoded = encoder.encode(&medium).unwrap();
            criterion::black_box(encoded.to_vec())
        })
    });

    c.bench_function("json_encode_medium", |b| {
        b.iter(|| {
            let bytes = serde_json::to_vec(&medium).unwrap();
            criterion::black_box(bytes)
        })
    });

    // Large encode
    c.bench_function("fluxpack_encode_large", |b| {
        let mut encoder = Encoder::new();
        b.iter(|| {
            let encoded = encoder.encode(&large).unwrap();
            criterion::black_box(encoded.to_vec())
        })
    });

    c.bench_function("json_encode_large", |b| {
        b.iter(|| {
            let bytes = serde_json::to_vec(&large).unwrap();
            criterion::black_box(bytes)
        })
    });

    // === DECODE BENCHMARKS ===

    let fluxpack_small = {
        let mut enc = Encoder::new();
        enc.encode(&small).unwrap().to_vec()
    };
    let fluxpack_medium = {
        let mut enc = Encoder::new();
        enc.encode(&medium).unwrap().to_vec()
    };
    let fluxpack_large = {
        let mut enc = Encoder::new();
        enc.encode(&large).unwrap().to_vec()
    };
    let json_small = serde_json::to_vec(&small).unwrap();
    let json_medium = serde_json::to_vec(&medium).unwrap();
    let json_large = serde_json::to_vec(&large).unwrap();

    c.bench_function("fluxpack_decode_small", |b| {
        let mut decoder = Decoder::new();
        let data = fluxpack_small.clone();
        b.iter(|| {
            let decoded = decoder.decode(&data).unwrap();
            criterion::black_box(decoded)
        })
    });

    c.bench_function("json_decode_small", |b| {
        let data = json_small.clone();
        b.iter(|| {
            let val: serde_json::Value = serde_json::from_slice(&data).unwrap();
            criterion::black_box(val)
        })
    });

    c.bench_function("fluxpack_decode_medium", |b| {
        let mut decoder = Decoder::new();
        let data = fluxpack_medium.clone();
        b.iter(|| {
            let decoded = decoder.decode(&data).unwrap();
            criterion::black_box(decoded)
        })
    });

    c.bench_function("json_decode_medium", |b| {
        let data = json_medium.clone();
        b.iter(|| {
            let val: serde_json::Value = serde_json::from_slice(&data).unwrap();
            criterion::black_box(val)
        })
    });

    c.bench_function("fluxpack_decode_large", |b| {
        let mut decoder = Decoder::new();
        let data = fluxpack_large.clone();
        b.iter(|| {
            let decoded = decoder.decode(&data).unwrap();
            criterion::black_box(decoded)
        })
    });

    c.bench_function("json_decode_large", |b| {
        let data = json_large.clone();
        b.iter(|| {
            let val: serde_json::Value = serde_json::from_slice(&data).unwrap();
            criterion::black_box(val)
        })
    });

    // === SIZE COMPARISON ===
    println!("\n=== Size Comparison ===");
    println!("--- Small Payload ---");
    println!("  JSON:      {} bytes", json_small.len());
    println!("  FluxPack:  {} bytes", fluxpack_small.len());
    println!("  Reduction: {:.1}%", (1.0 - fluxpack_small.len() as f64 / json_small.len() as f64) * 100.0);

    println!("--- Medium Payload ---");
    println!("  JSON:      {} bytes", json_medium.len());
    println!("  FluxPack:  {} bytes", fluxpack_medium.len());
    println!("  Reduction: {:.1}%", (1.0 - fluxpack_medium.len() as f64 / json_medium.len() as f64) * 100.0);

    println!("--- Large Payload ---");
    println!("  JSON:      {} bytes", json_large.len());
    println!("  FluxPack:  {} bytes", fluxpack_large.len());
    println!("  Reduction: {:.1}%", (1.0 - fluxpack_large.len() as f64 / json_large.len() as f64) * 100.0);
}

criterion_group!(benches, bench_fluxpack_vs_json);
criterion_main!(benches);
