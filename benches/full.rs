use criterion::{criterion_group, criterion_main, Criterion};
use fluxpack::Encoder;
use serde_json::json;

fn bench_all_modes(c: &mut Criterion) {
    // === PAYLOAD DEFINITIONS ===

    let small = json!({
        "user_id": 8821,
        "email": "user@example.com",
        "active": true
    });

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

    // Small - JSON
    c.bench_function("json_encode_small", |b| {
        b.iter(|| {
            let bytes = serde_json::to_vec(&small).unwrap();
            criterion::black_box(bytes)
        })
    });

    // Small - FluxPack standard
    c.bench_function("fp_standard_encode_small", |b| {
        let mut encoder = Encoder::new();
        b.iter(|| {
            let encoded = encoder.encode(&small).unwrap();
            criterion::black_box(encoded.to_vec())
        })
    });

    // Small - FluxPack inline
    c.bench_function("fp_inline_encode_small", |b| {
        let mut encoder = Encoder::new();
        encoder.set_inline_mode(true);
        b.iter(|| {
            let encoded = encoder.encode(&small).unwrap();
            criterion::black_box(encoded.to_vec())
        })
    });

    // Medium - JSON
    c.bench_function("json_encode_medium", |b| {
        b.iter(|| {
            let bytes = serde_json::to_vec(&medium).unwrap();
            criterion::black_box(bytes)
        })
    });

    // Medium - FluxPack standard
    c.bench_function("fp_standard_encode_medium", |b| {
        let mut encoder = Encoder::new();
        b.iter(|| {
            let encoded = encoder.encode(&medium).unwrap();
            criterion::black_box(encoded.to_vec())
        })
    });

    // Medium - FluxPack inline
    c.bench_function("fp_inline_encode_medium", |b| {
        let mut encoder = Encoder::new();
        encoder.set_inline_mode(true);
        b.iter(|| {
            let encoded = encoder.encode(&medium).unwrap();
            criterion::black_box(encoded.to_vec())
        })
    });

    // Large - JSON
    c.bench_function("json_encode_large", |b| {
        b.iter(|| {
            let bytes = serde_json::to_vec(&large).unwrap();
            criterion::black_box(bytes)
        })
    });

    // Large - FluxPack standard
    c.bench_function("fp_standard_encode_large", |b| {
        let mut encoder = Encoder::new();
        b.iter(|| {
            let encoded = encoder.encode(&large).unwrap();
            criterion::black_box(encoded.to_vec())
        })
    });

    // === DECODE BENCHMARKS ===

    // Pre-encode all payloads
    let json_small = serde_json::to_vec(&small).unwrap();
    let json_medium = serde_json::to_vec(&medium).unwrap();
    let json_large = serde_json::to_vec(&large).unwrap();

    let mut enc = Encoder::new();
    let fp_std_small = enc.encode(&small).unwrap().to_vec();
    enc.reset();
    let fp_std_medium = enc.encode(&medium).unwrap().to_vec();
    enc.reset();
    let fp_std_large = enc.encode(&large).unwrap().to_vec();

    let mut enc_inline = Encoder::new();
    enc_inline.set_inline_mode(true);
    let fp_inline_small = enc_inline.encode(&small).unwrap().to_vec();
    enc_inline.reset();
    let fp_inline_medium = enc_inline.encode(&medium).unwrap().to_vec();

    // Decode - JSON
    c.bench_function("json_decode_small", |b| {
        let data = json_small.clone();
        b.iter(|| {
            let val: serde_json::Value = serde_json::from_slice(&data).unwrap();
            criterion::black_box(val)
        })
    });

    c.bench_function("json_decode_medium", |b| {
        let data = json_medium.clone();
        b.iter(|| {
            let val: serde_json::Value = serde_json::from_slice(&data).unwrap();
            criterion::black_box(val)
        })
    });

    c.bench_function("json_decode_large", |b| {
        let data = json_large.clone();
        b.iter(|| {
            let val: serde_json::Value = serde_json::from_slice(&data).unwrap();
            criterion::black_box(val)
        })
    });

    // Decode - FluxPack standard
    c.bench_function("fp_standard_decode_small", |b| {
        let mut decoder = fluxpack::Decoder::new();
        let data = fp_std_small.clone();
        b.iter(|| {
            let decoded = decoder.decode(&data).unwrap();
            criterion::black_box(decoded)
        })
    });

    c.bench_function("fp_standard_decode_medium", |b| {
        let mut decoder = fluxpack::Decoder::new();
        let data = fp_std_medium.clone();
        b.iter(|| {
            let decoded = decoder.decode(&data).unwrap();
            criterion::black_box(decoded)
        })
    });

    c.bench_function("fp_standard_decode_large", |b| {
        let mut decoder = fluxpack::Decoder::new();
        let data = fp_std_large.clone();
        b.iter(|| {
            let decoded = decoder.decode(&data).unwrap();
            criterion::black_box(decoded)
        })
    });

    // Decode - FluxPack inline
    c.bench_function("fp_inline_decode_small", |b| {
        let mut decoder = fluxpack::Decoder::new();
        let data = fp_inline_small.clone();
        b.iter(|| {
            let decoded = decoder.decode(&data).unwrap();
            criterion::black_box(decoded)
        })
    });

    c.bench_function("fp_inline_decode_medium", |b| {
        let mut decoder = fluxpack::Decoder::new();
        let data = fp_inline_medium.clone();
        b.iter(|| {
            let decoded = decoder.decode(&data).unwrap();
            criterion::black_box(decoded)
        })
    });

    // === CLEAN SIZE COMPARISON (absolute numbers only) ===
    println!("\n=== SIZE COMPARISON (absolute bytes) ===\n");

    println!("--- Small Payload ---");
    println!("  JSON:         {} bytes", json_small.len());
    println!("  FP standard:  {} bytes ({:.1}% vs JSON)", fp_std_small.len(),
        (1.0 - fp_std_small.len() as f64 / json_small.len() as f64) * 100.0);
    println!("  FP inline:    {} bytes ({:.1}% vs JSON)", fp_inline_small.len(),
        (1.0 - fp_inline_small.len() as f64 / json_small.len() as f64) * 100.0);

    println!("\n--- Medium Payload ---");
    println!("  JSON:         {} bytes", json_medium.len());
    println!("  FP standard:  {} bytes ({:.1}% vs JSON)", fp_std_medium.len(),
        (1.0 - fp_std_medium.len() as f64 / json_medium.len() as f64) * 100.0);
    println!("  FP inline:    {} bytes ({:.1}% vs JSON)", fp_inline_medium.len(),
        (1.0 - fp_inline_medium.len() as f64 / json_medium.len() as f64) * 100.0);

    println!("\n--- Large Payload ---");
    println!("  JSON:         {} bytes", json_large.len());
    println!("  FP standard:  {} bytes ({:.1}% vs JSON)", fp_std_large.len(),
        (1.0 - fp_std_large.len() as f64 / json_large.len() as f64) * 100.0);
}

criterion_group!(benches, bench_all_modes);
criterion_main!(benches);
