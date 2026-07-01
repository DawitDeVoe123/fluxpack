use criterion::{criterion_group, criterion_main, Criterion};
use fluxpack::{Encoder, Decoder, decode_zero_copy};
use serde_json::json;

fn bench_decode_large(c: &mut Criterion) {
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

    println!("\n=== LARGE PAYLOAD DECODE BENCHMARK ===");
    println!("JSON size: {} bytes", json_encoded.len());
    println!("FluxPack size: {} bytes ({:.1}% smaller)", fp_encoded.len(),
        (1.0 - fp_encoded.len() as f64 / json_encoded.len() as f64) * 100.0);

    // JSON decode
    c.bench_function("json_decode_large", |b| {
        let data = json_encoded.clone();
        b.iter(|| {
            let val: serde_json::Value = serde_json::from_slice(&data).unwrap();
            criterion::black_box(val)
        })
    });

    // FluxPack standard decode
    c.bench_function("fp_standard_decode_large", |b| {
        let mut decoder = Decoder::new();
        let data = fp_encoded.clone();
        b.iter(|| {
            let decoded = decoder.decode(&data).unwrap();
            criterion::black_box(decoded)
        })
    });

    // FluxPack zero-copy decode
    c.bench_function("fp_zerocopy_decode_large", |b| {
        let data = fp_encoded.clone();
        b.iter(|| {
            let decoded = decode_zero_copy(&data).unwrap();
            criterion::black_box(decoded)
        })
    });

    // FluxPack standard decode with fresh decoder each time
    c.bench_function("fp_standard_decode_large_fresh", |b| {
        let data = fp_encoded.clone();
        b.iter(|| {
            let mut decoder = Decoder::new();
            let decoded = decoder.decode(&data).unwrap();
            criterion::black_box(decoded)
        })
    });
}

criterion_group!(benches, bench_decode_large);
criterion_main!(benches);
