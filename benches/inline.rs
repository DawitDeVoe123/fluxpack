use criterion::{criterion_group, criterion_main, Criterion};
use fluxpack::Encoder;
use serde_json::json;

fn bench_inline_vs_json(c: &mut Criterion) {
    let small = json!({
        "user_id": 8821,
        "email": "user@example.com",
        "active": true
    });

    let medium = json!({
        "job_id": "training_job_001",
        "model_type": "random_forest",
        "config": {
            "n_estimators": 100,
            "max_depth": 10,
            "learning_rate": 0.1
        },
        "metrics": {
            "accuracy": 0.95,
            "precision": 0.93
        }
    });

    // === ENCODE BENCHMARKS ===

    // Small: inline mode
    c.bench_function("fluxpack_inline_encode_small", |b| {
        let mut encoder = Encoder::new();
        encoder.set_inline_mode(true);
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

    // Medium: inline mode (if fits)
    c.bench_function("fluxpack_inline_encode_medium", |b| {
        let mut encoder = Encoder::new();
        encoder.set_inline_mode(true);
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

    // Standard mode for comparison
    c.bench_function("fluxpack_standard_encode_small", |b| {
        let mut encoder = Encoder::new();
        b.iter(|| {
            let encoded = encoder.encode(&small).unwrap();
            criterion::black_box(encoded.to_vec())
        })
    });

    // === SIZE COMPARISON ===

    // Small payload
    let mut enc_inline = Encoder::new();
    enc_inline.set_inline_mode(true);
    let fp_inline_small = enc_inline.encode(&small).unwrap().to_vec();

    let mut enc_std = Encoder::new();
    let fp_std_small = enc_std.encode(&small).unwrap().to_vec();

    let json_small = serde_json::to_vec(&small).unwrap();

    println!("\n=== Small Payload Size ===");
    println!("  JSON:         {} bytes", json_small.len());
    println!("  FluxPack std: {} bytes", fp_std_small.len());
    println!("  FluxPack inl: {} bytes", fp_inline_small.len());
    println!("  Inline vs JSON: {:.1}%",
        (1.0 - fp_inline_small.len() as f64 / json_small.len() as f64) * 100.0);

    // Medium payload
    let mut enc_inline = Encoder::new();
    enc_inline.set_inline_mode(true);
    let fp_inline_med = enc_inline.encode(&medium).unwrap().to_vec();

    let mut enc_std = Encoder::new();
    let fp_std_med = enc_std.encode(&medium).unwrap().to_vec();

    let json_med = serde_json::to_vec(&medium).unwrap();

    println!("\n=== Medium Payload Size ===");
    println!("  JSON:         {} bytes", json_med.len());
    println!("  FluxPack std: {} bytes", fp_std_med.len());
    println!("  FluxPack inl: {} bytes", fp_inline_med.len());
    println!("  Inline vs JSON: {:.1}%",
        (1.0 - fp_inline_med.len() as f64 / json_med.len() as f64) * 100.0);
}

criterion_group!(benches, bench_inline_vs_json);
criterion_main!(benches);
