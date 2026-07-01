use criterion::{criterion_group, criterion_main, Criterion};
use fluxpack::{Encoder, decode_zero_copy};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize, Clone)]
struct TrainingConfig {
    n_estimators: u32,
    max_depth: u32,
    learning_rate: f64,
    batch_size: u32,
    epochs: u32,
}

#[derive(Serialize, Deserialize, Clone)]
struct Metrics {
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1_score: f64,
}

#[derive(Serialize, Deserialize, Clone)]
struct MLPayload {
    job_id: String,
    model_type: String,
    training_config: TrainingConfig,
    metrics: Metrics,
    status: String,
}

impl Default for MLPayload {
    fn default() -> Self {
        Self {
            job_id: "training_job_001".to_string(),
            model_type: "random_forest".to_string(),
            training_config: TrainingConfig {
                n_estimators: 100,
                max_depth: 10,
                learning_rate: 0.1,
                batch_size: 32,
                epochs: 100,
            },
            metrics: Metrics {
                accuracy: 0.95,
                precision: 0.93,
                recall: 0.97,
                f1_score: 0.95,
            },
            status: "running".to_string(),
        }
    }
}

fn bench_encode(c: &mut Criterion) {
    let payload = MLPayload::default();
    let json_value = json!({
        "job_id": "training_job_001",
        "model_type": "random_forest",
        "training_config": {
            "n_estimators": 100,
            "max_depth": 10,
            "learning_rate": 0.1,
            "batch_size": 32,
            "epochs": 100
        },
        "metrics": {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.97,
            "f1_score": 0.95
        },
        "status": "running"
    });

    let mut encoder = Encoder::new();

    // N=1 encode
    c.bench_function("json_encode_n1", |b| {
        b.iter(|| serde_json::to_vec(&json_value).unwrap())
    });

    c.bench_function("fluxpack_encode_n1", |b| {
        b.iter(|| encoder.encode(&json_value).unwrap().to_vec())
    });

    c.bench_function("msgpack_encode_n1", |b| {
        b.iter(|| rmp_serde::to_vec(&payload).unwrap())
    });

    // N=10 encode
    let batch_10: Vec<MLPayload> = (0..10).map(|_| payload.clone()).collect();
    let json_batch_10: Vec<serde_json::Value> = (0..10).map(|_| json_value.clone()).collect();

    c.bench_function("json_encode_n10", |b| {
        b.iter(|| {
            json_batch_10.iter().map(|v| serde_json::to_vec(v).unwrap().len()).sum::<usize>()
        })
    });

    c.bench_function("fluxpack_encode_n10", |b| {
        let mut enc = Encoder::new();
        b.iter(|| {
            let mut total = 0;
            for v in &json_batch_10 {
                total += enc.encode(v).unwrap().len();
            }
            total
        })
    });

    c.bench_function("msgpack_encode_n10", |b| {
        b.iter(|| {
            batch_10.iter().map(|p| rmp_serde::to_vec(p).unwrap().len()).sum::<usize>()
        })
    });

    // N=100 encode
    let batch_100: Vec<MLPayload> = (0..100).map(|_| payload.clone()).collect();
    let json_batch_100: Vec<serde_json::Value> = (0..100).map(|_| json_value.clone()).collect();

    c.bench_function("json_encode_n100", |b| {
        b.iter(|| {
            json_batch_100.iter().map(|v| serde_json::to_vec(v).unwrap().len()).sum::<usize>()
        })
    });

    c.bench_function("fluxpack_encode_n100", |b| {
        let mut enc = Encoder::new();
        b.iter(|| {
            let mut total = 0;
            for v in &json_batch_100 {
                total += enc.encode(v).unwrap().len();
            }
            total
        })
    });

    c.bench_function("msgpack_encode_n100", |b| {
        b.iter(|| {
            batch_100.iter().map(|p| rmp_serde::to_vec(p).unwrap().len()).sum::<usize>()
        })
    });

    // N=1000 encode
    let batch_1000: Vec<MLPayload> = (0..1000).map(|_| payload.clone()).collect();
    let json_batch_1000: Vec<serde_json::Value> = (0..1000).map(|_| json_value.clone()).collect();

    c.bench_function("json_encode_n1000", |b| {
        b.iter(|| {
            json_batch_1000.iter().map(|v| serde_json::to_vec(v).unwrap().len()).sum::<usize>()
        })
    });

    c.bench_function("fluxpack_encode_n1000", |b| {
        let mut enc = Encoder::new();
        b.iter(|| {
            let mut total = 0;
            for v in &json_batch_1000 {
                total += enc.encode(v).unwrap().len();
            }
            total
        })
    });

    c.bench_function("msgpack_encode_n1000", |b| {
        b.iter(|| {
            batch_1000.iter().map(|p| rmp_serde::to_vec(p).unwrap().len()).sum::<usize>()
        })
    });
}

fn bench_decode(c: &mut Criterion) {
    let payload = MLPayload::default();
    let json_value = json!({
        "job_id": "training_job_001",
        "model_type": "random_forest",
        "training_config": {
            "n_estimators": 100,
            "max_depth": 10,
            "learning_rate": 0.1,
            "batch_size": 32,
            "epochs": 100
        },
        "metrics": {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.97,
            "f1_score": 0.95
        },
        "status": "running"
    });

    // Pre-encode for decode benchmarks
    let json_bytes = serde_json::to_vec(&json_value).unwrap();
    let mut encoder = Encoder::new();
    let fp_bytes = encoder.encode(&json_value).unwrap().to_vec();
    let msgpack_bytes = rmp_serde::to_vec(&payload).unwrap();

    // N=1 decode
    c.bench_function("json_decode_n1", |b| {
        let data = json_bytes.clone();
        b.iter(|| {
            let _: serde_json::Value = serde_json::from_slice(&data).unwrap();
        })
    });

    c.bench_function("fluxpack_decode_n1", |b| {
        let data = fp_bytes.clone();
        b.iter(|| {
            let mut decoder = fluxpack::Decoder::new();
            decoder.decode(&data).unwrap();
        })
    });

    c.bench_function("fluxpack_zerocopy_decode_n1", |b| {
        let data = fp_bytes.clone();
        b.iter(|| {
            decode_zero_copy(&data).unwrap();
        })
    });

    c.bench_function("msgpack_decode_n1", |b| {
        let data = msgpack_bytes.clone();
        b.iter(|| {
            let _: MLPayload = rmp_serde::from_slice(&data).unwrap();
        })
    });

    // N=100 decode
    let batch_100: Vec<MLPayload> = (0..100).map(|_| payload.clone()).collect();
    let json_batch_100: Vec<serde_json::Value> = (0..100).map(|_| json_value.clone()).collect();

    let json_batch_bytes: Vec<Vec<u8>> = json_batch_100.iter().map(|v| serde_json::to_vec(v).unwrap()).collect();
    let mut enc = Encoder::new();
    let fp_batch_bytes: Vec<Vec<u8>> = json_batch_100.iter().map(|v| enc.encode(v).unwrap().to_vec()).collect();
    let msgpack_batch_bytes: Vec<Vec<u8>> = batch_100.iter().map(|p| rmp_serde::to_vec(p).unwrap()).collect();

    c.bench_function("json_decode_n100", |b| {
        let data = json_batch_bytes.clone();
        b.iter(|| {
            for d in &data {
                let _: serde_json::Value = serde_json::from_slice(d).unwrap();
            }
        })
    });

    c.bench_function("fluxpack_decode_n100", |b| {
        let data = fp_batch_bytes.clone();
        b.iter(|| {
            let mut decoder = fluxpack::Decoder::new();
            for d in &data {
                decoder.decode(d).unwrap();
            }
        })
    });

    c.bench_function("fluxpack_zerocopy_decode_n100", |b| {
        let data = fp_batch_bytes.clone();
        b.iter(|| {
            for d in &data {
                decode_zero_copy(d).unwrap();
            }
        })
    });

    c.bench_function("msgpack_decode_n100", |b| {
        let data = msgpack_batch_bytes.clone();
        b.iter(|| {
            for d in &data {
                let _: MLPayload = rmp_serde::from_slice(d).unwrap();
            }
        })
    });
}

fn bench_size(c: &mut Criterion) {
    let payload = MLPayload::default();
    let json_value = json!({
        "job_id": "training_job_001",
        "model_type": "random_forest",
        "training_config": {
            "n_estimators": 100,
            "max_depth": 10,
            "learning_rate": 0.1,
            "batch_size": 32,
            "epochs": 100
        },
        "metrics": {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.97,
            "f1_score": 0.95
        },
        "status": "running"
    });

    c.bench_function("json_size_n1", |b| {
        b.iter(|| serde_json::to_vec(&json_value).unwrap().len())
    });

    c.bench_function("fluxpack_size_n1", |b| {
        let mut encoder = Encoder::new();
        b.iter(|| encoder.encode(&json_value).unwrap().len())
    });

    c.bench_function("msgpack_size_n1", |b| {
        b.iter(|| rmp_serde::to_vec(&payload).unwrap().len())
    });

    // Size at N=100
    let batch_100: Vec<MLPayload> = (0..100).map(|_| payload.clone()).collect();
    let json_batch_100: Vec<serde_json::Value> = (0..100).map(|_| json_value.clone()).collect();

    c.bench_function("json_size_n100", |b| {
        b.iter(|| {
            json_batch_100.iter().map(|v| serde_json::to_vec(v).unwrap().len()).sum::<usize>()
        })
    });

    c.bench_function("fluxpack_size_n100", |b| {
        let mut enc = Encoder::new();
        b.iter(|| {
            let mut total = 0;
            for v in &json_batch_100 {
                total += enc.encode(v).unwrap().len();
            }
            total
        })
    });

    c.bench_function("msgpack_size_n100", |b| {
        b.iter(|| {
            batch_100.iter().map(|p| rmp_serde::to_vec(p).unwrap().len()).sum::<usize>()
        })
    });

    // Print size comparison table
    println!("\n=== FORMAT SIZE COMPARISON ===\n");

    let json_single = serde_json::to_vec(&json_value).unwrap();
    let mut enc = Encoder::new();
    let fp_single = enc.encode(&json_value).unwrap();
    let msgpack_single = rmp_serde::to_vec(&payload).unwrap();

    println!("--- Single Message (N=1) ---");
    println!("  JSON:       {} bytes", json_single.len());
    println!("  FluxPack:   {} bytes ({:.1}% vs JSON)", fp_single.len(),
        (1.0 - fp_single.len() as f64 / json_single.len() as f64) * 100.0);
    println!("  MessagePack: {} bytes ({:.1}% vs JSON)", msgpack_single.len(),
        (1.0 - msgpack_single.len() as f64 / json_single.len() as f64) * 100.0);

    // Batch N=100
    enc.reset();
    let json_batch: Vec<Vec<u8>> = json_batch_100.iter().map(|v| serde_json::to_vec(v).unwrap()).collect();
    let fp_batch: Vec<Vec<u8>> = {
        let mut e = Encoder::new();
        json_batch_100.iter().map(|v| e.encode(v).unwrap().to_vec()).collect()
    };
    let msgpack_batch: Vec<Vec<u8>> = batch_100.iter().map(|p| rmp_serde::to_vec(p).unwrap()).collect();

    let json_total: usize = json_batch.iter().map(|b| b.len()).sum();
    let fp_total: usize = fp_batch.iter().map(|b| b.len()).sum();
    let msgpack_total: usize = msgpack_batch.iter().map(|b| b.len()).sum();

    println!("\n--- Batch (N=100) ---");
    println!("  JSON:       {} bytes ({} bytes/msg avg)", json_total, json_total / 100);
    println!("  FluxPack:   {} bytes ({} bytes/msg avg, {:.1}% vs JSON)", fp_total, fp_total / 100,
        (1.0 - fp_total as f64 / json_total as f64) * 100.0);
    println!("  MessagePack: {} bytes ({} bytes/msg avg, {:.1}% vs JSON)", msgpack_total, msgpack_total / 100,
        (1.0 - msgpack_total as f64 / json_total as f64) * 100.0);
}

criterion_group!(benches, bench_encode, bench_decode, bench_size);
criterion_main!(benches);
