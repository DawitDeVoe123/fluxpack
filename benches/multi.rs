use criterion::{criterion_group, criterion_main, Criterion};
use fluxpack::{Encoder, StreamWriter, StreamReader};
use serde_json::json;

fn bench_multi_message(c: &mut Criterion) {
    let data = json!({
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

    // === BATCH ENCODING: N messages per session ===
    let scenarios = vec![("N=1", 1usize), ("N=10", 10), ("N=100", 100)];

    for (name, n) in &scenarios {
        // FluxPack: encoder reuses symbol table across messages
        c.bench_function(&format!("fluxpack_{}", name), |b| {
            let mut encoder = Encoder::new();
            b.iter(|| {
                let mut total_size = 0;
                for _ in 0..*n {
                    let encoded = encoder.encode(&data).unwrap();
                    total_size += encoded.len();
                }
                criterion::black_box(total_size)
            })
        });

        // JSON: no state reuse
        c.bench_function(&format!("json_{}", name), |b| {
            b.iter(|| {
                let mut total_size = 0;
                for _ in 0..*n {
                    let bytes = serde_json::to_vec(&data).unwrap();
                    total_size += bytes.len();
                }
                criterion::black_box(total_size)
            })
        });
    }

    // === BATCH API: encode_batch vs repeated encode ===
    let batch_data: Vec<serde_json::Value> = (0..100)
        .map(|i| {
            json!({
                "job_id": format!("job_{}", i),
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
            })
        })
        .collect();

    c.bench_function("fluxpack_batch_encode_100", |b| {
        let mut encoder = Encoder::new();
        let data_clone = batch_data.clone();
        b.iter(|| {
            let encoded = encoder.encode_batch(&data_clone).unwrap();
            criterion::black_box(encoded.to_vec())
        })
    });

    c.bench_function("fluxpack_repeated_encode_100", |b| {
        let mut encoder = Encoder::new();
        let data_clone = batch_data.clone();
        b.iter(|| {
            let mut total_size = 0;
            for msg in &data_clone {
                let encoded = encoder.encode(msg).unwrap();
                total_size += encoded.len();
            }
            criterion::black_box(total_size)
        })
    });

    // === STREAMING API ===
    c.bench_function("fluxpack_stream_write_100", |b| {
        b.iter(|| {
            let mut writer = StreamWriter::new();
            for msg in &batch_data {
                writer.write_data(msg).unwrap();
            }
            criterion::black_box(writer.into_buffer())
        })
    });

    c.bench_function("fluxpack_stream_read_100", |b| {
        let mut writer = StreamWriter::new();
        for msg in &batch_data {
            writer.write_data(msg).unwrap();
        }
        let bytes = writer.into_buffer();

        b.iter(|| {
            let mut reader = StreamReader::new();
            let frames = reader.read_all(&bytes).unwrap();
            criterion::black_box(frames)
        })
    });

    // === SYMBOL TABLE REUSE ===
    // Measure the per-message cost after the first message
    c.bench_function("fluxpack_subsequent_message", |b| {
        let mut encoder = Encoder::new();
        // First message builds the symbol table
        let _ = encoder.encode(&data).unwrap();
        
        b.iter(|| {
            let encoded = encoder.encode(&data).unwrap();
            criterion::black_box(encoded.len())
        })
    });

    // === COLUMNAR ENCODING ===
    let tabular_data = json!({
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

    c.bench_function("fluxpack_encode_columnar", |b| {
        let mut encoder = Encoder::new();
        let data_clone = tabular_data.clone();
        b.iter(|| {
            let encoded = encoder.encode_with_columnar(&data_clone).unwrap();
            criterion::black_box(encoded.to_vec())
        })
    });

    c.bench_function("fluxpack_encode_standard", |b| {
        let mut encoder = Encoder::new();
        let data_clone = tabular_data.clone();
        b.iter(|| {
            let encoded = encoder.encode(&data_clone).unwrap();
            criterion::black_box(encoded.to_vec())
        })
    });

    c.bench_function("json_encode_tabular", |b| {
        let data_clone = tabular_data.clone();
        b.iter(|| {
            let bytes = serde_json::to_vec(&data_clone).unwrap();
            criterion::black_box(bytes)
        })
    });

    // === SIZE COMPARISONS ===
    println!("\n=== Size Comparison ===");

    // Single message
    let mut enc = Encoder::new();
    let fp_single = enc.encode(&data).unwrap().len();
    let json_single = serde_json::to_vec(&data).unwrap().len();
    println!("Single message:");
    println!("  JSON:      {} bytes", json_single);
    println!("  FluxPack:  {} bytes", fp_single);
    println!("  Reduction: {:.1}%", (1.0 - fp_single as f64 / json_single as f64) * 100.0);

    // Batch of 100
    enc.reset();
    let fp_batch = enc.encode_batch(&batch_data).unwrap().len();
    let json_batch: usize = batch_data.iter().map(|m| serde_json::to_vec(m).unwrap().len()).sum();
    println!("\nBatch of 100 messages:");
    println!("  JSON total:     {} bytes", json_batch);
    println!("  FluxPack total: {} bytes", fp_batch);
    println!("  Reduction:      {:.1}%", (1.0 - fp_batch as f64 / json_batch as f64) * 100.0);
    println!("  FluxPack per-msg avg: {} bytes", fp_batch / 100);

    // Columnar vs standard
    let mut enc = Encoder::new();
    let fp_col = enc.encode_with_columnar(&tabular_data).unwrap().len();
    enc.reset();
    let fp_std = enc.encode(&tabular_data).unwrap().len();
    let json_tab = serde_json::to_vec(&tabular_data).unwrap().len();
    println!("\nTabular data (8 objects):");
    println!("  JSON:          {} bytes", json_tab);
    println!("  FluxPack std:  {} bytes", fp_std);
    println!("  FluxPack col:  {} bytes", fp_col);
    println!("  Columnar vs JSON:      {:.1}%", (1.0 - fp_col as f64 / json_tab as f64) * 100.0);
    println!("  Columnar vs standard:  {:.1}%", (1.0 - fp_col as f64 / fp_std as f64) * 100.0);
}

criterion_group!(benches, bench_multi_message);
criterion_main!(benches);
