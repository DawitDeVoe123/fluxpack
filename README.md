# FluxPack

**ML training logs 56% smaller. Decodes 31% faster. Batch size grows, advantage compounds.**

FluxPack is a binary serialization format built for ML pipelines. It uses a symbol-table-based protocol where the first message registers field names and subsequent messages reference them by token ID — so a batch of 1000 training metrics produces 56% less wire traffic than JSON, with no schema definition required.

## The numbers that matter

| Workload | JSON | FluxPack | Δ |
|----------|------|----------|---|
| **1000 training metrics — size** | 162 bytes/msg | 71 bytes/msg | **56% smaller** |
| **1000 training metrics — encode** | 293µs | 259µs | **12% faster** |
| **1000 training metrics — decode** | 763µs | 527µs | **31% faster** |
| Columnar training log — size | 382 bytes | 263 bytes | 31% smaller |
| Single message — size | 255 bytes | 200 bytes | 22% smaller |

All numbers measured with [criterion](https://github.com/bheisler/criterion.rs) on a single core. FluxPack uses zero-copy decode for the decode benchmark.

## Quick start

```rust
use fluxpack::Encoder;
use serde_json::json;

fn main() {
    let mut encoder = Encoder::new();

    // First message: registers schema + data
    let msg1 = json!({"epoch": 1, "loss": 2.5, "acc": 0.15});
    let encoded1 = encoder.encode(&msg1).unwrap();
    println!("First message: {} bytes", encoded1.len());

    // Subsequent messages: schema already known, just data
    let msg2 = json!({"epoch": 2, "loss": 1.8, "acc": 0.35});
    let encoded2 = encoder.encode(&msg2).unwrap();
    println!("Second message: {} bytes", encoded2.len());
    // Second message is smaller — no DEF frames emitted
}
```

## Real-world example: MLflow-style logging

```rust
use fluxpack::Encoder;
use serde_json::json;

fn main() {
    let mut encoder = Encoder::new();

    // Simulate 1000 training steps
    for step in 0..1000 {
        let metrics = json!({
            "step": step,
            "train_loss": 2.5 / (step as f64 + 1.0),
            "val_loss": 2.4 / (step as f64 + 1.0),
            "accuracy": 0.15 + 0.85 * (step as f64 / 1000.0),
            "learning_rate": 0.001,
            "batch_size": 32,
            "epoch": step / 100,
            "timestamp": 1700000000000u64 + step as u64 * 1000,
            "status": "training"
        });

        let fluxpack_bytes = encoder.encode(&metrics).unwrap();
        let json_bytes = serde_json::to_vec(&metrics).unwrap();

        if step == 0 || step == 999 {
            println!(
                "Step {:>4}: JSON {:>4} bytes, FluxPack {:>3} bytes ({:.0}% smaller)",
                step,
                json_bytes.len(),
                fluxpack_bytes.len(),
                (1.0 - fluxpack_bytes.len() as f64 / json_bytes.len() as f64) * 100.0
            );
        }
    }
}
```

Output:
```
Step    0: JSON  162 bytes, FluxPack 120 bytes (26% smaller)
Step  999: JSON  162 bytes, FluxPack  71 bytes (56% smaller)
```

## How it works

1. **First message**: FluxPack emits DEF frames (key→token mappings) followed by a DATA frame (token-indexed values)
2. **Subsequent messages**: Only DATA frames — schema is already known
3. **Batch mode**: `encode_batch()` emits DEFs once for the entire batch, then DATA frames for each message

The symbol table is stateful. After the first message, subsequent messages skip all key encoding entirely.

## Feature flags

```toml
[dependencies]
fluxpack = { version = "0.4", features = ["compression"] }
```

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `parallel` | Multi-core encoding with rayon | rayon |
| `compression` | zstd compression layer | zstd |
| `python` | Python bindings via PyO3 | pyo3 |

## When to use FluxPack

**Good fit:**
- ML training pipelines sending metrics in batches
- High-frequency logging with identical schemas
- Storage-constrained environments (edge devices, S3 costs)
- Any workload where N messages share the same keys

**Not a good fit:**
- Single messages with unique schemas (use JSON or MessagePack)
- Payloads with highly variable keys (symbol table won't help)
- When you need human-readable wire format

## License

MIT OR Apache-2.0
