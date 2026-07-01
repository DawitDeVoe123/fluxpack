use fluxpack::Encoder;
use serde_json::json;

/// Dead-simple MLflow-style logging example.
///
/// Compare: how big is 1000 training metrics in JSON vs FluxPack?
fn main() {
    let mut encoder = Encoder::new();

    let mut json_total = 0usize;
    let mut fluxpack_total = 0usize;

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

        json_total += json_bytes.len();
        fluxpack_total += fluxpack_bytes.len();

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

    println!();
    println!("=== After 1000 steps ===");
    println!("JSON total:       {:>6} bytes", json_total);
    println!("FluxPack total:   {:>6} bytes", fluxpack_total);
    println!(
        "Savings:          {:.1}% smaller",
        (1.0 - fluxpack_total as f64 / json_total as f64) * 100.0
    );
    println!(
        "Per-message avg:  JSON {} bytes, FluxPack {} bytes",
        json_total / 1000,
        fluxpack_total / 1000
    );
}
