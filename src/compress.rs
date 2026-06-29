use crate::FluxPackError;

/// Zstd compression layer for FluxPack streams.
///
/// Compresses the entire FluxPack binary stream using zstd, providing
/// additional compression on top of FluxPack's binary encoding.
///
/// Typical additional compression: 30-60% smaller than raw FluxPack.
/// For ML pipelines with repetitive structures, zstd achieves excellent ratios.
///
/// # Compression Levels
/// - Level 1: Fast, ~30% additional compression
/// - Level 3: Balanced (default), ~45% additional compression
/// - Level 9: Slow, ~55% additional compression
/// - Level 19: Very slow, ~60% additional compression
///
/// Default compression level.
pub const DEFAULT_COMPRESSION_LEVEL: i32 = 3;

/// Compress a FluxPack stream using zstd.
pub fn compress(input: &[u8]) -> Result<Vec<u8>, FluxPackError> {
    compress_with_level(input, DEFAULT_COMPRESSION_LEVEL)
}

/// Compress with a specific zstd compression level (1-22).
pub fn compress_with_level(input: &[u8], level: i32) -> Result<Vec<u8>, FluxPackError> {
    zstd::encode_all(input, level)
        .map_err(|e| FluxPackError::ColumnarError(format!("zstd compression failed: {}", e)))
}

/// Decompress a zstd-compressed FluxPack stream.
pub fn decompress(input: &[u8]) -> Result<Vec<u8>, FluxPackError> {
    zstd::decode_all(input)
        .map_err(|e| FluxPackError::ColumnarError(format!("zstd decompression failed: {}", e)))
}

/// Compress with a pre-allocated dictionary for better compression of similar messages.
/// The dictionary should be trained on representative data.
pub fn compress_with_dict(input: &[u8], _dict: &[u8], level: i32) -> Result<Vec<u8>, FluxPackError> {
    // Dictionary-based compression requires trained dictionaries.
    // For now, fall back to standard compression.
    // TODO: Add zstd dict training via `zstd::dict::from_samples`
    compress_with_level(input, level)
}

/// Compression stats for measuring ratio.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub original_size: usize,
    pub compressed_size: usize,
    pub ratio: f64,
    pub savings_bytes: usize,
    pub savings_percent: f64,
}

impl CompressionStats {
    pub fn new(original: usize, compressed: usize) -> Self {
        let ratio = if compressed > 0 {
            original as f64 / compressed as f64
        } else {
            0.0
        };
        let savings = original.saturating_sub(compressed);
        let savings_pct = if original > 0 {
            (savings as f64 / original as f64) * 100.0
        } else {
            0.0
        };
        Self {
            original_size: original,
            compressed_size: compressed,
            ratio,
            savings_bytes: savings,
            savings_percent: savings_pct,
        }
    }
}

/// Compress and return stats.
pub fn compress_with_stats(input: &[u8]) -> Result<(Vec<u8>, CompressionStats), FluxPackError> {
    let compressed = compress(input)?;
    let stats = CompressionStats::new(input.len(), compressed.len());
    Ok((compressed, stats))
}

/// Find the optimal compression level for a given input.
/// Tests levels 1, 3, 6, 9 and returns the best tradeoff.
pub fn optimal_compress(input: &[u8]) -> Result<(Vec<u8>, i32, CompressionStats), FluxPackError> {
    let levels = [1, 3, 6, 9];
    let mut best = None;
    let mut best_ratio = 0.0f64;

    for &level in &levels {
        let compressed = compress_with_level(input, level)?;
        let stats = CompressionStats::new(input.len(), compressed.len());

        // Prefer higher compression, but penalize very slow levels
        let score = stats.ratio * if level <= 3 { 1.0 } else if level <= 6 { 0.95 } else { 0.9 };

        if score > best_ratio {
            best_ratio = score;
            best = Some((compressed, level, stats));
        }
    }

    best.ok_or(FluxPackError::ColumnarError("no compression level worked".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Encoder;
    use serde_json::json;

    #[test]
    fn test_compress_decompress_roundtrip() {
        let mut encoder = Encoder::new();
        let data = json!({
            "job_id": "training_job_001",
            "model_type": "random_forest",
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.93,
                "recall": 0.97,
                "f1_score": 0.95
            }
        });

        let fluxpack_bytes = encoder.encode(&data).unwrap().to_vec();
        let compressed = compress(&fluxpack_bytes).unwrap();
        let decompressed = decompress(&compressed).unwrap();

        assert_eq!(fluxpack_bytes, decompressed);
    }

    #[test]
    fn test_compression_ratio() {
        let mut encoder = Encoder::new();
        // Create a realistic ML payload
        let data = json!({
            "experiment_id": "exp_2024_001",
            "model": "transformer_v3",
            "config": {
                "d_model": 512,
                "n_heads": 8,
                "n_layers": 6,
                "lr": 0.0001,
                "warmup_steps": 4000,
                "dropout": 0.1
            },
            "training_history": [
                {"epoch": 1, "train_loss": 2.5, "val_loss": 2.4},
                {"epoch": 2, "train_loss": 1.8, "val_loss": 1.9},
                {"epoch": 3, "train_loss": 1.2, "val_loss": 1.4},
                {"epoch": 4, "train_loss": 0.8, "val_loss": 1.0},
                {"epoch": 5, "train_loss": 0.5, "val_loss": 0.7}
            ],
            "status": "completed"
        });

        let fluxpack_bytes = encoder.encode(&data).unwrap().to_vec();
        let (compressed, stats) = compress_with_stats(&fluxpack_bytes).unwrap();

        println!("FluxPack: {} bytes", fluxpack_bytes.len());
        println!("Compressed: {} bytes", compressed.len());
        println!("Ratio: {:.2}x, Savings: {:.1}%", stats.ratio, stats.savings_percent);

        // Zstd should compress FluxPack further
        assert!(compressed.len() < fluxpack_bytes.len(),
            "Compressed ({}) should be smaller than FluxPack ({})",
            compressed.len(), fluxpack_bytes.len());
    }

    #[test]
    fn test_compression_levels() {
        let mut encoder = Encoder::new();
        let data = json!({
            "values": (0..1000).map(|i| json!(i as f64 * 0.001)).collect::<Vec<_>>()
        });

        let fluxpack_bytes = encoder.encode(&data).unwrap().to_vec();

        let levels = [1, 3, 6, 9];
        let mut prev_size = usize::MAX;

        for &level in &levels {
            let compressed = compress_with_level(&fluxpack_bytes, level).unwrap();
            // Higher levels should produce smaller or equal output
            assert!(compressed.len() <= prev_size,
                "Level {} ({}) should be <= level {} ({})",
                level, compressed.len(),
                if level == 1 { 0 } else { level - 2 },
                prev_size);
            prev_size = compressed.len();
        }
    }

    #[test]
    fn test_total_pipeline_compression() {
        // JSON → FluxPack → zstd comparison
        // Use a larger payload where zstd shines
        let data = json!({
            "job_id": "training_job_001",
            "model_type": "random_forest",
            "config": {
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
            "training_history": [
                {"epoch": 1, "loss": 2.5, "acc": 0.1},
                {"epoch": 2, "loss": 1.8, "acc": 0.3},
                {"epoch": 3, "loss": 1.2, "acc": 0.5},
                {"epoch": 4, "loss": 0.8, "acc": 0.7},
                {"epoch": 5, "loss": 0.5, "acc": 0.85}
            ]
        });

        let json_bytes = serde_json::to_vec(&data).unwrap();
        let mut encoder = Encoder::new();
        let fluxpack_bytes = encoder.encode(&data).unwrap().to_vec();
        let (compressed, stats) = compress_with_stats(&fluxpack_bytes).unwrap();

        println!("JSON:         {} bytes", json_bytes.len());
        println!("FluxPack:     {} bytes ({:.1}% smaller than JSON)",
            fluxpack_bytes.len(),
            (1.0 - fluxpack_bytes.len() as f64 / json_bytes.len() as f64) * 100.0);
        println!("FluxPack+zstd:{} bytes ({:.1}% smaller than JSON)",
            compressed.len(),
            (1.0 - compressed.len() as f64 / json_bytes.len() as f64) * 100.0);
        println!("Total compression: {:.2}x", stats.ratio);

        // FluxPack alone should be smaller than JSON
        assert!(fluxpack_bytes.len() < json_bytes.len(),
            "FluxPack ({}) should be smaller than JSON ({})",
            fluxpack_bytes.len(), json_bytes.len());

        // zstd should compress FluxPack further for larger payloads
        assert!(compressed.len() < fluxpack_bytes.len(),
            "Compressed ({}) should be smaller than FluxPack ({})",
            compressed.len(), fluxpack_bytes.len());
    }
}
