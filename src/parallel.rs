use serde_json::Value;
use crate::{Encoder, Decoder, FluxPackError, SymbolTable, encode_varint};

/// Parallel encoding and decoding using rayon.
///
/// Strategy: build the complete symbol table first, then encode chunks in parallel.
/// Each parallel encoder gets the full symbol table so tokens are consistent.
///
/// Encode multiple independent messages in parallel.
///
/// 1. Pre-scan all messages to build complete symbol table
/// 2. Emit all DEF frames once
/// 3. Encode each message's DATA frame in parallel
/// 4. Concatenate: DEFs + all DATA frames
pub fn encode_batch_parallel(messages: &[Value]) -> Result<Vec<u8>, FluxPackError> {
    use rayon::prelude::*;

    if messages.is_empty() {
        return Ok(Vec::new());
    }

    // Phase 1: Build complete symbol table (sequential)
    let mut shared_table = SymbolTable::with_predefined();
    for msg in messages {
        if let Some(obj) = msg.as_object() {
            for (key, value) in obj {
                shared_table.intern(key)?;
                intern_nested(&mut shared_table, value)?;
            }
        }
    }

    // Phase 2: Encode all messages in parallel using the shared table
    let encoded_parts: Result<Vec<Vec<u8>>, FluxPackError> = messages
        .par_iter()
        .map(|msg| {
            let mut encoder = Encoder::new();
            encoder.clone_table_from(&shared_table);
            encoder.encode_data_only(msg).map(|b| b.to_vec())
        })
        .collect();

    let parts = encoded_parts?;

    // Phase 3: Build output = DEFs + DATA frames only
    let mut result = Vec::new();
    for (token, key) in shared_table.iter() {
        result.push(0x01u8); // DEF frame
        encode_varint(token as u64, &mut result);
        encode_varint(key.len() as u64, &mut result);
        result.extend_from_slice(key.as_bytes());
    }

    // Append DATA frames directly (encode_data_only produces only DATA frames)
    for part in &parts {
        result.extend_from_slice(part);
    }

    Ok(result)
}

/// Decode multiple messages from a parallel-encoded stream.
pub fn decode_batch_parallel(input: &[u8]) -> Result<Vec<Value>, FluxPackError> {
    let mut decoder = Decoder::new();
    decoder.decode_all(input)
}

/// Pre-scan messages to build a symbol table without encoding.
pub fn prebuild_symbol_table(messages: &[Value]) -> Result<SymbolTable, FluxPackError> {
    let mut table = SymbolTable::with_predefined();
    for msg in messages {
        if let Some(obj) = msg.as_object() {
            for (key, value) in obj {
                table.intern(key)?;
                intern_nested(&mut table, value)?;
            }
        }
    }
    Ok(table)
}

fn intern_nested(table: &mut SymbolTable, value: &Value) -> Result<(), FluxPackError> {
    match value {
        Value::Object(obj) => {
            for (key, val) in obj {
                table.intern(key)?;
                intern_nested(table, val)?;
            }
        }
        Value::Array(arr) => {
            for item in arr {
                intern_nested(table, item)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parallel_batch_roundtrip() {
        let messages: Vec<Value> = (0..20)
            .map(|i| {
                json!({
                    "id": i,
                    "name": format!("item_{}", i),
                    "value": i as f64 * 1.5,
                    "active": i % 2 == 0
                })
            })
            .collect();

        let encoded = encode_batch_parallel(&messages).unwrap();
        let decoded = decode_batch_parallel(&encoded).unwrap();

        assert_eq!(decoded.len(), 20);
        for (i, msg) in decoded.iter().enumerate() {
            let obj = msg.as_object().unwrap();
            assert_eq!(obj["id"], json!(i));
        }
    }

    #[test]
    fn test_prebuild_symbol_table() {
        let messages = vec![
            json!({"a": 1, "b": 2}),
            json!({"a": 3, "b": 4, "c": 5}),
        ];

        let table = prebuild_symbol_table(&messages).unwrap();
        assert!(table.contains_key("a"));
        assert!(table.contains_key("b"));
        assert!(table.contains_key("c"));
    }

    #[test]
    fn test_parallel_output_matches_sequential() {
        let messages: Vec<Value> = (0..10)
            .map(|i| {
                json!({
                    "epoch": i,
                    "loss": 2.5 - (i as f64 * 0.05),
                    "accuracy": 0.5 + (i as f64 * 0.01)
                })
            })
            .collect();

        let parallel = encode_batch_parallel(&messages).unwrap();
        let mut sequential_encoder = Encoder::new();
        let sequential = sequential_encoder.encode_batch(&messages).unwrap().to_vec();

        // Both should decode to the same messages
        let decoded_p = decode_batch_parallel(&parallel).unwrap();
        let mut dec = Decoder::new();
        let decoded_s = dec.decode_all(&sequential).unwrap();

        assert_eq!(decoded_p.len(), decoded_s.len());
        for (a, b) in decoded_p.iter().zip(decoded_s.iter()) {
            assert_eq!(a, b);
        }
    }
}
