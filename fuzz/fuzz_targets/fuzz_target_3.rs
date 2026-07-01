#![no_main]
use libfuzzer_sys::fuzz_target;
use serde_json::{json, Value};

fuzz_target!(|data: &[u8]| {
    // Try to parse input as JSON first
    if let Ok(json_val) = serde_json::from_slice::<Value>(data) {
        // Roundtrip test: encode then decode
        let mut encoder = fluxpack::Encoder::new();
        if let Ok(encoded) = encoder.encode(&json_val) {
            let encoded_bytes = encoded.to_vec();

            // Standard decode must match
            let mut decoder = fluxpack::Decoder::new();
            if let Ok(decoded) = decoder.decode(&encoded_bytes) {
                // Values must be equal
                assert_eq!(json_val, decoded, "Roundtrip mismatch");
            }
        }
    }
});
