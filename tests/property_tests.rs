use proptest::prelude::*;
use serde_json::json;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_decode_never_panics_on_random_bytes(bytes in prop::collection::vec(any::<u8>(), 0..1024)) {
        let mut decoder = fluxpack::Decoder::new();
        let _ = decoder.decode(&bytes);
    }

    #[test]
    fn test_zero_copy_never_panics_on_random_bytes(bytes in prop::collection::vec(any::<u8>(), 0..1024)) {
        let _ = fluxpack::decode_zero_copy(&bytes);
    }

    #[test]
    fn test_decode_all_never_panics_on_random_bytes(bytes in prop::collection::vec(any::<u8>(), 0..1024)) {
        let mut decoder = fluxpack::Decoder::new();
        let _ = decoder.decode_all(&bytes);
    }

    #[test]
    fn test_json_roundtrip_i64(i in any::<i64>()) {
        let original = json!({"value": i});
        let mut encoder = fluxpack::Encoder::new();
        let encoded = encoder.encode(&original).unwrap().to_vec();
        let mut decoder = fluxpack::Decoder::new();
        let decoded = decoder.decode(&encoded).unwrap();
        prop_assert_eq!(original, decoded);
    }

    #[test]
    fn test_json_roundtrip_f64(f in (-1e10f64..1e10f64).prop_filter("Non-NaN", |f: &f64| !f.is_nan())) {
        let original = json!({"value": f});
        let mut encoder = fluxpack::Encoder::new();
        let encoded = encoder.encode(&original).unwrap().to_vec();
        let mut decoder = fluxpack::Decoder::new();
        let decoded = decoder.decode(&encoded).unwrap();
        let orig_f = decoded["value"].as_f64().unwrap();
        prop_assert!((orig_f - f).abs() < 1e-6 || (orig_f / f - 1.0).abs() < 1e-6,
            "f64 roundtrip failed: {} -> {}", f, orig_f);
    }

    #[test]
    fn test_json_roundtrip_bool(b in any::<bool>()) {
        let original = json!({"value": b});
        let mut encoder = fluxpack::Encoder::new();
        let encoded = encoder.encode(&original).unwrap().to_vec();
        let mut decoder = fluxpack::Decoder::new();
        let decoded = decoder.decode(&encoded).unwrap();
        prop_assert_eq!(original, decoded);
    }

    #[test]
    fn test_json_roundtrip_string(s in "[a-zA-Z0-9_]{0,100}") {
        let original = json!({"value": s});
        let mut encoder = fluxpack::Encoder::new();
        let encoded = encoder.encode(&original).unwrap().to_vec();
        let mut decoder = fluxpack::Decoder::new();
        let decoded = decoder.decode(&encoded).unwrap();
        prop_assert_eq!(original, decoded);
    }

    #[test]
    fn test_json_roundtrip_array(arr in prop::collection::vec(any::<i64>(), 0..50)) {
        let original = json!({"values": arr});
        let mut encoder = fluxpack::Encoder::new();
        let encoded = encoder.encode(&original).unwrap().to_vec();
        let mut decoder = fluxpack::Decoder::new();
        let decoded = decoder.decode(&encoded).unwrap();
        prop_assert_eq!(original, decoded);
    }

    #[test]
    fn test_json_roundtrip_nested(
        a in any::<i64>(),
        b in "[a-z]{1,10}",
        c in (-1e10f64..1e10f64).prop_filter("Non-NaN", |f: &f64| !f.is_nan())
    ) {
        let original = json!({
            "int": a,
            "str": b,
            "float": c
        });
        let mut encoder = fluxpack::Encoder::new();
        let encoded = encoder.encode(&original).unwrap().to_vec();
        let mut decoder = fluxpack::Decoder::new();
        let decoded = decoder.decode(&encoded).unwrap();
        prop_assert_eq!(original, decoded);
    }

    #[test]
    fn test_batch_roundtrip(msgs in prop::collection::vec(any::<i64>(), 1..100)) {
        let mut encoder = fluxpack::Encoder::new();
        let mut encoded_batch = Vec::new();
        for m in &msgs {
            let val = json!({"value": m});
            let encoded = encoder.encode(&val).unwrap().to_vec();
            encoded_batch.push(encoded);
        }

        let mut decoder = fluxpack::Decoder::new();
        for (i, encoded) in encoded_batch.iter().enumerate() {
            let decoded = decoder.decode(encoded).unwrap();
            prop_assert_eq!(json!({"value": msgs[i]}), decoded);
        }
    }
}
