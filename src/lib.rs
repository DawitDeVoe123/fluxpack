//! FluxPack — A schema-free, Shannon-optimal serialisation format.
//!
//! This crate implements the FluxPack wire format specification v1.0.
//! For full details, see the spec at `/docs/spec.md`.

pub mod symbol_table;
pub mod varint;
pub mod error;
pub mod encoder;
pub mod decoder;
pub mod columnar;
pub mod stream;
pub mod delta;
pub mod tensor;
pub mod zero_copy;
pub mod ml;
pub mod inline;
pub mod fast_decoder;

#[cfg(feature = "parallel")]
pub mod parallel;

#[cfg(feature = "compression")]
pub mod compress;

#[cfg(feature = "python")]
mod python;

pub use symbol_table::SymbolTable;
pub use varint::{encode_varint, decode_varint, varint_len, zigzag_encode, zigzag_decode,
    encode_signed_varint, decode_signed_varint};
pub use error::FluxPackError;
pub use encoder::Encoder;
pub use decoder::Decoder;
pub use fast_decoder::FastDecoder;
pub use columnar::{try_columnarize, encode_columnar, decode_columnar, reconstruct_array, ColType};
pub use stream::{StreamWriter, StreamReader, Frame};
pub use delta::{encode_delta, decode_delta, should_delta_encode};
pub use tensor::{Tensor, TensorDtype, encode_tensor, decode_tensor};
pub use zero_copy::{ZeroCopyValue, decode_zero_copy, decode_all_zero_copy, to_owned};
pub use ml::{
    FeatureVector, encode_feature_vector, decode_feature_vector,
    SparseTensor, encode_sparse_tensor, decode_sparse_tensor, sparse_to_dense,
    Hyperparams, encode_hyperparams, decode_hyperparams,
    encode_timestamp, decode_timestamp,
    encode_timestamps_deltas, decode_timestamps_deltas,
};

/// Magic bytes that identify a FluxPack stream: F X P 0x01
pub const MAGIC: [u8; 4] = [0x46, 0x58, 0x50, 0x01];

/// Maximum number of tokens per session (14-bit space)
pub const MAX_TOKENS: u16 = 0x3FFF;

/// Frame types
#[repr(u8)]
pub enum FrameType {
    Def = 0x01,
    Data = 0x02,
    Struct = 0x03,
    Reset = 0x04,
    Debug = 0x05,
    Ack = 0x06,
    Columnar = 0x0D,
    Eos = 0xFF,
}

/// Value types
#[repr(u8)]
pub enum ValueType {
    Null = 0x00,
    BoolTrue = 0x01,
    BoolFalse = 0x02,
    IntVar = 0x03,
    UintVar = 0x04,
    String = 0x05,
    Float64 = 0x06,
    Float32 = 0x07,
    Bytes = 0x08,
    Array = 0x09,
    Object = 0x0A,
    Interned = 0x0B,
    Timestamp = 0x0C,
    Columnar = 0x0D,
    Tensor = 0x10,
    DeltaNums = 0x11,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    #[test]
    fn test_encoder_basic() {
        let mut encoder = Encoder::new();
        let msg = json!({
            "user_id": 8821,
            "email": "user@example.com",
            "active": true
        });

        let encoded = encoder.encode(&msg).unwrap();
        assert!(!encoded.is_empty());
        
        let mut data_frame_start = None;
        for i in 0..encoded.len() {
            if encoded[i] == 0x02 {
                data_frame_start = Some(i);
                break;
            }
        }
        assert!(data_frame_start.is_some(), "DATA frame not found");
    }

    #[test]
    fn test_encoder_decoder_roundtrip() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        
        let original = json!({
            "user_id": 8821,
            "email": "user@example.com",
            "active": true,
            "nested": {
                "field1": "value1",
                "field2": 42
            },
            "list": [1, 2, 3, "hello"]
        });

        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();
        
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_roundtrip_negative_numbers() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        
        let original = json!({
            "loss": -0.5,
            "gradient": -42,
            "delta": -1,
            "positive": 100
        });

        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();
        
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_roundtrip_nested_objects() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        
        let original = json!({
            "training": {
                "config": {
                    "lr": 0.001,
                    "epochs": 100,
                    "batch_size": 32
                },
                "metrics": {
                    "accuracy": 0.95,
                    "loss": 0.05
                }
            }
        });

        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();
        
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_roundtrip_arrays() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        
        let original = json!({
            "values": [1, 2, 3, 4, 5],
            "names": ["a", "b", "c"],
            "mixed": [1, "two", true, null, 3.14]
        });

        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();
        
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_schema_reuse_skips_defs() {
        let mut encoder = Encoder::new();
        
        let msg1 = json!({"a": 1, "b": 2});
        let msg2 = json!({"a": 3, "b": 4});

        let encoded1 = encoder.encode(&msg1).unwrap().to_vec();
        let encoded2 = encoder.encode(&msg2).unwrap().to_vec();

        assert!(encoded2.len() < encoded1.len(), 
            "Second message ({}) should be smaller than first ({})", 
            encoded2.len(), encoded1.len());
    }

    #[test]
    fn test_batch_encode() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();

        let messages = vec![
            json!({"x": 1, "y": 2}),
            json!({"x": 3, "y": 4}),
            json!({"x": 5, "y": 6}),
        ];

        let encoded = encoder.encode_batch(&messages).unwrap().to_vec();
        let decoded = decoder.decode_all(&encoded).unwrap();

        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0], messages[0]);
        assert_eq!(decoded[1], messages[1]);
        assert_eq!(decoded[2], messages[2]);
    }

    #[test]
    fn test_stream_roundtrip() {
        use stream::{StreamWriter, StreamReader, Frame};

        let messages = vec![
            json!({"epoch": 1, "loss": 0.5}),
            json!({"epoch": 2, "loss": 0.3}),
            json!({"epoch": 3, "loss": 0.1}),
        ];

        let mut writer = StreamWriter::new();
        for msg in &messages {
            writer.write_data(msg).unwrap();
        }
        let bytes = writer.into_buffer();

        let mut reader = StreamReader::new();
        let decoded = reader.read_all(&bytes).unwrap();

        let data_frames: Vec<&Value> = decoded.iter().filter_map(|f| match f {
            Frame::Data(v) => Some(v),
            _ => None,
        }).collect();

        assert_eq!(data_frames.len(), 3);
        assert_eq!(*data_frames[0], messages[0]);
        assert_eq!(*data_frames[1], messages[1]);
        assert_eq!(*data_frames[2], messages[2]);
    }

    #[test]
    fn test_columnar_roundtrip() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();

        let original = json!({
            "metrics": [
                {"epoch": 1, "loss": 0.5, "acc": 0.8},
                {"epoch": 2, "loss": 0.3, "acc": 0.9},
                {"epoch": 3, "loss": 0.1, "acc": 0.95}
            ]
        });

        let encoded = encoder.encode_with_columnar(&original).unwrap().to_vec();
        let decoded = decoder.decode(&encoded).unwrap();
        
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_zigzag_optimal_for_small_negatives() {
        let mut encoder = Encoder::new();
        
        let msg_small_neg = json!({"val": -1});
        let msg_large_uint = json!({"val": 200});

        let enc_small = encoder.encode(&msg_small_neg).unwrap().to_vec();
        encoder.reset();
        let enc_large = encoder.encode(&msg_large_uint).unwrap().to_vec();

        assert!(enc_small.len() <= enc_large.len());
    }

    #[test]
    fn test_encode_with_columnar() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();

        let original = json!({
            "training_log": [
                {"step": 0, "lr": 0.001, "loss": 1.0},
                {"step": 1, "lr": 0.001, "loss": 0.8},
                {"step": 2, "lr": 0.001, "loss": 0.6},
                {"step": 3, "lr": 0.0005, "loss": 0.4},
                {"step": 4, "lr": 0.0005, "loss": 0.2},
            ]
        });

        let encoded = encoder.encode_with_columnar(&original).unwrap().to_vec();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(original, decoded);
    }

    #[test]
    fn test_float64_precision() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();

        let original = json!({
            "pi": 3.141592653589793,
            "small": 1e-10,
            "large": 1e10
        });

        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();

        assert_eq!(original, decoded);
    }

    #[test]
    fn test_delta_encoding() {
        let values = vec![json!(100), json!(101), json!(103), json!(100)];
        let mut buf = Vec::new();
        encode_delta(&values, &mut buf).unwrap();
        let (decoded, _) = decode_delta(&buf).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_tensor_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_f32(&data, vec![2, 3]);

        let mut buf = Vec::new();
        encode_tensor(&tensor, &mut buf);

        let (decoded, _) = decode_tensor(&buf).unwrap();
        assert_eq!(decoded.shape, vec![2, 3]);
        assert_eq!(decoded.data, tensor.data);
    }

    #[test]
    fn test_zero_copy() {
        let mut encoder = Encoder::new();
        let original = json!({"key": "value", "count": 42});

        let encoded = encoder.encode(&original).unwrap().to_vec();
        let decoded = decode_zero_copy(&encoded).unwrap();

        assert_eq!(decoded.get("key").unwrap().as_str(), Some("value"));
        assert_eq!(decoded.get("count").unwrap().as_i64(), Some(42));
    }

    #[test]
    fn test_total_pipeline_json_vs_fluxpack() {
        let data = json!({
            "experiment_id": "exp_001",
            "model": "transformer_v3",
            "config": {"d_model": 512, "n_heads": 8, "n_layers": 6},
            "metrics": {"accuracy": 0.94, "f1": 0.93, "loss": 0.05},
            "history": [
                {"epoch": 1, "loss": 2.5},
                {"epoch": 2, "loss": 1.8},
                {"epoch": 3, "loss": 1.2}
            ]
        });

        let json_bytes = serde_json::to_vec(&data).unwrap();
        let mut encoder = Encoder::new();
        let fp_bytes = encoder.encode(&data).unwrap().to_vec();

        println!("JSON: {} bytes", json_bytes.len());
        println!("FluxPack: {} bytes ({:.1}% smaller)",
            fp_bytes.len(),
            (1.0 - fp_bytes.len() as f64 / json_bytes.len() as f64) * 100.0);

        assert!(fp_bytes.len() < json_bytes.len());
    }

    #[test]
    fn test_edge_empty_object() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        let original = json!({});
        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_edge_unicode_strings() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        let original = json!({
            "emoji": "🚀💻🧠",
            "chinese": "机器学习",
            "arabic": "تعلم الآلة",
            "japanese": "機械学習",
            "mixed": "Hello 世界 🌍"
        });
        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_edge_deeply_nested() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        let original = json!({
            "l1": {
                "l2": {
                    "l3": {
                        "l4": {
                            "l5": {
                                "l6": {
                                    "l7": {
                                        "l8": {
                                            "l9": {
                                                "l10": "deep value"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_edge_large_array() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        let values: Vec<serde_json::Value> = (0..10000).map(|i| json!(i)).collect();
        let original = json!({"values": values});
        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_edge_boundary_varint_values() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        let original = json!({
            "zero": 0,
            "one": 1,
            "127": 127,
            "128": 128,
            "16383": 16383,
            "16384": 16384,
            "max_u64": 18446744073709551615_u64
        });
        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_edge_negative_numbers() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        let original = json!({
            "neg_one": -1,
            "neg_max_i32": -2147483648,
            "neg_large": -9223372036854775808_i64,
            "float_neg": -3.14159,
            "float_small": -1e-10
        });
        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_edge_empty_arrays() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        let original = json!({
            "empty_arr": [],
            "empty_obj": {},
            "nested_empty": [[], [], []]
        });
        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_edge_mixed_types() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        let original = json!({
            "null_val": null,
            "bool_true": true,
            "bool_false": false,
            "int": 42,
            "float": 3.14,
            "string": "hello",
            "array": [1, 2, 3],
            "object": {"key": "value"}
        });
        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_edge_special_float_values() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        let original = json!({
            "pi": 3.141592653589793,
            "e": 2.718281828459045,
            "sqrt2": 1.4142135623730951,
            "zero": 0.0,
            "neg_zero": -0.0,
            "large": 1.7976931348623157e+308,
            "tiny": 5e-324
        });
        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_edge_long_strings() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        let long_string = "a".repeat(10000);
        let original = json!({
            "long": long_string,
            "short": "x"
        });
        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_edge_many_keys() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        let mut obj = serde_json::Map::new();
        for i in 0..100 {
            obj.insert(format!("key_{}", i), json!(i));
        }
        let original = Value::Object(obj);
        let encoded = encoder.encode(&original).unwrap();
        let decoded = decoder.decode(encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_edge_batch_large() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        let messages: Vec<Value> = (0..1000).map(|i| {
            json!({
                "id": i,
                "value": i as f64 * 0.1,
                "name": format!("item_{}", i)
            })
        }).collect();
        let encoded = encoder.encode_batch(&messages).unwrap().to_vec();
        let decoded = decoder.decode_all(&encoded).unwrap();
        assert_eq!(decoded.len(), 1000);
        for (i, msg) in decoded.iter().enumerate() {
            assert_eq!(msg["id"], json!(i));
        }
    }

    #[test]
    fn test_edge_zero_copy_unicode() {
        let mut encoder = Encoder::new();
        let original = json!({
            "emoji": "🚀",
            "text": "Hello 世界"
        });
        let encoded = encoder.encode(&original).unwrap().to_vec();
        let decoded = decode_zero_copy(&encoded).unwrap();
        assert_eq!(decoded.get("emoji").unwrap().as_str(), Some("🚀"));
        assert_eq!(decoded.get("text").unwrap().as_str(), Some("Hello 世界"));
    }

    #[test]
    fn test_edge_columnar_empty() {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        let original = json!({
            "empty_col": []
        });
        let encoded = encoder.encode_with_columnar(&original).unwrap().to_vec();
        let decoded = decoder.decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_edge_delta_encoding_negative() {
        let values = vec![json!(-100), json!(-99), json!(-97), json!(-93)];
        let mut buf = Vec::new();
        encode_delta(&values, &mut buf).unwrap();
        let (decoded, _) = decode_delta(&buf).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_edge_tensor_large() {
        let data: Vec<f64> = (0..100000).map(|i| i as f64 * 0.001).collect();
        let tensor = Tensor::from_f64(&data, vec![1000, 100]);

        let mut buf = Vec::new();
        encode_tensor(&tensor, &mut buf);

        let (decoded, _) = decode_tensor(&buf).unwrap();
        assert_eq!(decoded.shape, vec![1000, 100]);
        assert_eq!(decoded.data, tensor.data);
    }
}
