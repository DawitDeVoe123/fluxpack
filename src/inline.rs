use serde_json::Value;

/// Inline encoder for small payloads.
///
/// # The Problem
/// Standard FluxPack uses symbol tables + DEF frames which add overhead for
/// small messages. JSON is just `{"key":value}` with minimal structure.
///
/// # The Solution: Inline Mode
/// For payloads < 256 bytes, skip the symbol table entirely.
/// Use a compact inline format:
///   magic(1) | field_count(1) | for each field: key_len(1) | key_bytes | value_bytes
///
/// This eliminates:
/// - DEF frame overhead (saves 3-10 bytes per message)
/// - Symbol table hash map operations (saves CPU)
/// - Schema fingerprint computation (saves CPU)
/// - Varint encoding overhead (1-byte lengths for small strings)
///
/// Wire format (inline mode):
///   0xFE | field_count(u8) | for each field:
///     key_len(u8) | key_bytes | value_tag(u8) | value_bytes
///
/// Value tags (compact):
///   0x00 = null
///   0x01 = true
///   0x02 = false
///   0x03 = i8 (1 byte)
///   0x04 = i16 (2 bytes LE)
///   0x05 = i32 (4 bytes LE)
///   0x06 = i64 (8 bytes LE)
///   0x07 = f32 (4 bytes LE)
///   0x08 = f64 (8 bytes LE)
///   0x09 = string (u8 len + bytes)
///   0x0A = bytes (u8 len + bytes)

pub const INLINE_MAGIC: u8 = 0xFE;

/// Threshold: payloads smaller than this use inline mode.
pub const INLINE_THRESHOLD: usize = 256;

/// Encode a small JSON object in inline mode.
/// Returns the encoded bytes.
pub fn encode_inline(obj: &serde_json::Map<String, Value>) -> Vec<u8> {
    let field_count = obj.len();
    if field_count > 255 || !should_use_inline(obj) {
        return Vec::new(); // Signal to use standard mode
    }

    // Pre-calculate size
    let estimated_size = estimate_inline_size(obj);
    let mut buf = Vec::with_capacity(estimated_size);

    // Header
    buf.push(INLINE_MAGIC);
    buf.push(field_count as u8);

    // Fields
    for (key, value) in obj {
        // Key
        let key_bytes = key.as_bytes();
        buf.push(key_bytes.len() as u8);
        buf.extend_from_slice(key_bytes);

        // Value
        encode_inline_value(value, &mut buf);
    }

    buf
}

/// Estimate the size of inline encoding.
fn estimate_inline_size(obj: &serde_json::Map<String, Value>) -> usize {
    let mut size = 2; // magic + field_count
    for (key, value) in obj {
        size += 1 + key.len(); // key_len + key_bytes
        size += estimate_inline_value_size(value);
    }
    size
}

/// Check if inline mode should be used.
fn should_use_inline(obj: &serde_json::Map<String, Value>) -> bool {
    if obj.len() > 255 {
        return false;
    }

    // Don't use inline mode for nested structures
    for (_, value) in obj {
        match value {
            Value::Array(_) | Value::Object(_) => return false,
            _ => {}
        }
    }

    let mut total_size = 2; // magic + field_count
    for (key, value) in obj {
        total_size += 1 + key.len(); // key_len + key
        total_size += estimate_inline_value_size(value);
    }

    total_size <= INLINE_THRESHOLD
}

/// Estimate size of a single value in inline mode.
fn estimate_inline_value_size(value: &Value) -> usize {
    match value {
        Value::Null => 1,
        Value::Bool(_) => 1,
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                if i >= i8::MIN as i64 && i <= i8::MAX as i64 { 2 } // tag + i8
                else if i >= i16::MIN as i64 && i <= i16::MAX as i64 { 3 } // tag + i16
                else if i >= i32::MIN as i64 && i <= i32::MAX as i64 { 5 } // tag + i32
                else { 9 } // tag + i64
            } else if n.as_u64().is_some() {
                9 // tag + u64
            } else if let Some(f) = n.as_f64() {
                if (f as f32) as f64 == f { 5 } // tag + f32
                else { 9 } // tag + f64
            } else {
                9
            }
        }
        Value::String(s) => 1 + 1 + s.len(), // tag + len + bytes
        Value::Array(_) | Value::Object(_) => usize::MAX, // never use inline for nested
    }
}

/// Encode a value in inline mode.
fn encode_inline_value(value: &Value, buf: &mut Vec<u8>) {
    match value {
        Value::Null => buf.push(0x00),
        Value::Bool(true) => buf.push(0x01),
        Value::Bool(false) => buf.push(0x02),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                if i >= i8::MIN as i64 && i <= i8::MAX as i64 {
                    buf.push(0x03);
                    buf.push(i as u8);
                } else if i >= i16::MIN as i64 && i <= i16::MAX as i64 {
                    buf.push(0x04);
                    buf.extend_from_slice(&(i as i16).to_le_bytes());
                } else if i >= i32::MIN as i64 && i <= i32::MAX as i64 {
                    buf.push(0x05);
                    buf.extend_from_slice(&(i as i32).to_le_bytes());
                } else {
                    buf.push(0x06);
                    buf.extend_from_slice(&i.to_le_bytes());
                }
            } else if let Some(u) = n.as_u64() {
                // Use u64 tag for unsigned values
                buf.push(0x0B);
                buf.extend_from_slice(&u.to_le_bytes());
            } else if let Some(f) = n.as_f64() {
                if (f as f32) as f64 == f {
                    buf.push(0x07);
                    buf.extend_from_slice(&(f as f32).to_le_bytes());
                } else {
                    buf.push(0x08);
                    buf.extend_from_slice(&f.to_le_bytes());
                }
            }
        }
        Value::String(s) => {
            buf.push(0x09);
            let bytes = s.as_bytes();
            buf.push(bytes.len() as u8);
            buf.extend_from_slice(bytes);
        }
        Value::Array(_) | Value::Object(_) => {
            // Should not happen due to should_use_inline check
            buf.push(0x00);
        }
    }
}

/// Decode an inline-encoded payload, returning the map and bytes consumed.
pub fn decode_inline(input: &[u8]) -> Result<(serde_json::Map<String, Value>, usize), String> {
    if input.len() < 2 {
        return Err("input too short".into());
    }
    if input[0] != INLINE_MAGIC {
        return Err("not inline format".into());
    }

    let field_count = input[1] as usize;
    let mut cursor = 2;
    let mut obj = serde_json::Map::new();

    for _ in 0..field_count {
        if cursor >= input.len() {
            return Err("unexpected end of input".into());
        }

        let key_len = input[cursor] as usize;
        cursor += 1;

        if cursor + key_len > input.len() {
            return Err("key extends beyond input".into());
        }
        let key = std::str::from_utf8(&input[cursor..cursor + key_len])
            .map_err(|e| format!("invalid utf8: {}", e))?
            .to_string();
        cursor += key_len;

        let (value, new_cursor) = decode_inline_value(input, cursor)?;
        cursor = new_cursor;
        obj.insert(key, value);
    }

    Ok((obj, cursor))
}

/// Decode an inline value.
fn decode_inline_value(input: &[u8], cursor: usize) -> Result<(Value, usize), String> {
    if cursor >= input.len() {
        return Err("unexpected end in value".into());
    }

    let tag = input[cursor];
    let mut pos = cursor + 1;

    match tag {
        0x00 => Ok((Value::Null, pos)),
        0x01 => Ok((Value::Bool(true), pos)),
        0x02 => Ok((Value::Bool(false), pos)),
        0x03 => {
            if pos >= input.len() { return Err("truncated i8".into()); }
            let val = input[pos] as i8 as i64;
            pos += 1;
            Ok((Value::Number(val.into()), pos))
        }
        0x04 => {
            if pos + 2 > input.len() { return Err("truncated i16".into()); }
            let val = i16::from_le_bytes([input[pos], input[pos+1]]);
            pos += 2;
            Ok((Value::Number((val as i64).into()), pos))
        }
        0x05 => {
            if pos + 4 > input.len() { return Err("truncated i32".into()); }
            let val = i32::from_le_bytes([input[pos], input[pos+1], input[pos+2], input[pos+3]]);
            pos += 4;
            Ok((Value::Number((val as i64).into()), pos))
        }
        0x06 => {
            if pos + 8 > input.len() { return Err("truncated i64".into()); }
            let val = i64::from_le_bytes([
                input[pos], input[pos+1], input[pos+2], input[pos+3],
                input[pos+4], input[pos+5], input[pos+6], input[pos+7],
            ]);
            pos += 8;
            Ok((Value::Number(val.into()), pos))
        }
        0x07 => {
            if pos + 4 > input.len() { return Err("truncated f32".into()); }
            let bits = u32::from_le_bytes([input[pos], input[pos+1], input[pos+2], input[pos+3]]);
            pos += 4;
            let val = f32::from_bits(bits);
            Ok((Value::Number(serde_json::Number::from_f64(val as f64).unwrap_or(0.into())), pos))
        }
        0x08 => {
            if pos + 8 > input.len() { return Err("truncated f64".into()); }
            let bits = u64::from_le_bytes([
                input[pos], input[pos+1], input[pos+2], input[pos+3],
                input[pos+4], input[pos+5], input[pos+6], input[pos+7],
            ]);
            pos += 8;
            let val = f64::from_bits(bits);
            Ok((Value::Number(serde_json::Number::from_f64(val).unwrap_or(0.into())), pos))
        }
        0x09 => {
            if pos >= input.len() { return Err("truncated string len".into()); }
            let len = input[pos] as usize;
            pos += 1;
            if pos + len > input.len() { return Err("truncated string".into()); }
            let s = std::str::from_utf8(&input[pos..pos + len])
                .map_err(|e| format!("invalid utf8: {}", e))?
                .to_string();
            pos += len;
            Ok((Value::String(s), pos))
        }
        0x0B => {
            if pos + 8 > input.len() { return Err("truncated u64".into()); }
            let val = u64::from_le_bytes([
                input[pos], input[pos+1], input[pos+2], input[pos+3],
                input[pos+4], input[pos+5], input[pos+6], input[pos+7],
            ]);
            pos += 8;
            Ok((Value::Number(val.into()), pos))
        }
        _ => Err(format!("unknown tag: {}", tag)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_inline_roundtrip() {
        let obj = json!({
            "user_id": 8821,
            "email": "user@example.com",
            "active": true
        }).as_object().unwrap().clone();

        let encoded = encode_inline(&obj);
        assert!(!encoded.is_empty());
        assert_eq!(encoded[0], INLINE_MAGIC);

        let (decoded, consumed) = decode_inline(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(obj.len(), decoded.len());
        assert_eq!(obj["user_id"], decoded["user_id"]);
        assert_eq!(obj["email"], decoded["email"]);
        assert_eq!(obj["active"], decoded["active"]);
    }

    #[test]
    fn test_inline_size_comparison() {
        let obj = json!({
            "user_id": 8821,
            "email": "user@example.com",
            "active": true
        }).as_object().unwrap().clone();

        let json_bytes = serde_json::to_vec(&obj).unwrap();
        let inline_bytes = encode_inline(&obj);

        println!("JSON: {} bytes", json_bytes.len());
        println!("Inline: {} bytes", inline_bytes.len());
        println!("Savings: {} bytes ({:.1}%)",
            json_bytes.len() as isize - inline_bytes.len() as isize,
            (1.0 - inline_bytes.len() as f64 / json_bytes.len() as f64) * 100.0);

        assert!(inline_bytes.len() <= json_bytes.len(),
            "Inline ({}) should be <= JSON ({})",
            inline_bytes.len(), json_bytes.len());
    }

    #[test]
    fn test_inline_small_integers() {
        let obj = json!({
            "a": 42,
            "b": -1,
            "c": 127,
            "d": -128
        }).as_object().unwrap().clone();

        let encoded = encode_inline(&obj);
        let (decoded, _) = decode_inline(&encoded).unwrap();

        assert_eq!(obj["a"], decoded["a"]);
        assert_eq!(obj["b"], decoded["b"]);
        assert_eq!(obj["c"], decoded["c"]);
        assert_eq!(obj["d"], decoded["d"]);
    }

    #[test]
    fn test_inline_boundary_values() {
        let obj = json!({
            "i8_min": -128,
            "i8_max": 127,
            "i16_min": -32768,
            "i16_max": 32767
        }).as_object().unwrap().clone();

        let encoded = encode_inline(&obj);
        let (decoded, _) = decode_inline(&encoded).unwrap();

        assert_eq!(obj["i8_min"], decoded["i8_min"]);
        assert_eq!(obj["i8_max"], decoded["i8_max"]);
        assert_eq!(obj["i16_min"], decoded["i16_min"]);
        assert_eq!(obj["i16_max"], decoded["i16_max"]);
    }

    #[test]
    fn test_inline_floats() {
        let obj = json!({
            "f32_ok": 3.14,
            "f64_needed": 3.141592653589793
        }).as_object().unwrap().clone();

        let encoded = encode_inline(&obj);
        let (decoded, _) = decode_inline(&encoded).unwrap();

        let f32_val = decoded["f32_ok"].as_f64().unwrap();
        assert!((f32_val - 3.14).abs() < 0.01);

        let f64_val = decoded["f64_needed"].as_f64().unwrap();
        assert!((f64_val - 3.141592653589793).abs() < 1e-10);
    }

    #[test]
    fn test_inline_strings() {
        let obj = json!({
            "short": "hi",
            "long": "this is a longer string for testing"
        }).as_object().unwrap().clone();

        let encoded = encode_inline(&obj);
        let (decoded, _) = decode_inline(&encoded).unwrap();

        assert_eq!(obj["short"], decoded["short"]);
        assert_eq!(obj["long"], decoded["long"]);
    }

    #[test]
    fn test_inline_ml_payload() {
        let obj = json!({
            "user_id": 8821,
            "email": "user@example.com",
            "active": true,
            "score": 0.95
        }).as_object().unwrap().clone();

        let json_bytes = serde_json::to_vec(&obj).unwrap();
        let inline_bytes = encode_inline(&obj);
        let (decoded, _) = decode_inline(&inline_bytes).unwrap();

        println!("JSON: {} bytes", json_bytes.len());
        println!("Inline: {} bytes", inline_bytes.len());

        assert_eq!(obj.len(), decoded.len());
        assert_eq!(obj["user_id"], decoded["user_id"]);
        assert_eq!(obj["email"], decoded["email"]);
        assert_eq!(obj["active"], decoded["active"]);
    }

    #[test]
    fn test_inline_rejects_large() {
        let mut obj = serde_json::Map::new();
        for i in 0..100 {
            obj.insert(format!("key_{}", i), json!(i));
        }
        let encoded = encode_inline(&obj);
        assert!(encoded.is_empty(), "Large objects should use standard mode");
    }
}
