use serde_json::Value;

/// Inline encoder for small payloads — single-pass design.
///
/// # The Problem
/// Standard FluxPack uses symbol tables + DEF frames which add overhead for
/// small messages. JSON is just `{"key":value}` with minimal structure.
///
/// # The Solution: Single-Pass Inline Mode
/// For payloads < 256 bytes, skip the symbol table entirely.
/// Encode everything in ONE pass over the object.
///
/// Wire format (inline mode):
///   0xFE | field_count(u8) | for each field:
///     key_len(u8) | key_bytes | value_tag(u8) | value_bytes
///
/// Value tags:
///   0x00 = null       0x03 = i8         0x07 = f32
///   0x01 = true       0x04 = i16        0x08 = f64
///   0x02 = false      0x05 = i32        0x09 = string
///                     0x06 = i64        0x0B = u64

pub const INLINE_MAGIC: u8 = 0xFE;
pub const INLINE_THRESHOLD: usize = 256;

/// Single-pass encode: check eligibility AND encode in one iteration.
/// Returns None if the object can't use inline mode.
pub fn encode_inline(obj: &serde_json::Map<String, Value>) -> Option<Vec<u8>> {
    let field_count = obj.len();
    if field_count > 255 {
        return None;
    }

    let mut buf = Vec::with_capacity(64);
    buf.push(INLINE_MAGIC);
    buf.push(field_count as u8);

    for (key, value) in obj {
        // Check eligibility inline — no separate pass
        match value {
            Value::Array(_) | Value::Object(_) => return None,
            _ => {}
        }

        let key_bytes = key.as_bytes();
        if key_bytes.len() > 255 {
            return None;
        }

        buf.push(key_bytes.len() as u8);
        buf.extend_from_slice(key_bytes);

        if !encode_inline_value_fast(value, &mut buf) {
            return None;
        }
    }

    // Check final size
    if buf.len() > INLINE_THRESHOLD {
        return None;
    }

    Some(buf)
}

/// Fast-path value encoding. Returns false if value can't use inline mode.
#[inline(always)]
fn encode_inline_value_fast(value: &Value, buf: &mut Vec<u8>) -> bool {
    match value {
        Value::Null => {
            buf.push(0x00);
            true
        }
        Value::Bool(true) => {
            buf.push(0x01);
            true
        }
        Value::Bool(false) => {
            buf.push(0x02);
            true
        }
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                // Small integers: use smallest encoding
                if i >= 0 && i <= 255 {
                    buf.push(0x0C); // u8 tag
                    buf.push(i as u8);
                } else if i >= i8::MIN as i64 && i <= i8::MAX as i64 {
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
                true
            } else if let Some(u) = n.as_u64() {
                if u <= 255 {
                    buf.push(0x0C);
                    buf.push(u as u8);
                } else {
                    buf.push(0x0B);
                    buf.extend_from_slice(&u.to_le_bytes());
                }
                true
            } else if let Some(f) = n.as_f64() {
                if (f as f32) as f64 == f {
                    buf.push(0x07);
                    buf.extend_from_slice(&(f as f32).to_le_bytes());
                } else {
                    buf.push(0x08);
                    buf.extend_from_slice(&f.to_le_bytes());
                }
                true
            } else {
                false
            }
        }
        Value::String(s) => {
            let bytes = s.as_bytes();
            if bytes.len() > 255 {
                return false;
            }
            buf.push(0x09);
            buf.push(bytes.len() as u8);
            buf.extend_from_slice(bytes);
            true
        }
        Value::Array(_) | Value::Object(_) => false,
    }
}

/// Decode an inline-encoded payload, returning the map and bytes consumed.
pub fn decode_inline(input: &[u8]) -> Result<(serde_json::Map<String, Value>, usize), String> {
    if input.len() < 2 || input[0] != INLINE_MAGIC {
        return Err("not inline format".into());
    }

    let field_count = input[1] as usize;
    let mut cursor = 2;
    let mut obj = serde_json::Map::new();

    for _ in 0..field_count {
        if cursor >= input.len() {
            return Err("unexpected end".into());
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

#[inline(always)]
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
        0x0C => {
            if pos >= input.len() { return Err("truncated u8".into()); }
            let val = input[pos] as u64;
            pos += 1;
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

        let encoded = encode_inline(&obj).unwrap();
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
        let inline_bytes = encode_inline(&obj).unwrap();

        println!("JSON: {} bytes", json_bytes.len());
        println!("Inline: {} bytes", inline_bytes.len());

        assert!(inline_bytes.len() <= json_bytes.len());
    }

    #[test]
    fn test_inline_small_integers() {
        let obj = json!({
            "a": 42,
            "b": -1,
            "c": 127,
            "d": -128
        }).as_object().unwrap().clone();

        let encoded = encode_inline(&obj).unwrap();
        let (decoded, _) = decode_inline(&encoded).unwrap();

        assert_eq!(obj["a"], decoded["a"]);
        assert_eq!(obj["b"], decoded["b"]);
        assert_eq!(obj["c"], decoded["c"]);
        assert_eq!(obj["d"], decoded["d"]);
    }

    #[test]
    fn test_inline_u8_optimization() {
        // Values 0-255 should use 1-byte encoding
        let obj = json!({
            "small": 42,
            "zero": 0,
            "max": 255
        }).as_object().unwrap().clone();

        let encoded = encode_inline(&obj).unwrap();
        let (decoded, _) = decode_inline(&encoded).unwrap();

        assert_eq!(obj["small"], decoded["small"]);
        assert_eq!(obj["zero"], decoded["zero"]);
        assert_eq!(obj["max"], decoded["max"]);
    }

    #[test]
    fn test_inline_floats() {
        let obj = json!({
            "f32_ok": 3.14,
            "f64_needed": 3.141592653589793
        }).as_object().unwrap().clone();

        let encoded = encode_inline(&obj).unwrap();
        let (decoded, _) = decode_inline(&encoded).unwrap();

        let f32_val = decoded["f32_ok"].as_f64().unwrap();
        assert!((f32_val - 3.14).abs() < 0.01);

        let f64_val = decoded["f64_needed"].as_f64().unwrap();
        assert!((f64_val - 3.141592653589793).abs() < 1e-10);
    }

    #[test]
    fn test_inline_rejects_nested() {
        let obj = json!({
            "nested": {"inner": 1}
        }).as_object().unwrap().clone();

        assert!(encode_inline(&obj).is_none());
    }

    #[test]
    fn test_inline_rejects_arrays() {
        let obj = json!({
            "arr": [1, 2, 3]
        }).as_object().unwrap().clone();

        assert!(encode_inline(&obj).is_none());
    }

    #[test]
    fn test_inline_rejects_large() {
        let mut obj = serde_json::Map::new();
        for i in 0..100 {
            obj.insert(format!("key_{}", i), json!(i));
        }
        assert!(encode_inline(&obj).is_none());
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
        let inline_bytes = encode_inline(&obj).unwrap();
        let (decoded, _) = decode_inline(&inline_bytes).unwrap();

        println!("JSON: {} bytes", json_bytes.len());
        println!("Inline: {} bytes", inline_bytes.len());

        assert_eq!(obj.len(), decoded.len());
        assert_eq!(obj["user_id"], decoded["user_id"]);
    }
}
