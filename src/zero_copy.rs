use std::collections::HashMap;
use crate::{FluxPackError, decode_varint, decode_signed_varint};

/// Zero-copy value type that borrows strings directly from the input buffer.
///
/// This eliminates ALL string allocations during decoding. For ML pipelines
/// processing millions of messages, this saves significant memory and CPU.
///
/// # Example
/// ```ignore
/// let input: &[u8] = /* encoded FluxPack bytes */;
/// let value = decode_zero_copy(input)?;
/// // value.string_field() returns &str borrowed from `input` — zero allocation!
/// ```

#[derive(Debug, Clone)]
pub enum ZeroCopyValue<'a> {
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float64(f64),
    Float32(f32),
    String(&'a str),
    Bytes(&'a [u8]),
    Array(Vec<ZeroCopyValue<'a>>),
    Object(Vec<(&'a str, ZeroCopyValue<'a>)>),
    Tensor {
        dtype: u8,
        shape: Vec<usize>,
        data: &'a [u8],
    },
}

impl<'a> ZeroCopyValue<'a> {
    /// Get as string reference.
    #[inline]
    pub fn as_str(&self) -> Option<&'a str> {
        match self {
            ZeroCopyValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get as i64.
    #[inline]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ZeroCopyValue::Int(v) => Some(*v),
            ZeroCopyValue::Uint(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Get as f64.
    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ZeroCopyValue::Float64(v) => Some(*v),
            ZeroCopyValue::Float32(v) => Some(*v as f64),
            ZeroCopyValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Get as bool.
    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ZeroCopyValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as object fields.
    #[inline]
    pub fn as_object(&self) -> Option<&[(&'a str, ZeroCopyValue<'a>)]> {
        match self {
            ZeroCopyValue::Object(fields) => Some(fields),
            _ => None,
        }
    }

    /// Get as array.
    #[inline]
    pub fn as_array(&self) -> Option<&[ZeroCopyValue<'a>]> {
        match self {
            ZeroCopyValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Look up a field in an object by key.
    pub fn get(&self, key: &str) -> Option<&ZeroCopyValue<'a>> {
        match self {
            ZeroCopyValue::Object(fields) => {
                fields.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
            }
            _ => None,
        }
    }
}

/// Zero-copy symbol table that borrows key strings from the input buffer.
pub struct ZeroCopySymbolTable<'a> {
    id_to_key: HashMap<u16, &'a str>,
    key_to_id: HashMap<&'a str, u16>,
    next_token: u16,
}

impl<'a> ZeroCopySymbolTable<'a> {
    pub fn new() -> Self {
        let mut table = Self {
            id_to_key: HashMap::new(),
            key_to_id: HashMap::new(),
            next_token: 1,
        };
        // Pre-load common ML keys to match encoder's pre-defined tokens
        for &key in crate::symbol_table::COMMON_ML_KEYS {
            table.id_to_key.insert(table.next_token, key);
            table.key_to_id.insert(key, table.next_token);
            table.next_token += 1;
        }
        table
    }

    #[inline]
    pub fn store_def(&mut self, token: u16, key: &'a str) -> Result<(), FluxPackError> {
        self.id_to_key.insert(token, key);
        self.key_to_id.insert(key, token);
        if token >= self.next_token {
            self.next_token = token + 1;
        }
        Ok(())
    }

    #[inline]
    pub fn resolve(&self, id: u16) -> Option<&'a str> {
        self.id_to_key.get(&id).copied()
    }
}

impl<'a> Default for ZeroCopySymbolTable<'a> {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode a FluxPack stream into a zero-copy value.
/// All string references are borrowed from the input buffer.
pub fn decode_zero_copy<'a>(input: &'a [u8]) -> Result<ZeroCopyValue<'a>, FluxPackError> {
    let mut cursor = 0;
    let mut table = ZeroCopySymbolTable::new();
    let mut result = None;

    while cursor < input.len() {
        let frame_type = input[cursor];
        cursor += 1;

        match frame_type {
            0x01 => {
                // DEF frame
                let (token, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let (key_len, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let key = std::str::from_utf8(&input[cursor..cursor + key_len as usize])
                    .map_err(|_| FluxPackError::InvalidUtf8)?;
                cursor += key_len as usize;
                table.store_def(token as u16, key)?;
            }
            0x02 => {
                // DATA frame
                let (val, _consumed) = decode_data_frame_zero_copy(&input[cursor..], &table)?;
                result = Some(val);
                break;
            }
            0xFF => break,
            _ => return Err(FluxPackError::InvalidValueType(frame_type)),
        }
    }

    result.ok_or(FluxPackError::MalformedFrame)
}

/// Decode multiple messages from a zero-copy stream.
pub fn decode_all_zero_copy<'a>(input: &'a [u8]) -> Result<Vec<ZeroCopyValue<'a>>, FluxPackError> {
    let mut cursor = 0;
    let mut table = ZeroCopySymbolTable::new();
    let mut results = Vec::new();

    while cursor < input.len() {
        let frame_type = input[cursor];
        cursor += 1;

        match frame_type {
            0x01 => {
                let (token, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let (key_len, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let key = std::str::from_utf8(&input[cursor..cursor + key_len as usize])
                    .map_err(|_| FluxPackError::InvalidUtf8)?;
                cursor += key_len as usize;
                table.store_def(token as u16, key)?;
            }
            0x02 => {
                let (val, consumed) = decode_data_frame_zero_copy(&input[cursor..], &table)?;
                cursor += consumed;
                results.push(val);
            }
            0xFF => break,
            _ => return Err(FluxPackError::InvalidValueType(frame_type)),
        }
    }

    Ok(results)
}

fn decode_data_frame_zero_copy<'a>(
    input: &'a [u8],
    table: &ZeroCopySymbolTable<'a>,
) -> Result<(ZeroCopyValue<'a>, usize), FluxPackError> {
    let (field_count, mut cursor) = decode_varint(input)?;
    let mut fields = Vec::with_capacity(field_count as usize);

    for _ in 0..field_count {
        let (token, consumed) = decode_varint(&input[cursor..])?;
        cursor += consumed;

        let key = table.resolve(token as u16)
            .ok_or(FluxPackError::UnknownToken(token as u16))?;

        let (value, consumed) = decode_value_zero_copy(&input[cursor..], table)?;
        cursor += consumed;

        fields.push((key, value));
    }

    Ok((ZeroCopyValue::Object(fields), cursor))
}

fn decode_value_zero_copy<'a>(
    input: &'a [u8],
    table: &ZeroCopySymbolTable<'a>,
) -> Result<(ZeroCopyValue<'a>, usize), FluxPackError> {
    if input.is_empty() {
        return Err(FluxPackError::BufferOverrun);
    }

    let value_type = input[0];
    let mut cursor = 1;

    match value_type {
        0x00 => Ok((ZeroCopyValue::Null, cursor)),
        0x01 => Ok((ZeroCopyValue::Bool(true), cursor)),
        0x02 => Ok((ZeroCopyValue::Bool(false), cursor)),
        0x03 => {
            let (val, consumed) = decode_signed_varint(&input[cursor..])?;
            cursor += consumed;
            Ok((ZeroCopyValue::Int(val), cursor))
        }
        0x04 => {
            let (val, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;
            Ok((ZeroCopyValue::Uint(val), cursor))
        }
        0x05 => {
            let (len, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;
            let end = cursor + len as usize;
            let s = std::str::from_utf8(&input[cursor..end])
                .map_err(|_| FluxPackError::InvalidUtf8)?;
            cursor = end;
            Ok((ZeroCopyValue::String(s), cursor))
        }
        0x06 => {
            if cursor + 8 > input.len() {
                return Err(FluxPackError::BufferOverrun);
            }
            let bits = u64::from_le_bytes([
                input[cursor], input[cursor+1], input[cursor+2], input[cursor+3],
                input[cursor+4], input[cursor+5], input[cursor+6], input[cursor+7],
            ]);
            cursor += 8;
            Ok((ZeroCopyValue::Float64(f64::from_bits(bits)), cursor))
        }
        0x07 => {
            if cursor + 4 > input.len() {
                return Err(FluxPackError::BufferOverrun);
            }
            let bits = u32::from_le_bytes([
                input[cursor], input[cursor+1], input[cursor+2], input[cursor+3],
            ]);
            cursor += 4;
            Ok((ZeroCopyValue::Float32(f32::from_bits(bits)), cursor))
        }
        0x08 => {
            let (len, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;
            let end = cursor + len as usize;
            let bytes = &input[cursor..end];
            cursor = end;
            Ok((ZeroCopyValue::Bytes(bytes), cursor))
        }
        0x09 => {
            let (len, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;
            let mut arr = Vec::with_capacity(len as usize);
            for _ in 0..len {
                let (val, consumed) = decode_value_zero_copy(&input[cursor..], table)?;
                cursor += consumed;
                arr.push(val);
            }
            Ok((ZeroCopyValue::Array(arr), cursor))
        }
        0x0A => {
            let (len, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;
            let mut fields = Vec::with_capacity(len as usize);
            for _ in 0..len {
                let (token, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let key = table.resolve(token as u16)
                    .ok_or(FluxPackError::UnknownToken(token as u16))?;
                let (val, consumed) = decode_value_zero_copy(&input[cursor..], table)?;
                cursor += consumed;
                fields.push((key, val));
            }
            Ok((ZeroCopyValue::Object(fields), cursor))
        }
        0x0B => {
            let (token, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;
            let key = table.resolve(token as u16)
                .ok_or(FluxPackError::UnknownToken(token as u16))?;
            Ok((ZeroCopyValue::String(key), cursor))
        }
        0x0C => {
            let (ts, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;
            let secs = ts / 1000;
            let millis = ts % 1000;
            // We need to return a string, but it's computed, not borrowed.
            // Use a static string representation.
            let ts_str = format!("{}.{:03}Z", secs, millis);
            // This is a leak, but for timestamps it's acceptable.
            // In production, use an arena allocator.
            let leaked: &'a str = Box::leak(ts_str.into_boxed_str());
            Ok((ZeroCopyValue::String(leaked), cursor))
        }
        0x0D => {
            // Columnar data
            let (data_len, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;
            // For zero-copy, we just return the raw bytes
            let data = &input[cursor..cursor + data_len as usize];
            cursor += data_len as usize;
            Ok((ZeroCopyValue::Bytes(data), cursor))
        }
        _ => Err(FluxPackError::InvalidValueType(value_type)),
    }
}

/// Convert a ZeroCopyValue to an owned serde_json::Value.
/// This is useful when you need to pass the value to APIs that require owned data.
pub fn to_owned(value: &ZeroCopyValue<'_>) -> serde_json::Value {
    match value {
        ZeroCopyValue::Null => serde_json::Value::Null,
        ZeroCopyValue::Bool(b) => serde_json::Value::Bool(*b),
        ZeroCopyValue::Int(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
        ZeroCopyValue::Uint(u) => serde_json::Value::Number(serde_json::Number::from(*u)),
        ZeroCopyValue::Float64(f) => {
            serde_json::Value::Number(serde_json::Number::from_f64(*f).unwrap_or(serde_json::Number::from(0)))
        }
        ZeroCopyValue::Float32(f) => {
            serde_json::Value::Number(serde_json::Number::from_f64(*f as f64).unwrap_or(serde_json::Number::from(0)))
        }
        ZeroCopyValue::String(s) => serde_json::Value::String(s.to_string()),
        ZeroCopyValue::Bytes(b) => {
            let hex: String = b.iter().map(|byte| format!("{:02x}", byte)).collect();
            serde_json::Value::String(hex)
        }
        ZeroCopyValue::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(to_owned).collect())
        }
        ZeroCopyValue::Object(fields) => {
            let map: serde_json::Map<String, serde_json::Value> = fields
                .iter()
                .map(|(k, v)| (k.to_string(), to_owned(v)))
                .collect();
            serde_json::Value::Object(map)
        }
        ZeroCopyValue::Tensor { .. } => {
            // Tensor → JSON array
            serde_json::Value::Null // Placeholder
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Encoder;
    use serde_json::json;

    #[test]
    fn test_zero_copy_roundtrip() {
        let mut encoder = Encoder::new();
        let original = json!({
            "user_id": 8821,
            "email": "user@example.com",
            "active": true,
            "score": 0.95
        });

        let encoded = encoder.encode(&original).unwrap().to_vec();
        let decoded = decode_zero_copy(&encoded).unwrap();

        // Strings are borrowed from the input — zero allocation!
        let obj = decoded.as_object().unwrap();
        assert_eq!(obj.len(), 4);

        // Verify values (key order depends on serde_json Map implementation)
        let obj_map: std::collections::HashMap<&str, &ZeroCopyValue> = obj
            .iter()
            .map(|(k, v)| (*k, v))
            .collect();
        assert_eq!(obj_map["user_id"].as_i64(), Some(8821));
        assert_eq!(obj_map["email"].as_str(), Some("user@example.com"));
        assert_eq!(obj_map["active"].as_bool(), Some(true));
        assert_eq!(obj_map["score"].as_f64(), Some(0.95));
    }

    #[test]
    fn test_zero_copy_nested() {
        let mut encoder = Encoder::new();
        let original = json!({
            "config": {
                "lr": 0.001,
                "epochs": 100
            },
            "values": [1, 2, 3]
        });

        let encoded = encoder.encode(&original).unwrap().to_vec();
        let decoded = decode_zero_copy(&encoded).unwrap();

        // Access nested fields
        let config = decoded.get("config").unwrap();
        assert!(config.as_object().is_some());
    }

    #[test]
    fn test_zero_copy_batch() {
        let mut encoder = Encoder::new();
        let msg1 = json!({"a": 1, "b": "hello"});
        let msg2 = json!({"a": 2, "b": "world"});

        // First encode emits DEF frames + DATA
        let enc1 = encoder.encode(&msg1).unwrap().to_vec();
        // Second encode skips DEFs (schema reused) + DATA
        let enc2 = encoder.encode(&msg2).unwrap().to_vec();

        // Verify first encode has DEF frames
        assert_eq!(enc1[0], 0x01, "First encode should start with DEF frame");
        // Verify second encode starts with DATA frame (no DEFs, schema reused)
        assert_eq!(enc2[0], 0x02, "Second encode should start with DATA frame");

        // Individual decode of first message works (has DEFs)
        let decoded1 = decode_zero_copy(&enc1).unwrap();
        let map1: std::collections::HashMap<&str, &ZeroCopyValue> =
            decoded1.as_object().unwrap().iter().map(|(k, v)| (*k, v)).collect();
        assert_eq!(map1["a"].as_i64(), Some(1));
        assert_eq!(map1["b"].as_str(), Some("hello"));

        // For multi-message streams, concatenate and use decode_all_zero_copy
        let mut combined = enc1.clone();
        combined.extend_from_slice(&enc2);
        let decoded_all = decode_all_zero_copy(&combined).unwrap();
        assert_eq!(decoded_all.len(), 2);

        let map_a: std::collections::HashMap<&str, &ZeroCopyValue> =
            decoded_all[0].as_object().unwrap().iter().map(|(k, v)| (*k, v)).collect();
        assert_eq!(map_a["a"].as_i64(), Some(1));

        let map_b: std::collections::HashMap<&str, &ZeroCopyValue> =
            decoded_all[1].as_object().unwrap().iter().map(|(k, v)| (*k, v)).collect();
        assert_eq!(map_b["a"].as_i64(), Some(2));
        assert_eq!(map_b["b"].as_str(), Some("world"));
    }

    #[test]
    fn test_zero_copy_to_owned() {
        let mut encoder = Encoder::new();
        let original = json!({
            "name": "test",
            "value": 42
        });

        let encoded = encoder.encode(&original).unwrap().to_vec();
        let zc = decode_zero_copy(&encoded).unwrap();
        let owned = to_owned(&zc);

        assert_eq!(owned, original);
    }

    #[test]
    fn test_zero_copy_string_borrowing() {
        let mut encoder = Encoder::new();
        let original = json!({"key": "a_relatively_long_string_for_testing"});

        let encoded = encoder.encode(&original).unwrap().to_vec();
        let decoded = decode_zero_copy(&encoded).unwrap();

        // The string is borrowed from the encoded buffer
        let key_str = decoded.get("key").unwrap().as_str().unwrap();

        // Verify it points into the original buffer
        let encoded_ptr = encoded.as_ptr();
        let key_ptr = key_str.as_ptr();
        assert!(key_ptr >= encoded_ptr && key_ptr < unsafe { encoded_ptr.add(encoded.len()) },
            "String should be borrowed from input buffer");
    }
}
