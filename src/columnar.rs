use serde_json::{Value, Map, Number};
use crate::{encode_varint, decode_varint, encode_signed_varint};

/// Minimum array length to trigger columnar encoding.
pub const COLUMNAR_THRESHOLD: usize = 3;

/// Column type tags
const COL_NULL: u8 = 0x00;
const COL_BOOL: u8 = 0x01;
const COL_INT: u8 = 0x02;
const COL_UINT: u8 = 0x03;
const COL_FLOAT64: u8 = 0x04;
const COL_FLOAT32: u8 = 0x05;
const COL_STRING: u8 = 0x06;
const COL_MIXED: u8 = 0xFF;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColType {
    Null,
    Bool,
    Int,
    UInt,
    Float64,
    Float32,
    String,
    Mixed,
}

impl ColType {
    #[inline]
    pub fn tag(self) -> u8 {
        match self {
            ColType::Null => COL_NULL,
            ColType::Bool => COL_BOOL,
            ColType::Int => COL_INT,
            ColType::UInt => COL_UINT,
            ColType::Float64 => COL_FLOAT64,
            ColType::Float32 => COL_FLOAT32,
            ColType::String => COL_STRING,
            ColType::Mixed => COL_MIXED,
        }
    }

    #[inline]
    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            COL_NULL => Some(ColType::Null),
            COL_BOOL => Some(ColType::Bool),
            COL_INT => Some(ColType::Int),
            COL_UINT => Some(ColType::UInt),
            COL_FLOAT64 => Some(ColType::Float64),
            COL_FLOAT32 => Some(ColType::Float32),
            COL_STRING => Some(ColType::String),
            COL_MIXED => Some(ColType::Mixed),
            _ => None,
        }
    }
}

/// A decoded column with owned values.
pub struct DecodedColumn {
    pub key: String,
    pub col_type: ColType,
    pub values: Vec<Value>,
}

/// Attempts to columnarize an array of objects.
/// Returns Some(cols) if all elements are objects with the same keys.
pub fn try_columnarize(arr: &[Value]) -> Option<Vec<(String, ColType)>> {
    if arr.len() < COLUMNAR_THRESHOLD {
        return None;
    }

    let first = match &arr[0] {
        Value::Object(o) => o,
        _ => return None,
    };

    let keys: Vec<String> = first.keys().cloned().collect();
    if keys.is_empty() {
        return None;
    }

    // Verify all elements have the same keys
    for elem in arr.iter().skip(1) {
        match elem {
            Value::Object(obj) => {
                if obj.len() != keys.len() {
                    return None;
                }
                for key in &keys {
                    if !obj.contains_key(key) {
                        return None;
                    }
                }
            }
            _ => return None,
        }
    }

    // All objects have the same keys — determine column types
    let mut columns = Vec::with_capacity(keys.len());
    for key in &keys {
        let values: Vec<&Value> = arr
            .iter()
            .filter_map(|v| v.as_object()?.get(key.as_str()))
            .collect();
        let col_type = classify_column(&values);
        columns.push((key.clone(), col_type));
    }

    Some(columns)
}

/// Classify a column by detecting the dominant type.
fn classify_column(values: &[&Value]) -> ColType {
    if values.is_empty() {
        return ColType::Null;
    }

    let mut has_null = false;
    let mut has_bool = false;
    let mut has_int = false;
    let mut has_uint = false;
    let mut has_float = false;
    let mut has_string = false;

    for v in values {
        match v {
            Value::Null => has_null = true,
            Value::Bool(_) => has_bool = true,
            Value::Number(n) => {
                if n.is_i64() {
                    if n.as_i64().unwrap() < 0 {
                        has_int = true;
                    } else {
                        has_uint = true;
                    }
                } else if n.is_u64() {
                    has_uint = true;
                } else if n.is_f64() {
                    has_float = true;
                }
            }
            Value::String(_) => has_string = true,
            _ => return ColType::Mixed,
        }
    }

    let type_count = [has_null, has_bool, has_int, has_uint, has_float, has_string]
        .iter()
        .filter(|&&b| b)
        .count();

    if type_count == 1 {
        if has_bool { return ColType::Bool; }
        if has_int { return ColType::Int; }
        if has_uint { return ColType::UInt; }
        if has_float { return ColType::Float64; }
        if has_string { return ColType::String; }
        return ColType::Null;
    }

    // Mixed numeric → Float64
    if !has_string && !has_bool && (has_int || has_uint || has_float) {
        return ColType::Float64;
    }

    // Bools + nulls
    if has_bool && has_null && !has_int && !has_uint && !has_float && !has_string {
        return ColType::Bool;
    }

    ColType::Mixed
}

/// Encode a columnar array into a buffer.
/// Format: col_count(varint) | for each column: key_len(varint) | key_bytes | col_type(byte) | row_count(varint) | values
pub fn encode_columnar(arr: &[Value], columns: &[(String, ColType)], buffer: &mut Vec<u8>) {
    let row_count = arr.len();
    encode_varint(columns.len() as u64, buffer);

    for (key, col_type) in columns {
        // Column header
        encode_varint(key.len() as u64, buffer);
        buffer.extend_from_slice(key.as_bytes());
        buffer.push(col_type.tag());
        encode_varint(row_count as u64, buffer);

        // Column values
        encode_column_values(arr, key, *col_type, buffer);
    }
}

#[inline]
fn encode_column_values(arr: &[Value], key: &str, col_type: ColType, buffer: &mut Vec<u8>) {
    match col_type {
        ColType::Null => {}
        ColType::Bool => {
            let mut byte = 0u8;
            let mut bit_idx = 0;
            for v in arr {
                if let Some(obj) = v.as_object() {
                    if let Some(Value::Bool(true)) = obj.get(key) {
                        byte |= 1 << bit_idx;
                    }
                }
                bit_idx += 1;
                if bit_idx == 8 {
                    buffer.push(byte);
                    byte = 0;
                    bit_idx = 0;
                }
            }
            if bit_idx > 0 {
                buffer.push(byte);
            }
        }
        ColType::Int => {
            for v in arr {
                if let Some(obj) = v.as_object() {
                    if let Some(val) = obj.get(key) {
                        let i = val.as_i64().unwrap_or(0);
                        encode_signed_varint(i, buffer);
                    }
                }
            }
        }
        ColType::UInt => {
            for v in arr {
                if let Some(obj) = v.as_object() {
                    if let Some(val) = obj.get(key) {
                        let u = val.as_u64().unwrap_or(0);
                        encode_varint(u, buffer);
                    }
                }
            }
        }
        ColType::Float64 => {
            for v in arr {
                if let Some(obj) = v.as_object() {
                    match obj.get(key) {
                        Some(Value::Number(n)) => {
                            let f = n.as_f64().unwrap_or(0.0);
                            buffer.extend_from_slice(&f.to_bits().to_le_bytes());
                        }
                        Some(Value::Null) | None => {
                            buffer.extend_from_slice(&f64::NAN.to_bits().to_le_bytes());
                        }
                        _ => {
                            buffer.extend_from_slice(&0.0f64.to_bits().to_le_bytes());
                        }
                    }
                }
            }
        }
        ColType::Float32 => {
            for v in arr {
                if let Some(obj) = v.as_object() {
                    match obj.get(key) {
                        Some(Value::Number(n)) => {
                            let f = n.as_f64().unwrap_or(0.0) as f32;
                            buffer.extend_from_slice(&f.to_bits().to_le_bytes());
                        }
                        Some(Value::Null) | None => {
                            buffer.extend_from_slice(&f32::NAN.to_bits().to_le_bytes());
                        }
                        _ => {
                            buffer.extend_from_slice(&0.0f32.to_bits().to_le_bytes());
                        }
                    }
                }
            }
        }
        ColType::String => {
            for v in arr {
                if let Some(obj) = v.as_object() {
                    match obj.get(key) {
                        Some(Value::String(s)) => {
                            encode_varint(s.len() as u64, buffer);
                            buffer.extend_from_slice(s.as_bytes());
                        }
                        _ => {
                            encode_varint(0, buffer);
                        }
                    }
                }
            }
        }
        ColType::Mixed => {
            for v in arr {
                if let Some(obj) = v.as_object() {
                    if let Some(val) = obj.get(key) {
                        encode_mixed_value(val, buffer);
                    }
                }
            }
        }
    }
}

#[inline]
fn encode_mixed_value(v: &Value, buffer: &mut Vec<u8>) {
    match v {
        Value::Null => buffer.push(0x00),
        Value::Bool(true) => buffer.push(0x01),
        Value::Bool(false) => buffer.push(0x02),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                buffer.push(0x03);
                encode_signed_varint(i, buffer);
            } else if let Some(f) = n.as_f64() {
                buffer.push(0x04);
                buffer.extend_from_slice(&f.to_bits().to_le_bytes());
            }
        }
        Value::String(s) => {
            buffer.push(0x05);
            encode_varint(s.len() as u64, buffer);
            buffer.extend_from_slice(s.as_bytes());
        }
        _ => buffer.push(0x00),
    }
}

/// Decode columnar data from a buffer.
pub fn decode_columnar(input: &[u8]) -> Result<(usize, Vec<DecodedColumn>, usize), crate::FluxPackError> {
    let mut cursor = 0;

    let (col_count, consumed) = decode_varint(&input[cursor..])?;
    cursor += consumed;

    let mut columns = Vec::with_capacity(col_count as usize);

    for _ in 0..col_count {
        let (key_len, consumed) = decode_varint(&input[cursor..])?;
        cursor += consumed;

        let key_end = cursor + key_len as usize;
        if key_end > input.len() {
            return Err(crate::FluxPackError::BufferOverrun);
        }
        let key = std::str::from_utf8(&input[cursor..key_end])
            .map_err(|_| crate::FluxPackError::InvalidUtf8)?
            .to_string();
        cursor = key_end;

        if cursor >= input.len() {
            return Err(crate::FluxPackError::BufferOverrun);
        }
        let col_type = ColType::from_tag(input[cursor])
            .ok_or(crate::FluxPackError::InvalidValueType(input[cursor]))?;
        cursor += 1;

        let (row_count, consumed) = decode_varint(&input[cursor..])?;
        cursor += consumed;

        let (values, consumed) = decode_column_values_with_count(col_type, row_count as usize, &input[cursor..])?;
        cursor += consumed;

        columns.push(DecodedColumn { key, col_type, values });
    }

    let row_count = columns.first().map(|c| c.values.len()).unwrap_or(0);
    Ok((row_count, columns, cursor))
}

fn decode_column_values_with_count(col_type: ColType, count: usize, input: &[u8]) -> Result<(Vec<Value>, usize), crate::FluxPackError> {
    match col_type {
        ColType::Null => Ok((vec![Value::Null; count], 0)),
        ColType::Bool => {
            let bytes_needed = count.div_ceil(8);
            let mut values = Vec::with_capacity(count);
            for &byte in input.iter().take(bytes_needed) {
                for bit in 0..8 {
                    if values.len() >= count {
                        break;
                    }
                    values.push(Value::Bool((byte >> bit) & 1 == 1));
                }
            }
            Ok((values, bytes_needed))
        }
        ColType::Int => {
            let mut values = Vec::with_capacity(count);
            let mut cursor = 0;
            for _ in 0..count {
                let (val, consumed) = crate::decode_signed_varint(&input[cursor..])?;
                values.push(Value::Number(Number::from(val)));
                cursor += consumed;
            }
            Ok((values, cursor))
        }
        ColType::UInt => {
            let mut values = Vec::with_capacity(count);
            let mut cursor = 0;
            for _ in 0..count {
                let (val, consumed) = decode_varint(&input[cursor..])?;
                values.push(Value::Number(Number::from(val)));
                cursor += consumed;
            }
            Ok((values, cursor))
        }
        ColType::Float64 => {
            let bytes_needed = count * 8;
            let mut values = Vec::with_capacity(count);
            let mut cursor = 0;
            for _ in 0..count {
                let bits = u64::from_le_bytes([
                    input[cursor], input[cursor+1], input[cursor+2], input[cursor+3],
                    input[cursor+4], input[cursor+5], input[cursor+6], input[cursor+7],
                ]);
                let f = f64::from_bits(bits);
                if f.is_nan() {
                    values.push(Value::Null);
                } else {
                    values.push(Value::Number(Number::from_f64(f).unwrap_or(Number::from(0))));
                }
                cursor += 8;
            }
            let _ = bytes_needed;
            Ok((values, cursor))
        }
        ColType::Float32 => {
            let mut values = Vec::with_capacity(count);
            let mut cursor = 0;
            for _ in 0..count {
                let bits = u32::from_le_bytes([
                    input[cursor], input[cursor+1], input[cursor+2], input[cursor+3],
                ]);
                let f = f32::from_bits(bits);
                if f.is_nan() {
                    values.push(Value::Null);
                } else {
                    values.push(Value::Number(Number::from_f64(f as f64).unwrap_or(Number::from(0))));
                }
                cursor += 4;
            }
            Ok((values, cursor))
        }
        ColType::String => {
            let mut values = Vec::with_capacity(count);
            let mut cursor = 0;
            for _ in 0..count {
                let (len, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                if len == 0 {
                    values.push(Value::Null);
                } else {
                    let s = std::str::from_utf8(&input[cursor..cursor + len as usize])
                        .map_err(|_| crate::FluxPackError::InvalidUtf8)?
                        .to_string();
                    cursor += len as usize;
                    values.push(Value::String(s));
                }
            }
            Ok((values, cursor))
        }
        ColType::Mixed => {
            let mut values = Vec::with_capacity(count);
            let mut cursor = 0;
            for _ in 0..count {
                let (val, consumed) = decode_mixed_value(&input[cursor..])?;
                values.push(val);
                cursor += consumed;
            }
            Ok((values, cursor))
        }
    }
}

fn decode_mixed_value(input: &[u8]) -> Result<(Value, usize), crate::FluxPackError> {
    if input.is_empty() {
        return Err(crate::FluxPackError::BufferOverrun);
    }
    let tag = input[0];
    match tag {
        0x00 => Ok((Value::Null, 1)),
        0x01 => Ok((Value::Bool(true), 1)),
        0x02 => Ok((Value::Bool(false), 1)),
        0x03 => {
            let (val, consumed) = crate::decode_signed_varint(&input[1..])?;
            Ok((Value::Number(Number::from(val)), consumed + 1))
        }
        0x04 => {
            if input.len() < 9 {
                return Err(crate::FluxPackError::BufferOverrun);
            }
            let bits = u64::from_le_bytes([
                input[1], input[2], input[3], input[4],
                input[5], input[6], input[7], input[8],
            ]);
            let f = f64::from_bits(bits);
            match Number::from_f64(f) {
                Some(n) => Ok((Value::Number(n), 9)),
                None => Ok((Value::Null, 9)),
            }
        }
        0x05 => {
            let (len, consumed) = decode_varint(&input[1..])?;
            let start = 1 + consumed;
            let s = std::str::from_utf8(&input[start..start + len as usize])
                .map_err(|_| crate::FluxPackError::InvalidUtf8)?
                .to_string();
            Ok((Value::String(s), start + len as usize))
        }
        _ => Err(crate::FluxPackError::InvalidValueType(tag)),
    }
}

/// Reconstruct a JSON array from decoded columns.
pub fn reconstruct_array(row_count: usize, columns: Vec<DecodedColumn>) -> Value {
    let mut arr = Vec::with_capacity(row_count);

    for row_idx in 0..row_count {
        let mut obj = Map::with_capacity(columns.len());
        for col in &columns {
            let val = if row_idx < col.values.len() {
                col.values[row_idx].clone()
            } else {
                Value::Null
            };
            obj.insert(col.key.clone(), val);
        }
        arr.push(Value::Object(obj));
    }

    Value::Array(arr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_columnarize_simple() {
        let arr = vec![
            json!({"a": 1, "b": 2}),
            json!({"a": 3, "b": 4}),
            json!({"a": 5, "b": 6}),
        ];
        let columns = try_columnarize(&arr).unwrap();
        assert_eq!(columns.len(), 2);
    }

    #[test]
    fn test_columnar_roundtrip() {
        let arr = vec![
            json!({"x": 10, "y": 20.5, "name": "foo"}),
            json!({"x": 30, "y": 40.5, "name": "bar"}),
            json!({"x": 50, "y": 60.5, "name": "baz"}),
        ];

        let columns = try_columnarize(&arr).unwrap();
        let mut buf = Vec::new();
        encode_columnar(&arr, &columns, &mut buf);

        let (row_count, decoded_cols, consumed) = decode_columnar(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(row_count, 3);

        let result = reconstruct_array(row_count, decoded_cols);
        assert_eq!(result, Value::Array(arr));
    }

    #[test]
    fn test_columnar_bool_packing() {
        let arr = vec![
            json!({"flag": true}),
            json!({"flag": false}),
            json!({"flag": true}),
            json!({"flag": true}),
            json!({"flag": false}),
        ];

        let columns = try_columnarize(&arr).unwrap();
        assert_eq!(columns[0].1, ColType::Bool);

        let mut buf = Vec::new();
        encode_columnar(&arr, &columns, &mut buf);

        let (_, decoded_cols, _) = decode_columnar(&buf).unwrap();
        assert_eq!(decoded_cols[0].values.len(), 5);
        assert_eq!(decoded_cols[0].values[0], Value::Bool(true));
        assert_eq!(decoded_cols[0].values[1], Value::Bool(false));
    }

    #[test]
    fn test_columnar_rejects_heterogeneous() {
        let arr = vec![
            json!({"a": 1, "b": "hello"}),
            json!({"a": 2}),
        ];
        assert!(try_columnarize(&arr).is_none());
    }

    #[test]
    fn test_columnar_rejects_short_array() {
        let arr = vec![
            json!({"a": 1}),
            json!({"a": 2}),
        ];
        assert!(try_columnarize(&arr).is_none());
    }

    #[test]
    fn test_columnar_null_handling() {
        let arr = vec![
            json!({"val": 1}),
            json!({"val": null}),
            json!({"val": 3}),
        ];
        let columns = try_columnarize(&arr).unwrap();
        // Mixed int + null → Float64
        assert_eq!(columns[0].1, ColType::Float64);
    }
}
