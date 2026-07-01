use ahash::AHashMap;
use crate::{FluxPackError, decode_varint, decode_signed_varint};
use crate::inline::INLINE_MAGIC;
use crate::symbol_table::COMMON_ML_KEYS;
use serde_json::{Value, Number, Map};
use std::sync::Arc;

/// Pre-computed key cache for hot-path decoding.
/// Stores Arc<str> so cloning is O(1) (just a reference count increment).
struct KeyCache {
    keys: Vec<Arc<str>>,
}

impl KeyCache {
    fn new() -> Self {
        let mut keys = Vec::with_capacity(256);
        for &key in COMMON_ML_KEYS {
            keys.push(Arc::from(key));
        }
        Self { keys }
    }

    #[inline]
    fn get_or_insert(&mut self, token: u16, key: &str) -> Arc<str> {
        let idx = token as usize;
        if idx < self.keys.len() && !self.keys[idx].is_empty() {
            return Arc::clone(&self.keys[idx]);
        }
        let arc = Arc::from(key);
        if idx >= self.keys.len() {
            self.keys.resize_with(idx + 1, || Arc::from(""));
        }
        self.keys[idx] = Arc::clone(&arc);
        arc
    }

    #[inline]
    fn get(&self, token: u16) -> Option<Arc<str>> {
        let idx = token as usize;
        if idx < self.keys.len() && !self.keys[idx].is_empty() {
            Some(Arc::clone(&self.keys[idx]))
        } else {
            None
        }
    }
}

/// Fast FluxPack decoder optimized for ML workloads.
///
/// Key optimizations over standard Decoder:
/// 1. Arc<str> key cache — cloning is O(1) instead of O(n)
/// 2. Pre-computed common ML keys — no allocation for epoch, loss, accuracy, etc.
/// 3. Inlined hot paths — eliminates function call overhead
pub struct FastDecoder {
    key_cache: KeyCache,
    key_to_id: AHashMap<Arc<str>, u16>,
    next_token: u16,
}

impl FastDecoder {
    pub fn new() -> Self {
        let key_cache = KeyCache::new();
        let mut key_to_id = AHashMap::with_capacity(64);
        let mut next_token: u16 = 1;

        for &key in COMMON_ML_KEYS {
            let arc = Arc::from(key);
            key_to_id.insert(Arc::clone(&arc), next_token);
            next_token += 1;
        }

        Self {
            key_cache,
            key_to_id,
            next_token,
        }
    }

    #[inline]
    fn store_def(&mut self, token: u16, key: &str) -> Result<(), FluxPackError> {
        if token > crate::MAX_TOKENS {
            return Err(FluxPackError::TableOverflow);
        }
        let arc = self.key_cache.get_or_insert(token, key);
        self.key_to_id.insert(arc, token);
        if token >= self.next_token {
            self.next_token = token + 1;
        }
        Ok(())
    }

    #[inline]
    fn resolve(&self, token: u16) -> Option<Arc<str>> {
        self.key_cache.get(token)
    }

    /// Decode a FluxPack stream into a JSON value.
    pub fn decode(&mut self, input: &[u8]) -> Result<Value, FluxPackError> {
        if !input.is_empty() && input[0] == INLINE_MAGIC {
            let (obj, _) = crate::inline::decode_inline(input)
                .map_err(|e| FluxPackError::ColumnarError(e))?;
            return Ok(Value::Object(obj));
        }

        let mut cursor = 0;
        let mut result = None;

        while cursor < input.len() {
            let frame_type = input[cursor];
            cursor += 1;

            match frame_type {
                0x01 => {
                    let (token, consumed) = decode_varint(&input[cursor..])?;
                    cursor += consumed;
                    let (key_len, consumed) = decode_varint(&input[cursor..])?;
                    cursor += consumed;
                    let key_end = cursor + key_len as usize;
                    if key_end > input.len() {
                        return Err(FluxPackError::BufferOverrun);
                    }
                    let key = std::str::from_utf8(&input[cursor..key_end])
                        .map_err(|_| FluxPackError::InvalidUtf8)?;
                    cursor = key_end;
                    self.store_def(token as u16, key)?;
                }
                0x02 => {
                    result = Some(self.decode_data_frame(&input[cursor..])?);
                    break;
                }
                0x0D => {
                    result = Some(self.decode_columnar_frame(&input[cursor..])?);
                    break;
                }
                0xFF => break,
                _ => return Err(FluxPackError::InvalidValueType(frame_type)),
            }
        }

        result.ok_or(FluxPackError::MalformedFrame)
    }

    #[inline]
    fn decode_data_frame(&mut self, input: &[u8]) -> Result<Value, FluxPackError> {
        let (field_count, mut cursor) = decode_varint(input)?;
        let mut obj = Map::with_capacity(field_count as usize);

        for _ in 0..field_count {
            let (token, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;

            let key_arc = self.resolve(token as u16)
                .ok_or(FluxPackError::UnknownToken(token as u16))?;
            // Arc<str> -> String is a single allocation, no copy of Arc internals
            let key: String = key_arc.to_string();

            let (value, consumed) = self.decode_value(&input[cursor..])?;
            cursor += consumed;
            obj.insert(key, value);
        }

        Ok(Value::Object(obj))
    }

    fn decode_columnar_frame(&mut self, input: &[u8]) -> Result<Value, FluxPackError> {
        let (row_count, columns, _) = crate::columnar::decode_columnar(input)?;
        Ok(crate::columnar::reconstruct_array(row_count, columns))
    }

    #[inline(always)]
    fn decode_value(&mut self, input: &[u8]) -> Result<(Value, usize), FluxPackError> {
        if input.is_empty() {
            return Err(FluxPackError::BufferOverrun);
        }

        let value_type = input[0];
        let mut cursor = 1;

        match value_type {
            0x00 => Ok((Value::Null, cursor)),
            0x01 => Ok((Value::Bool(true), cursor)),
            0x02 => Ok((Value::Bool(false), cursor)),
            0x03 => {
                let (val, consumed) = decode_signed_varint(&input[cursor..])?;
                cursor += consumed;
                Ok((Value::Number(Number::from(val)), cursor))
            }
            0x04 => {
                let (val, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                Ok((Value::Number(Number::from(val)), cursor))
            }
            0x05 => {
                let (len, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let end = cursor + len as usize;
                if end > input.len() {
                    return Err(FluxPackError::BufferOverrun);
                }
                let s = std::str::from_utf8(&input[cursor..end])
                    .map_err(|_| FluxPackError::InvalidUtf8)?
                    .to_string();
                cursor = end;
                Ok((Value::String(s), cursor))
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
                let f = f64::from_bits(bits);
                match Number::from_f64(f) {
                    Some(n) => Ok((Value::Number(n), cursor)),
                    None => Ok((Value::Null, cursor)),
                }
            }
            0x07 => {
                if cursor + 4 > input.len() {
                    return Err(FluxPackError::BufferOverrun);
                }
                let bits = u32::from_le_bytes([
                    input[cursor], input[cursor+1], input[cursor+2], input[cursor+3],
                ]);
                cursor += 4;
                let f = f32::from_bits(bits);
                match Number::from_f64(f as f64) {
                    Some(n) => Ok((Value::Number(n), cursor)),
                    None => Ok((Value::Null, cursor)),
                }
            }
            0x08 => {
                let (len, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let end = cursor + len as usize;
                if end > input.len() {
                    return Err(FluxPackError::BufferOverrun);
                }
                let bytes = &input[cursor..end];
                cursor = end;
                let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
                Ok((Value::String(hex), cursor))
            }
            0x09 => {
                let (len, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let mut arr = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    let (val, consumed) = self.decode_value(&input[cursor..])?;
                    cursor += consumed;
                    arr.push(val);
                }
                Ok((Value::Array(arr), cursor))
            }
            0x0A => {
                let (len, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let mut obj = Map::with_capacity(len as usize);
                for _ in 0..len {
                    let (token, consumed) = decode_varint(&input[cursor..])?;
                    cursor += consumed;
                    let key_arc = self.resolve(token as u16)
                        .ok_or(FluxPackError::UnknownToken(token as u16))?;
                    let key: String = key_arc.to_string();
                    let (val, consumed) = self.decode_value(&input[cursor..])?;
                    cursor += consumed;
                    obj.insert(key, val);
                }
                Ok((Value::Object(obj), cursor))
            }
            0x0B => {
                let (token, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let key_arc = self.resolve(token as u16)
                    .ok_or(FluxPackError::UnknownToken(token as u16))?;
                Ok((Value::String(key_arc.to_string()), cursor))
            }
            0x0C => {
                let (ts, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let secs = ts / 1000;
                let millis = ts % 1000;
                let ts_str = format!("{}.{:03}Z", secs, millis);
                Ok((Value::String(ts_str), cursor))
            }
            0x0D => {
                let (data_len, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let end = cursor + data_len as usize;
                let (row_count, columns, _) = crate::columnar::decode_columnar(&input[cursor..end])?;
                cursor = end;
                let arr = crate::columnar::reconstruct_array(row_count, columns);
                Ok((arr, cursor))
            }
            _ => Err(FluxPackError::InvalidValueType(value_type)),
        }
    }

    pub fn reset(&mut self) {
        self.key_cache = KeyCache::new();
        self.key_to_id.clear();
        self.next_token = 1;
        for &key in COMMON_ML_KEYS {
            let arc = Arc::from(key);
            self.key_to_id.insert(arc, self.next_token);
            self.next_token += 1;
        }
    }
}

impl Default for FastDecoder {
    fn default() -> Self {
        Self::new()
    }
}
