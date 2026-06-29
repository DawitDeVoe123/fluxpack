use serde_json::{Value, Number, Map};
use crate::{SymbolTable, FluxPackError, decode_varint, decode_signed_varint};
use crate::columnar::{decode_columnar, reconstruct_array};

/// The FluxPack decoder.
/// Takes a FluxPack binary stream and reconstructs the JSON.
pub struct Decoder {
    symbol_table: SymbolTable,
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            symbol_table: SymbolTable::new(),
        }
    }

    /// Decode a FluxPack stream into a JSON value.
    /// Handles DEF frames, DATA frames, and columnar frames.
    pub fn decode(&mut self, input: &[u8]) -> Result<Value, FluxPackError> {
        let mut cursor = 0;
        let mut result = None;

        while cursor < input.len() {
            let frame_type = input[cursor];
            cursor += 1;

            match frame_type {
                0x01 => {
                    // DEF frame — build the symbol table using EXACT token IDs from the wire.
                    // CRITICAL: We must use store_def() instead of intern() to preserve
                    // the encoder's token assignments.
                    let (token, consumed) = decode_varint(&input[cursor..])?;
                    cursor += consumed;

                    let (key_len, consumed) = decode_varint(&input[cursor..])?;
                    cursor += consumed;

                    let key = std::str::from_utf8(&input[cursor..cursor + key_len as usize])
                        .map_err(|_| FluxPackError::InvalidUtf8)?;
                    cursor += key_len as usize;

                    self.symbol_table.store_def(token as u16, key)?;
                }
                0x02 => {
                    // DATA frame
                    result = Some(self.decode_data_frame(&input[cursor..])?);
                    break;
                }
                0x0D => {
                    // Columnar DATA frame
                    result = Some(self.decode_columnar_frame(&input[cursor..])?);
                    break;
                }
                0xFF => {
                    // End of stream
                    break;
                }
                _ => {
                    return Err(FluxPackError::InvalidValueType(frame_type));
                }
            }
        }

        result.ok_or(FluxPackError::MalformedFrame)
    }

    /// Decode multiple messages from a stream.
    pub fn decode_all(&mut self, input: &[u8]) -> Result<Vec<Value>, FluxPackError> {
        let mut results = Vec::new();
        let mut cursor = 0;

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
                        .map_err(|_| FluxPackError::InvalidUtf8)?
                        .to_string();
                    cursor += key_len as usize;
                    self.symbol_table.store_def(token as u16, &key)?;
                }
                0x02 => {
                    let (obj, consumed) = self.decode_data_frame_at(&input[cursor..])?;
                    cursor += consumed;
                    results.push(obj);
                }
                0x0D => {
                    let (val, consumed) = self.decode_columnar_frame_at(&input[cursor..])?;
                    cursor += consumed;
                    results.push(val);
                }
                0xFF => break,
                _ => return Err(FluxPackError::InvalidValueType(frame_type)),
            }
        }

        Ok(results)
    }

    fn decode_data_frame(&mut self, input: &[u8]) -> Result<Value, FluxPackError> {
        let (obj, _) = self.decode_data_frame_at(input)?;
        Ok(obj)
    }

    /// Decode a DATA frame, returning the value and bytes consumed.
    #[inline]
    fn decode_data_frame_at(&mut self, input: &[u8]) -> Result<(Value, usize), FluxPackError> {
        let (field_count, mut cursor) = decode_varint(input)?;

        let mut obj = Map::with_capacity(field_count as usize);

        for _ in 0..field_count {
            let (token, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;

            let token_u16 = token as u16;
            let key = self.symbol_table.resolve(token_u16)
                .ok_or(FluxPackError::UnknownToken(token_u16))?
                .to_string();

            let (value, consumed) = self.decode_value(&input[cursor..])?;
            cursor += consumed;

            obj.insert(key, value);
        }

        Ok((Value::Object(obj), cursor))
    }

    fn decode_columnar_frame(&mut self, input: &[u8]) -> Result<Value, FluxPackError> {
        let (val, _) = self.decode_columnar_frame_at(input)?;
        Ok(val)
    }

    fn decode_columnar_frame_at(&mut self, input: &[u8]) -> Result<(Value, usize), FluxPackError> {
        let (row_count, columns, consumed) = decode_columnar(input)?;
        let arr = reconstruct_array(row_count, columns);
        Ok((arr, consumed))
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
                // Signed integer (ZigZag encoded)
                let (val, consumed) = decode_signed_varint(&input[cursor..])?;
                cursor += consumed;
                Ok((Value::Number(Number::from(val)), cursor))
            }
            0x04 => {
                // Unsigned integer
                let (val, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                Ok((Value::Number(Number::from(val)), cursor))
            }
            0x05 => {
                // String
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
                // Float64 (little-endian for consistency with columnar)
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
                // Float32 (little-endian)
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
                // Bytes — decode as base64-like string for JSON compatibility
                let (len, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let end = cursor + len as usize;
                if end > input.len() {
                    return Err(FluxPackError::BufferOverrun);
                }
                let bytes = &input[cursor..end];
                cursor = end;
                // Encode as a hex string for JSON compatibility
                let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
                Ok((Value::String(hex), cursor))
            }
            0x09 => {
                // Array
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
                // Object
                let (len, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let mut obj = Map::with_capacity(len as usize);
                for _ in 0..len {
                    let (token, consumed) = decode_varint(&input[cursor..])?;
                    cursor += consumed;
                    let token_u16 = token as u16;
                    let key = self.symbol_table.resolve(token_u16)
                        .ok_or(FluxPackError::UnknownToken(token_u16))?
                        .to_string();
                    let (val, consumed) = self.decode_value(&input[cursor..])?;
                    cursor += consumed;
                    obj.insert(key, val);
                }
                Ok((Value::Object(obj), cursor))
            }
            0x0B => {
                // Interned value — resolve from symbol table
                let (token, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let key = self.symbol_table.resolve(token as u16)
                    .ok_or(FluxPackError::UnknownToken(token as u16))?
                    .to_string();
                Ok((Value::String(key), cursor))
            }
            0x0C => {
                // Timestamp — decode as ISO 8601 string
                let (ts, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                // Convert millisecond timestamp to string
                let secs = ts / 1000;
                let millis = ts % 1000;
                let ts_str = format!("{}.{:03}Z", secs, millis);
                Ok((Value::String(ts_str), cursor))
            }
            0x0D => {
                // Columnar data embedded as a value
                let (data_len, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let end = cursor + data_len as usize;
                let (row_count, columns, _) = decode_columnar(&input[cursor..end])?;
                cursor = end;
                let arr = reconstruct_array(row_count, columns);
                Ok((arr, cursor))
            }
            _ => Err(FluxPackError::InvalidValueType(value_type)),
        }
    }

    /// Reset the decoder state (clears symbol table).
    pub fn reset(&mut self) {
        self.symbol_table.reset();
    }

    /// Get the current symbol table size.
    pub fn symbol_table_size(&self) -> usize {
        self.symbol_table.size()
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}
