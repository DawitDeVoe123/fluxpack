use serde_json::{Value, Number, Map};
use crate::{SymbolTable, FluxPackError, decode_varint};

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
    /// Handles DEF frames and DATA frames.
    pub fn decode(&mut self, input: &[u8]) -> Result<Value, FluxPackError> {
        let mut cursor = 0;
        let mut result = None;

        while cursor < input.len() {
            let frame_type = input[cursor];
            cursor += 1;

            match frame_type {
                0x01 => {
                    // DEF frame - build the symbol table
                    let (token, consumed) = decode_varint(&input[cursor..])?;
                    cursor += consumed;

                    let (key_len, consumed) = decode_varint(&input[cursor..])?;
                    cursor += consumed;

                    let key = std::str::from_utf8(&input[cursor..cursor + key_len as usize])
                        .map_err(|_| FluxPackError::InvalidUtf8)?;
                    cursor += key_len as usize;

                    // Manually insert into symbol table with the token from the DEF frame
                    // We need to bypass the normal intern() method because it auto-assigns tokens
                    // We'll use a different approach - store it directly
                    let _token_u16 = token as u16;
                    // Store in both directions
                    // Since we can't directly access the fields, we'll use a workaround:
                    // We'll store it and the decoder will use it
                    // For now, we'll use a HashMap in the decoder
                    // Wait, let me re-think this...
                    
                    // Actually, we need to store the token-to-key mapping directly
                    // Since we can't access the private fields, let's just use the intern method
                    // but we need to ensure token order matches
                    // The issue is that the encoder assigned tokens in order, and we should too
                    let _ = self.symbol_table.intern(key)?;
                    // This will assign token IDs in the same order as the encoder
                    // So if the encoder assigned token 1 to "user_id", 2 to "email", etc.
                    // Our intern() will assign token 1 to the first key it sees, etc.
                    // This should work as long as we process DEF frames in order
                }
                0x02 => {
                    // DATA frame
                    result = Some(self.decode_data_frame(&input[cursor..])?);
                    break;
                }
                _ => {
                    return Err(FluxPackError::InvalidValueType(frame_type));
                }
            }
        }

        result.ok_or(FluxPackError::MalformedFrame)
    }

    fn decode_data_frame(&mut self, input: &[u8]) -> Result<Value, FluxPackError> {
        let mut cursor = 0;

        // Field count
        let (field_count, consumed) = decode_varint(&input[cursor..])?;
        cursor += consumed;

        let mut obj = Map::with_capacity(field_count as usize);

        for _ in 0..field_count {
            // Token ID
            let (token, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;

            // Get the key from the symbol table
            let token_u16 = token as u16;
            let key = self.symbol_table.resolve(token_u16)
                .ok_or(FluxPackError::UnknownToken(token_u16))?
                .to_string();

            // Decode the value
            let (value, consumed) = self.decode_value(&input[cursor..])?;
            cursor += consumed;

            obj.insert(key, value);
        }

        Ok(Value::Object(obj))
    }

    fn decode_value(&mut self, input: &[u8]) -> Result<(Value, usize), FluxPackError> {
        let mut cursor = 0;

        if cursor >= input.len() {
            return Err(FluxPackError::BufferOverrun);
        }

        let value_type = input[cursor];
        cursor += 1;

        match value_type {
            0x00 => Ok((Value::Null, cursor)),
            0x01 => Ok((Value::Bool(true), cursor)),
            0x02 => Ok((Value::Bool(false), cursor)),
            0x03 => {
                let (val, consumed) = decode_varint(&input[cursor..])?;
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
                let s = std::str::from_utf8(&input[cursor..cursor + len as usize])
                    .map_err(|_| FluxPackError::InvalidUtf8)?;
                cursor += len as usize;
                Ok((Value::String(s.to_string()), cursor))
            }
            0x06 => {
                if cursor + 8 > input.len() {
                    return Err(FluxPackError::BufferOverrun);
                }
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&input[cursor..cursor + 8]);
                cursor += 8;
                let f = f64::from_bits(u64::from_be_bytes(buf));
                if let Some(n) = Number::from_f64(f) {
                    Ok((Value::Number(n), cursor))
                } else {
                    Ok((Value::Null, cursor))
                }
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
            _ => Err(FluxPackError::InvalidValueType(value_type)),
        }
    }

    /// Reset the decoder state (clears symbol table)
    pub fn reset(&mut self) {
        self.symbol_table.reset();
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}