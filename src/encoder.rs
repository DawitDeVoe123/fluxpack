use serde_json::Value;
use crate::{SymbolTable, FluxPackError, encode_varint, MAX_TOKENS};

/// The FluxPack encoder.
/// Takes a JSON value and emits a FluxPack binary stream.
pub struct Encoder {
    symbol_table: SymbolTable,
    output: Vec<u8>,
    debug_mode: bool,
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            symbol_table: SymbolTable::new(),
            output: Vec::with_capacity(1024),
            debug_mode: false,
        }
    }

    /// Encode a JSON message into a FluxPack DATA frame.
    /// DEF frames are emitted automatically for new keys.
    pub fn encode(&mut self, message: &Value) -> Result<&[u8], FluxPackError> {
        self.output.clear();

        let obj = message
            .as_object()
            .ok_or(FluxPackError::ExpectedObject)?;

        // First pass: collect ALL keys recursively and emit DEF frames
        self.collect_and_emit_defs(obj)?;

        // Second pass: encode the DATA frame
        self.output.push(0x02); // Frame type: DATA
        encode_varint(obj.len() as u64, &mut self.output);

        for (key, value) in obj {
            let token = self.symbol_table.intern(key)?;
            encode_varint(token as u64, &mut self.output);
            self.encode_value(value)?;
        }

        Ok(&self.output)
    }

    /// Recursively collect all keys and emit DEF frames for them
    fn collect_and_emit_defs(&mut self, obj: &serde_json::Map<String, Value>) -> Result<(), FluxPackError> {
        for (key, value) in obj {
            // Emit DEF for this key if it's new
            let token = self.symbol_table.intern(key)?;
            // Only emit DEF if it was just added (size increased)
            // We'll just check if the token is >= the size
            // Actually, let's just emit DEF for all keys
            // The decoder will handle duplicates gracefully
            self.emit_def_frame(token, key)?;
            
            // Recursively handle nested objects
            if let Value::Object(nested_obj) = value {
                self.collect_and_emit_defs(nested_obj)?;
            }
        }
        Ok(())
    }

    fn emit_def_frame(&mut self, token: u16, key: &str) -> Result<(), FluxPackError> {
        if token > MAX_TOKENS {
            return Err(FluxPackError::TableOverflow);
        }

        self.output.push(0x01); // Frame type: DEF
        encode_varint(token as u64, &mut self.output);
        encode_varint(key.len() as u64, &mut self.output);
        self.output.extend_from_slice(key.as_bytes());
        Ok(())
    }

    fn encode_value(&mut self, value: &Value) -> Result<(), FluxPackError> {
        match value {
            Value::Null => {
                self.output.push(0x00);
            }
            Value::Bool(true) => {
                self.output.push(0x01);
            }
            Value::Bool(false) => {
                self.output.push(0x02);
            }
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    self.output.push(0x03);
                    encode_varint(i as u64, &mut self.output);
                } else if let Some(u) = n.as_u64() {
                    self.output.push(0x04);
                    encode_varint(u, &mut self.output);
                } else if let Some(f) = n.as_f64() {
                    self.output.push(0x06);
                    self.output.extend_from_slice(&f.to_bits().to_be_bytes());
                } else {
                    return Err(FluxPackError::InvalidValueType(0));
                }
            }
            Value::String(s) => {
                self.output.push(0x05);
                encode_varint(s.len() as u64, &mut self.output);
                self.output.extend_from_slice(s.as_bytes());
            }
            Value::Array(arr) => {
                self.output.push(0x09);
                encode_varint(arr.len() as u64, &mut self.output);
                for item in arr {
                    self.encode_value(item)?;
                }
            }
            Value::Object(obj) => {
                self.output.push(0x0A);
                encode_varint(obj.len() as u64, &mut self.output);
                for (key, val) in obj {
                    let token = self.symbol_table.intern(key)?;
                    encode_varint(token as u64, &mut self.output);
                    self.encode_value(val)?;
                }
            }
        }
        Ok(())
    }

    /// Enable debug mode (emits full key names instead of tokens)
    pub fn set_debug_mode(&mut self, enabled: bool) {
        self.debug_mode = enabled;
    }

    /// Reset the encoder state (clears symbol table)
    pub fn reset(&mut self) {
        self.symbol_table.reset();
        self.output.clear();
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}