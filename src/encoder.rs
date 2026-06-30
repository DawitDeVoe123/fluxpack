use serde_json::Value;
use crate::{SymbolTable, FluxPackError, encode_varint, encode_signed_varint, MAX_TOKENS};
use crate::columnar::{try_columnarize, encode_columnar};
use crate::inline::{encode_inline, INLINE_THRESHOLD};

/// The FluxPack encoder.
/// Takes a JSON value and emits a FluxPack binary stream.
pub struct Encoder {
    pub(crate) symbol_table: SymbolTable,
    output: Vec<u8>,
    /// Scratch buffer for columnar encoding (reused across calls)
    col_buf: Vec<u8>,
    debug_mode: bool,
    /// Minimum array length for columnar encoding (0 = disabled)
    columnar_threshold: usize,
    /// Previous schema fingerprint for DEF skip optimization
    previous_schema_fp: u64,
    /// Enable ZigZag encoding for signed integers (default: true)
    zigzag: bool,
    /// When true, skip DEF frame emission (for parallel batch encoding)
    skip_defs: bool,
    /// When true, use inline mode for small payloads
    inline_mode: bool,
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            symbol_table: SymbolTable::with_predefined(),
            output: Vec::with_capacity(4096),
            col_buf: Vec::with_capacity(2048),
            debug_mode: false,
            columnar_threshold: 3,
            previous_schema_fp: 0,
            zigzag: true,
            skip_defs: false,
            inline_mode: false,
        }
    }

    /// Enable or disable inline mode for small payloads.
    /// When enabled, payloads < 256 bytes use a compact format without symbol tables.
    pub fn set_inline_mode(&mut self, enabled: bool) {
        self.inline_mode = enabled;
    }

    /// Encode a JSON message into a FluxPack binary stream.
    /// DEF frames are emitted for new keys only. Subsequent messages
    /// with the same schema skip DEF frames entirely.
    #[inline]
    pub fn encode(&mut self, message: &Value) -> Result<&[u8], FluxPackError> {
        self.output.clear();

        let obj = message
            .as_object()
            .ok_or(FluxPackError::ExpectedObject)?;

        // Try inline mode for small payloads when enabled
        if self.inline_mode {
            let inline_result = encode_inline(obj);
            if !inline_result.is_empty() && inline_result.len() <= INLINE_THRESHOLD {
                self.output = inline_result;
                return Ok(&self.output);
            }
        }

        // Standard mode
        // First, intern all keys to compute the schema
        for (key, value) in obj {
            self.symbol_table.intern(key)?;
            self.intern_nested_keys(value)?;
        }

        let current_fp = self.symbol_table.schema_fingerprint();
        let schema_unchanged = self.previous_schema_fp != 0 && self.previous_schema_fp == current_fp;
        self.previous_schema_fp = current_fp;

        // Batch DEF emission: emit ALL pending DEFs in a single contiguous block
        if !schema_unchanged {
            self.emit_pending_defs()?;
        }

        // Encode the DATA frame
        self.output.push(0x02); // Frame type: DATA
        encode_varint(obj.len() as u64, &mut self.output);

        for (key, value) in obj {
            let token = self.symbol_table.intern(key)?;
            encode_varint(token as u64, &mut self.output);
            self.encode_value(value)?;
        }

        Ok(&self.output)
    }

    /// Encode multiple messages with shared schema.
    /// DEF frames are emitted once for the batch, then all DATA frames follow.
    /// This is the optimal path for streaming ML pipelines.
    pub fn encode_batch(&mut self, messages: &[Value]) -> Result<&[u8], FluxPackError> {
        self.output.clear();

        if messages.is_empty() {
            return Ok(&self.output);
        }

        // First pass: intern all keys from all messages
        for msg in messages {
            if let Some(obj) = msg.as_object() {
                for (key, value) in obj {
                    self.symbol_table.intern(key)?;
                    self.intern_nested_keys(value)?;
                }
            }
        }

        // Emit DEF frames once for the entire batch
        self.emit_pending_defs()?;

        // Second pass: encode all DATA frames
        for msg in messages {
            let obj = msg.as_object().ok_or(FluxPackError::ExpectedObject)?;
            self.output.push(0x02);
            encode_varint(obj.len() as u64, &mut self.output);
            for (key, value) in obj {
                let token = self.symbol_table.intern(key)?;
                encode_varint(token as u64, &mut self.output);
                self.encode_value(value)?;
            }
        }

        Ok(&self.output)
    }

    /// Encode a JSON value, using columnar encoding for arrays of objects
    /// when beneficial.
    pub fn encode_with_columnar(&mut self, message: &Value) -> Result<&[u8], FluxPackError> {
        self.output.clear();

        let obj = message
            .as_object()
            .ok_or(FluxPackError::ExpectedObject)?;

        // Intern all keys
        for (key, value) in obj {
            self.symbol_table.intern(key)?;
            self.intern_nested_keys(value)?;
        }

        let current_fp = self.symbol_table.schema_fingerprint();
        let schema_unchanged = self.previous_schema_fp != 0 && self.previous_schema_fp == current_fp;
        self.previous_schema_fp = current_fp;

        // Batch DEF emission
        if !schema_unchanged {
            self.emit_pending_defs()?;
        }

        // Encode DATA frame with columnar optimization
        self.output.push(0x02);
        encode_varint(obj.len() as u64, &mut self.output);

        for (key, value) in obj {
            let token = self.symbol_table.intern(key)?;
            encode_varint(token as u64, &mut self.output);

            // Try columnar encoding for arrays
            if let Value::Array(arr) = value {
                if arr.len() >= self.columnar_threshold {
                    if let Some(columns) = try_columnarize(arr) {
                        // Emit columnar value type marker
                        self.output.push(0x0D);
                        self.col_buf.clear();
                        encode_columnar(arr, &columns, &mut self.col_buf);
                        encode_varint(self.col_buf.len() as u64, &mut self.output);
                        self.output.extend_from_slice(&self.col_buf);
                        continue;
                    }
                }
            }

            self.encode_value(value)?;
        }

        Ok(&self.output)
    }

    /// Enable debug mode (emits full key names instead of tokens).
    pub fn set_debug_mode(&mut self, enabled: bool) {
        self.debug_mode = enabled;
    }

    /// Set the minimum array length for columnar encoding.
    /// Set to 0 to disable columnar encoding.
    pub fn set_columnar_threshold(&mut self, threshold: usize) {
        self.columnar_threshold = threshold;
    }

    /// Enable/disable ZigZag encoding for signed integers.
    pub fn set_zigzag(&mut self, enabled: bool) {
        self.zigzag = enabled;
    }

    /// Reset the encoder state (clears symbol table and buffers).
    pub fn reset(&mut self) {
        self.symbol_table.reset();
        self.output.clear();
        self.col_buf.clear();
        self.previous_schema_fp = 0;
    }

    /// Clone the symbol table from another encoder (for parallel encoding).
    pub fn clone_table_from(&mut self, other: &SymbolTable) {
        self.symbol_table = other.clone();
    }

    /// Enable/disable skipping DEF frame emission.
    /// Used for parallel batch encoding where DEFs are emitted separately.
    pub fn set_skip_defs(&mut self, skip: bool) {
        self.skip_defs = skip;
    }

    /// Encode only the DATA frame (no DEF frames).
    /// Used internally for parallel batch encoding.
    pub(crate) fn encode_data_only(&mut self, message: &Value) -> Result<&[u8], FluxPackError> {
        self.output.clear();

        let obj = message
            .as_object()
            .ok_or(FluxPackError::ExpectedObject)?;

        // Intern all keys
        for (key, value) in obj {
            self.symbol_table.intern(key)?;
            self.intern_nested_keys(value)?;
        }

        // Emit DATA frame only
        self.output.push(0x02); // Frame type: DATA
        encode_varint(obj.len() as u64, &mut self.output);

        for (key, value) in obj {
            let token = self.symbol_table.intern(key)?;
            encode_varint(token as u64, &mut self.output);
            self.encode_value(value)?;
        }

        Ok(&self.output)
    }

    /// Get the current symbol table size.
    pub fn symbol_table_size(&self) -> usize {
        self.symbol_table.size()
    }

    /// Emit all pending DEF frames in a single contiguous batch.
    /// This reduces frame overhead compared to interleaved DEF/DATA.
    fn emit_pending_defs(&mut self) -> Result<(), FluxPackError> {
        let pending: Vec<(u16, String)> = self.symbol_table.pending_defs()
            .into_iter()
            .map(|(t, k)| (t, k.to_string()))
            .collect();

        for (token, key) in pending {
            if token > MAX_TOKENS {
                return Err(FluxPackError::TableOverflow);
            }
            self.output.push(0x01); // DEF frame
            encode_varint(token as u64, &mut self.output);
            encode_varint(key.len() as u64, &mut self.output);
            self.output.extend_from_slice(key.as_bytes());
            self.symbol_table.mark_def_emitted(token);
        }

        Ok(())
    }

    fn intern_nested_keys(&mut self, value: &Value) -> Result<(), FluxPackError> {
        match value {
            Value::Object(obj) => {
                for (key, val) in obj {
                    self.symbol_table.intern(key)?;
                    self.intern_nested_keys(val)?;
                }
            }
            Value::Array(arr) => {
                for item in arr {
                    self.intern_nested_keys(item)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    #[inline(always)]
    fn encode_value(&mut self, value: &Value) -> Result<(), FluxPackError> {
        match value {
            Value::Null => self.output.push(0x00),
            Value::Bool(true) => self.output.push(0x01),
            Value::Bool(false) => self.output.push(0x02),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    self.output.push(0x03);
                    if self.zigzag {
                        encode_signed_varint(i, &mut self.output);
                    } else {
                        encode_varint(i as u64, &mut self.output);
                    }
                } else if let Some(u) = n.as_u64() {
                    self.output.push(0x04);
                    encode_varint(u, &mut self.output);
                } else if let Some(f) = n.as_f64() {
                    self.output.push(0x06);
                    self.output.extend_from_slice(&f.to_bits().to_le_bytes());
                } else {
                    return Err(FluxPackError::InvalidValueType(0));
                }
            }
            Value::String(s) => {
                self.output.push(0x05);
                let bytes = s.as_bytes();
                encode_varint(bytes.len() as u64, &mut self.output);
                self.output.extend_from_slice(bytes);
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
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}
