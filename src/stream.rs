use serde_json::Value;
use crate::{SymbolTable, FluxPackError, encode_varint, decode_varint, MAX_TOKENS};
use crate::columnar::{try_columnarize, encode_columnar, decode_columnar, reconstruct_array};

/// Frame types for the streaming protocol.
#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    /// Definition frame: token_id → key mapping
    Def { token: u16, key: String },
    /// Data frame: a single encoded JSON object
    Data(Value),
    /// Columnar data frame: array of objects encoded column-major
    Columnar { row_count: usize, data: Vec<u8> },
    /// End of stream marker
    Eos,
}

/// Streaming encoder: emits frames one at a time without buffering the entire message.
pub struct StreamWriter {
    symbol_table: SymbolTable,
    buffer: Vec<u8>,
    previous_schema_fingerprint: u64,
}

impl StreamWriter {
    pub fn new() -> Self {
        Self {
            symbol_table: SymbolTable::with_predefined(),
            buffer: Vec::with_capacity(4096),
            previous_schema_fingerprint: 0,
        }
    }

    /// Write a DEF frame for a key. Skips if already emitted.
    #[inline]
    pub fn write_def(&mut self, key: &str) -> Result<u16, FluxPackError> {
        let token = self.symbol_table.intern(key)?;
        if !self.symbol_table.def_emitted(token) {
            self.emit_def_frame(token, key)?;
            self.symbol_table.mark_def_emitted(token);
        }
        Ok(token)
    }

    /// Write a DATA frame for a JSON object.
    /// Automatically emits DEF frames for any new keys.
    pub fn write_data(&mut self, message: &Value) -> Result<(), FluxPackError> {
        let obj = message
            .as_object()
            .ok_or(FluxPackError::ExpectedObject)?;

        // Collect and emit any new DEFs
        for (key, value) in obj {
            self.write_def(key)?;
            self.collect_nested_defs(value)?;
        }

        // Encode the DATA frame
        self.buffer.push(0x02); // Frame type: DATA
        encode_varint(obj.len() as u64, &mut self.buffer);

        for (key, value) in obj {
            let token = self.symbol_table.intern(key)?;
            encode_varint(token as u64, &mut self.buffer);
            self.encode_value(value)?;
        }

        Ok(())
    }

    /// Write multiple messages in a batch.
    /// Emits all DEF frames first, then all DATA frames.
    pub fn write_batch(&mut self, messages: &[Value]) -> Result<(), FluxPackError> {
        // First: ensure all keys are interned and DEFs emitted
        for msg in messages {
            let obj = msg.as_object().ok_or(FluxPackError::ExpectedObject)?;
            for (key, value) in obj {
                self.write_def(key)?;
                self.collect_nested_defs(value)?;
            }
        }

        // Then: encode all DATA frames
        for msg in messages {
            let obj = msg.as_object().ok_or(FluxPackError::ExpectedObject)?;
            self.buffer.push(0x02);
            encode_varint(obj.len() as u64, &mut self.buffer);
            for (key, value) in obj {
                let token = self.symbol_table.intern(key)?;
                encode_varint(token as u64, &mut self.buffer);
                self.encode_value(value)?;
            }
        }

        Ok(())
    }

    /// Write an array as a columnar DATA frame if it's tabular.
    /// Returns true if columnar encoding was used.
    pub fn write_columnar_data(&mut self, key: &str, arr: &[Value]) -> Result<bool, FluxPackError> {
        if let Some(columns) = try_columnarize(arr) {
            // Emit DEF for the key itself
            let token = self.write_def(key)?;

            // Emit columnar frame
            self.buffer.push(0x0D); // Columnar frame type
            let mut col_buf = Vec::new();
            encode_columnar(arr, &columns, &mut col_buf);

            // Encode: token + columnar_data_len + columnar_data
            encode_varint(token as u64, &mut self.buffer);
            encode_varint(col_buf.len() as u64, &mut self.buffer);
            self.buffer.extend_from_slice(&col_buf);

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get the accumulated buffer.
    #[inline]
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Consume the writer and return the buffer.
    #[inline]
    pub fn into_buffer(self) -> Vec<u8> {
        self.buffer
    }

    /// Get the current schema fingerprint.
    pub fn schema_fingerprint(&mut self) -> u64 {
        self.symbol_table.schema_fingerprint()
    }

    /// Check if schema matches the previous message.
    pub fn schema_unchanged(&mut self) -> bool {
        let fp = self.symbol_table.schema_fingerprint();
        let unchanged = self.previous_schema_fingerprint != 0 && self.previous_schema_fingerprint == fp;
        self.previous_schema_fingerprint = fp;
        unchanged
    }

    /// Reset the writer state.
    pub fn reset(&mut self) {
        self.symbol_table.reset();
        self.buffer.clear();
        self.previous_schema_fingerprint = 0;
    }

    fn emit_def_frame(&mut self, token: u16, key: &str) -> Result<(), FluxPackError> {
        if token > MAX_TOKENS {
            return Err(FluxPackError::TableOverflow);
        }
        self.buffer.push(0x01); // DEF frame
        encode_varint(token as u64, &mut self.buffer);
        encode_varint(key.len() as u64, &mut self.buffer);
        self.buffer.extend_from_slice(key.as_bytes());
        Ok(())
    }

    fn collect_nested_defs(&mut self, value: &Value) -> Result<(), FluxPackError> {
        if let Value::Object(obj) = value {
            for (key, val) in obj {
                self.write_def(key)?;
                self.collect_nested_defs(val)?;
            }
        } else if let Value::Array(arr) = value {
            for item in arr {
                self.collect_nested_defs(item)?;
            }
        }
        Ok(())
    }

    #[inline]
    fn encode_value(&mut self, value: &Value) -> Result<(), FluxPackError> {
        match value {
            Value::Null => self.buffer.push(0x00),
            Value::Bool(true) => self.buffer.push(0x01),
            Value::Bool(false) => self.buffer.push(0x02),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    self.buffer.push(0x03);
                    crate::encode_signed_varint(i, &mut self.buffer);
                } else if let Some(f) = n.as_f64() {
                    self.buffer.push(0x06);
                    self.buffer.extend_from_slice(&f.to_bits().to_le_bytes());
                }
            }
            Value::String(s) => {
                self.buffer.push(0x05);
                encode_varint(s.len() as u64, &mut self.buffer);
                self.buffer.extend_from_slice(s.as_bytes());
            }
            Value::Array(arr) => {
                self.buffer.push(0x09);
                encode_varint(arr.len() as u64, &mut self.buffer);
                for item in arr {
                    self.encode_value(item)?;
                }
            }
            Value::Object(obj) => {
                self.buffer.push(0x0A);
                encode_varint(obj.len() as u64, &mut self.buffer);
                for (key, val) in obj {
                    let token = self.symbol_table.intern(key)?;
                    encode_varint(token as u64, &mut self.buffer);
                    self.encode_value(val)?;
                }
            }
        }
        Ok(())
    }
}

impl Default for StreamWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming decoder: parses frames one at a time.
pub struct StreamReader {
    symbol_table: SymbolTable,
    cursor: usize,
}

impl StreamReader {
    pub fn new() -> Self {
        Self {
            symbol_table: SymbolTable::new(),
            cursor: 0,
        }
    }

    /// Parse the next frame from the input.
    pub fn read_frame(&mut self, input: &[u8]) -> Result<Option<Frame>, FluxPackError> {
        if self.cursor >= input.len() {
            return Ok(None);
        }

        let frame_type = input[self.cursor];
        self.cursor += 1;

        match frame_type {
            0x01 => {
                // DEF frame
                let (token, consumed) = decode_varint(&input[self.cursor..])?;
                self.cursor += consumed;
                let (key_len, consumed) = decode_varint(&input[self.cursor..])?;
                self.cursor += consumed;
                let key = std::str::from_utf8(&input[self.cursor..self.cursor + key_len as usize])
                    .map_err(|_| FluxPackError::InvalidUtf8)?
                    .to_string();
                self.cursor += key_len as usize;

                self.symbol_table.store_def(token as u16, &key)?;

                Ok(Some(Frame::Def { token: token as u16, key }))
            }
            0x02 => {
                // DATA frame
                let obj = self.decode_data_frame(input)?;
                Ok(Some(Frame::Data(obj)))
            }
            0x0D => {
                // Columnar frame
                let (token, consumed) = decode_varint(&input[self.cursor..])?;
                self.cursor += consumed;
                let (data_len, consumed) = decode_varint(&input[self.cursor..])?;
                self.cursor += consumed;
                let col_data = input[self.cursor..self.cursor + data_len as usize].to_vec();
                self.cursor += data_len as usize;

                let _ = token; // Token is the key that holds this columnar data

                Ok(Some(Frame::Columnar {
                    row_count: 0, // Will be determined when decoding
                    data: col_data,
                }))
            }
            0xFF => Ok(Some(Frame::Eos)),
            _ => Err(FluxPackError::InvalidValueType(frame_type)),
        }
    }

    /// Read all frames from input.
    pub fn read_all(&mut self, input: &[u8]) -> Result<Vec<Frame>, FluxPackError> {
        let mut frames = Vec::new();
        while let Some(frame) = self.read_frame(input)? {
            if frame == Frame::Eos {
                break;
            }
            frames.push(frame);
        }
        Ok(frames)
    }

    /// Iterator-based reading.
    pub fn frames<'a>(&'a mut self, input: &'a [u8]) -> FrameIterator<'a> {
        FrameIterator {
            reader: self,
            input,
            done: false,
        }
    }

    /// Reset the reader state.
    pub fn reset(&mut self) {
        self.symbol_table.reset();
        self.cursor = 0;
    }

    /// Get a reference to the symbol table.
    pub fn symbol_table(&self) -> &SymbolTable {
        &self.symbol_table
    }

    fn decode_data_frame(&mut self, input: &[u8]) -> Result<Value, FluxPackError> {
        let (field_count, mut cursor) = decode_varint(&input[self.cursor..])?;
        cursor += self.cursor;

        let mut obj = serde_json::Map::with_capacity(field_count as usize);

        for _ in 0..field_count {
            let (token, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;

            let key = self.symbol_table.resolve(token as u16)
                .ok_or(FluxPackError::UnknownToken(token as u16))?
                .to_string();

            let (value, consumed) = decode_value(&input[cursor..], &self.symbol_table)?;
            cursor += consumed;

            obj.insert(key, value);
        }

        self.cursor = cursor;
        Ok(Value::Object(obj))
    }
}

impl Default for StreamReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over frames in a stream.
pub struct FrameIterator<'a> {
    reader: &'a mut StreamReader,
    input: &'a [u8],
    done: bool,
}

impl<'a> Iterator for FrameIterator<'a> {
    type Item = Frame;

    fn next(&mut self) -> Option<Frame> {
        if self.done {
            return None;
        }
        match self.reader.read_frame(self.input) {
            Ok(Some(frame)) => {
                if frame == Frame::Eos {
                    self.done = true;
                    None
                } else {
                    Some(frame)
                }
            }
            Ok(None) => {
                self.done = true;
                None
            }
            Err(_) => {
                self.done = true;
                None
            }
        }
    }
}

/// Decode a value from bytes with a symbol table reference.
fn decode_value(input: &[u8], table: &SymbolTable) -> Result<(Value, usize), FluxPackError> {
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
            let (val, consumed) = crate::decode_signed_varint(&input[cursor..])?;
            cursor += consumed;
            Ok((Value::Number(serde_json::Number::from(val)), cursor))
        }
        0x04 => {
            let (val, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;
            Ok((Value::Number(serde_json::Number::from(val)), cursor))
        }
        0x05 => {
            let (len, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;
            let s = std::str::from_utf8(&input[cursor..cursor + len as usize])
                .map_err(|_| FluxPackError::InvalidUtf8)?
                .to_string();
            cursor += len as usize;
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
            match serde_json::Number::from_f64(f) {
                Some(n) => Ok((Value::Number(n), cursor)),
                None => Ok((Value::Null, cursor)),
            }
        }
        0x09 => {
            let (len, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;
            let mut arr = Vec::with_capacity(len as usize);
            for _ in 0..len {
                let (val, consumed) = decode_value(&input[cursor..], table)?;
                cursor += consumed;
                arr.push(val);
            }
            Ok((Value::Array(arr), cursor))
        }
        0x0A => {
            let (len, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;
            let mut obj = serde_json::Map::with_capacity(len as usize);
            for _ in 0..len {
                let (token, consumed) = decode_varint(&input[cursor..])?;
                cursor += consumed;
                let key = table.resolve(token as u16)
                    .ok_or(FluxPackError::UnknownToken(token as u16))?
                    .to_string();
                let (val, consumed) = decode_value(&input[cursor..], table)?;
                cursor += consumed;
                obj.insert(key, val);
            }
            Ok((Value::Object(obj), cursor))
        }
        0x0D => {
            // Columnar data embedded in a value
            let (data_len, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;
            let (row_count, columns, _col_consumed) = decode_columnar(&input[cursor..cursor + data_len as usize])?;
            cursor += data_len as usize;
            let arr = reconstruct_array(row_count, columns);
            Ok((arr, cursor))
        }
        _ => Err(FluxPackError::InvalidValueType(value_type)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_stream_roundtrip() {
        let messages = vec![
            json!({"user_id": 8821, "email": "user@example.com", "active": true}),
            json!({"user_id": 8822, "email": "other@example.com", "active": false}),
            json!({"user_id": 8823, "email": "third@example.com", "active": true}),
        ];

        let mut writer = StreamWriter::new();
        for msg in &messages {
            writer.write_data(msg).unwrap();
        }
        let bytes = writer.into_buffer();

        let mut reader = StreamReader::new();
        let frames = reader.read_all(&bytes).unwrap();

        // Should have 3 DATA frames (DEFs are absorbed into symbol table)
        let data_frames: Vec<&Value> = frames.iter().filter_map(|f| match f {
            Frame::Data(v) => Some(v),
            _ => None,
        }).collect();

        assert_eq!(data_frames.len(), 3);
        assert_eq!(*data_frames[0], messages[0]);
        assert_eq!(*data_frames[1], messages[1]);
        assert_eq!(*data_frames[2], messages[2]);
    }

    #[test]
    fn test_stream_batch() {
        let messages = vec![
            json!({"a": 1, "b": 2}),
            json!({"a": 3, "b": 4}),
        ];

        let mut writer = StreamWriter::new();
        writer.write_batch(&messages).unwrap();
        let bytes = writer.into_buffer();

        let mut reader = StreamReader::new();
        let frames = reader.read_all(&bytes).unwrap();
        let data_count = frames.iter().filter(|f| matches!(f, Frame::Data(_))).count();
        assert_eq!(data_count, 2);
    }

    #[test]
    fn test_stream_iterator() {
        let messages = vec![
            json!({"x": 1}),
            json!({"x": 2}),
        ];

        let mut writer = StreamWriter::new();
        for msg in &messages {
            writer.write_data(msg).unwrap();
        }
        let bytes = writer.into_buffer();

        let mut reader = StreamReader::new();
        let frames: Vec<Frame> = reader.frames(&bytes).collect();
        let data_count = frames.iter().filter(|f| matches!(f, Frame::Data(_))).count();
        assert_eq!(data_count, 2);
    }
}
