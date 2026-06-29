/// Encodes a u64 as a LEB128 varint (compatible with Protocol Buffers).
pub fn encode_varint(mut value: u64, buffer: &mut Vec<u8>) {
    while value >= 0x80 {
        buffer.push(((value & 0x7F) | 0x80) as u8);
        value >>= 7;
    }
    buffer.push(value as u8);
}

/// Decodes a u64 from a LEB128 varint. Returns (value, bytes_consumed).
pub fn decode_varint(data: &[u8]) -> Result<(u64, usize), crate::FluxPackError> {
    let mut result = 0u64;
    let mut shift = 0;
    let mut consumed = 0;

    for &byte in data.iter() {
        consumed += 1;
        result |= ((byte & 0x7F) as u64) << shift;

        if (byte & 0x80) == 0 {
            return Ok((result, consumed));
        }

        shift += 7;
        if shift >= 64 {
            return Err(crate::FluxPackError::VarintOverflow);
        }
    }

    Err(crate::FluxPackError::BufferOverrun)
}

/// Returns the number of bytes a varint will occupy when encoded.
pub fn varint_len(mut value: u64) -> usize {
    if value == 0 {
        return 1;
    }
    let mut len = 0;
    while value > 0 {
        len += 1;
        value >>= 7;
    }
    len
}