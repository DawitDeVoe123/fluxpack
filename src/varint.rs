/// Encodes a u64 as a LEB128 varint (compatible with Protocol Buffers).
#[inline(always)]
pub fn encode_varint(mut value: u64, buffer: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buffer.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Decodes a u64 from a LEB128 varint. Returns (value, bytes_consumed).
#[inline(always)]
pub fn decode_varint(data: &[u8]) -> Result<(u64, usize), crate::FluxPackError> {
    let mut result = 0u64;
    let mut shift = 0u32;

    for (i, &byte) in data.iter().enumerate().take(10) {
        result |= ((byte & 0x7F) as u64) << shift;
        if (byte & 0x80) == 0 {
            return Ok((result, i + 1));
        }
        shift += 7;
    }

    Err(crate::FluxPackError::VarintOverflow)
}

/// Returns the number of bytes a varint will occupy when encoded.
#[inline(always)]
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

/// ZigZag encode: maps signed i64 to unsigned u64 for optimal varint encoding.
/// -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, etc.
/// Small negative numbers (common in ML loss/gradients) get small varint encodings.
#[inline(always)]
pub fn zigzag_encode(value: i64) -> u64 {
    let shifted = value.wrapping_shl(1);
    let sign = value >> 63;
    (shifted ^ sign) as u64
}

/// ZigZag decode: maps unsigned u64 back to signed i64.
#[inline(always)]
pub fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

/// Encode a signed i64 using ZigZag + varint.
#[inline(always)]
pub fn encode_signed_varint(value: i64, buffer: &mut Vec<u8>) {
    encode_varint(zigzag_encode(value), buffer);
}

/// Decode a signed varint (ZigZag + LEB128). Returns (value, bytes_consumed).
#[inline(always)]
pub fn decode_signed_varint(data: &[u8]) -> Result<(i64, usize), crate::FluxPackError> {
    let (val, consumed) = decode_varint(data)?;
    Ok((zigzag_decode(val), consumed))
}

/// Write a varint directly to a byte slice at a given offset.
/// Returns the number of bytes written. Caller must ensure enough space.
/// This avoids Vec overhead for hot paths.
#[inline(always)]
pub fn encode_varint_to_slice(mut value: u64, buf: &mut [u8]) -> usize {
    let start = 0;
    let mut pos = start;
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf[pos] = byte;
        pos += 1;
        if value == 0 {
            break;
        }
    }
    pos - start
}

/// Batch decode: decode `count` varints from a buffer. Returns values and total bytes consumed.
#[inline]
pub fn decode_varints_batch(data: &[u8], count: usize) -> Result<(Vec<u64>, usize), crate::FluxPackError> {
    let mut values = Vec::with_capacity(count);
    let mut cursor = 0;
    for _ in 0..count {
        let (val, consumed) = decode_varint(&data[cursor..])?;
        values.push(val);
        cursor += consumed;
    }
    Ok((values, cursor))
}

/// Encode a batch of varints into a buffer.
#[inline]
pub fn encode_varints_batch(values: &[u64], buffer: &mut Vec<u8>) {
    for &v in values {
        encode_varint(v, buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_roundtrip() {
        let values = [0u64, 1, 127, 128, 255, 256, 16383, 16384, 0x3FFF, u32::MAX as u64, u64::MAX];
        for &val in &values {
            let mut buf = Vec::new();
            encode_varint(val, &mut buf);
            let (decoded, consumed) = decode_varint(&buf).unwrap();
            assert_eq!(val, decoded, "Failed for value {}", val);
            assert_eq!(consumed, buf.len());
        }
    }

    #[test]
    fn test_varint_len() {
        assert_eq!(varint_len(0), 1);
        assert_eq!(varint_len(127), 1);
        assert_eq!(varint_len(128), 2);
        assert_eq!(varint_len(16383), 2);
        assert_eq!(varint_len(16384), 3);
    }

    #[test]
    fn test_zigzag_roundtrip() {
        let values = [0i64, 1, -1, 2, -2, 63, -64, 64, -65, i32::MAX as i64, i32::MIN as i64, i64::MAX, i64::MIN];
        for &val in &values {
            let encoded = zigzag_encode(val);
            let decoded = zigzag_decode(encoded);
            assert_eq!(val, decoded, "Failed for value {}", val);
            // Small absolute values should encode to small unsigned values
            if val.unsigned_abs() <= 64 {
                assert!(encoded <= 129, "ZigZag({}) = {} should be small", val, encoded);
            }
        }
    }

    #[test]
    fn test_zigzag_optimal_encoding() {
        // Negative numbers that are "small" in ML (loss values, gradients) should be compact
        assert!(zigzag_encode(-1) < 128); // 1 byte varint
        assert!(zigzag_encode(-2) < 128); // 1 byte varint
        assert!(zigzag_encode(-64) < 128); // 1 byte varint
        assert!(zigzag_encode(63) < 128); // 1 byte varint
    }

    #[test]
    fn test_batch_roundtrip() {
        let values = vec![1u64, 128, 256, 16384, u64::MAX];
        let mut buf = Vec::new();
        encode_varints_batch(&values, &mut buf);
        let (decoded, _) = decode_varints_batch(&buf, values.len()).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_signed_varint_roundtrip() {
        let values = [0i64, 1, -1, 42, -42, 1000, -1000, i64::MAX, i64::MIN];
        for &val in &values {
            let mut buf = Vec::new();
            encode_signed_varint(val, &mut buf);
            let (decoded, _) = decode_signed_varint(&buf).unwrap();
            assert_eq!(val, decoded, "Failed for value {}", val);
        }
    }
}
