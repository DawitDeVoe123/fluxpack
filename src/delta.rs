use crate::{FluxPackError, encode_varint, decode_varint, encode_signed_varint, decode_signed_varint};
use serde_json::{Value, Number};

/// Delta encoding for numeric sequences.
///
/// Instead of encoding [100, 101, 103, 100], we encode:
///   first_value=100, deltas=[+1, +2, -3]
///
/// Deltas are ZigZag encoded so small differences (common in ML metrics)
/// get tiny varint representations.
///
/// Wire format:
///   count(varint) | first_value(signed_varint) | deltas[(count-1) × signed_varint]
///
/// Minimum sequence length to benefit from delta encoding.
/// Sequences shorter than this are encoded normally.
pub const DELTA_THRESHOLD: usize = 4;

/// Check if a sequence of numbers would benefit from delta encoding.
/// Returns true if the values are close together (small deltas).
pub fn should_delta_encode(values: &[Value]) -> bool {
    if values.len() < DELTA_THRESHOLD {
        return false;
    }

    // All values must be numeric
    let nums: Vec<i64> = values.iter().filter_map(|v| {
        match v {
            Value::Number(n) => n.as_i64(),
            _ => None,
        }
    }).collect();

    if nums.len() != values.len() {
        return false;
    }

    // Check if deltas are small (average absolute delta < 1000)
    // This heuristic works well for ML metrics like loss, accuracy, etc.
    if nums.len() < 2 {
        return false;
    }

    let mut total_delta: u64 = 0;
    for w in nums.windows(2) {
        let delta = (w[1] - w[0]).unsigned_abs();
        total_delta += delta;
    }
    let avg_delta = total_delta / (nums.len() - 1) as u64;

    // Delta encoding is beneficial when average delta is small
    avg_delta < 1000
}

/// Encode a numeric sequence using delta encoding.
/// Returns the encoded bytes.
pub fn encode_delta(values: &[Value], buffer: &mut Vec<u8>) -> Result<(), FluxPackError> {
    let nums: Vec<i64> = values.iter().filter_map(|v| {
        match v {
            Value::Number(n) => n.as_i64(),
            _ => None,
        }
    }).collect();

    if nums.is_empty() {
        return Err(FluxPackError::ColumnarError("empty sequence".into()));
    }

    // Write count
    encode_varint(nums.len() as u64, buffer);

    // Write first value absolutely
    encode_signed_varint(nums[0], buffer);

    // Write deltas
    for window in nums.windows(2) {
        let delta = window[1] - window[0];
        encode_signed_varint(delta, buffer);
    }

    Ok(())
}

/// Decode a delta-encoded numeric sequence.
/// Returns the decoded values and bytes consumed.
pub fn decode_delta(input: &[u8]) -> Result<(Vec<Value>, usize), FluxPackError> {
    let mut cursor = 0;

    // Read count
    let (count, consumed) = decode_varint(&input[cursor..])?;
    cursor += consumed;

    if count == 0 {
        return Ok((vec![], cursor));
    }

    // Read first value
    let (first, consumed) = decode_signed_varint(&input[cursor..])?;
    cursor += consumed;

    let mut values = Vec::with_capacity(count as usize);
    values.push(Value::Number(Number::from(first)));

    // Read deltas and reconstruct
    let mut current = first;
    for _ in 1..count {
        let (delta, consumed) = decode_signed_varint(&input[cursor..])?;
        cursor += consumed;
        current = current.wrapping_add(delta);
        values.push(Value::Number(Number::from(current)));
    }

    Ok((values, cursor))
}

/// Calculate the size of delta-encoded data without actually encoding.
pub fn delta_encoded_size(values: &[Value]) -> usize {
    let nums: Vec<i64> = values.iter().filter_map(|v| {
        match v {
            Value::Number(n) => n.as_i64(),
            _ => None,
        }
    }).collect();

    if nums.is_empty() {
        return 0;
    }

    let mut size = varint_size(nums.len() as u64); // count
    size += signed_varint_size(nums[0]); // first value

    for window in nums.windows(2) {
        let delta = window[1] - window[0];
        size += signed_varint_size(delta);
    }

    size
}

fn varint_size(mut value: u64) -> usize {
    if value == 0 { return 1; }
    let mut len = 0;
    while value > 0 {
        len += 1;
        value >>= 7;
    }
    len
}

fn signed_varint_size(value: i64) -> usize {
    varint_size(crate::zigzag_encode(value))
}

/// Calculate the size of the original (non-delta) encoding.
pub fn original_encoded_size(values: &[Value]) -> usize {
    let mut size = 0;
    for v in values {
        size += 1; // type tag
        if let Value::Number(n) = v {
            if let Some(i) = n.as_i64() {
                size += signed_varint_size(i);
            } else if let Some(_f) = n.as_f64() {
                size += 8; // f64
            }
        }
    }
    size
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_delta_roundtrip() {
        let values = vec![
            json!(100), json!(101), json!(103), json!(100), json!(98),
        ];

        let mut buf = Vec::new();
        encode_delta(&values, &mut buf).unwrap();

        let (decoded, consumed) = decode_delta(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_delta_negative_numbers() {
        let values = vec![
            json!(-10), json!(-8), json!(-12), json!(-5),
        ];

        let mut buf = Vec::new();
        encode_delta(&values, &mut buf).unwrap();

        let (decoded, consumed) = decode_delta(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_delta_single_element() {
        let values = vec![json!(42)];

        let mut buf = Vec::new();
        encode_delta(&values, &mut buf).unwrap();

        let (decoded, _) = decode_delta(&buf).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_should_delta_encode() {
        // Close values → should encode
        let close = vec![json!(100), json!(101), json!(103), json!(100)];
        assert!(should_delta_encode(&close));

        // Too short → should not
        let short = vec![json!(100), json!(101)];
        assert!(!should_delta_encode(&short));

        // Spread out values → should not
        let spread = vec![json!(1), json!(10000), json!(2), json!(20000)];
        assert!(!should_delta_encode(&spread));
    }

    #[test]
    fn test_delta_compression_ratio() {
        // Integer sequence: values decrease gradually
        let loss = vec![
            json!(250), json!(180), json!(120), json!(80),
            json!(50), json!(30), json!(20), json!(10),
        ];

        let delta_size = {
            let mut buf = Vec::new();
            encode_delta(&loss, &mut buf).unwrap();
            buf.len()
        };

        let original_size = original_encoded_size(&loss);

        // Delta should be significantly smaller
        assert!(delta_size < original_size,
            "Delta ({}) should be smaller than original ({})",
            delta_size, original_size);
    }
}
