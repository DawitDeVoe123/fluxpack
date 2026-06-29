use crate::{FluxPackError, encode_varint, decode_varint};

/// Tensor encoding for ML models.
///
/// Native support for multi-dimensional numeric arrays with shape metadata.
/// This is dramatically more efficient than encoding individual numbers:
///
/// JSON:    [1.0, 2.0, 3.0] → "1.0","2.0","3.0" (type tags + string encoding)
/// FluxPack: → 3 × float64 (no type tags per element)
/// Tensor:  → shape + raw f32 bytes (4x smaller than float64)
///
/// Wire format:
///   dtype(byte) | ndims(varint) | shape[ndims](varint×ndims) | data(flat bytes)
///
/// Data type tags for tensors.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorDtype {
    F32 = 0x00,
    F64 = 0x01,
    I32 = 0x02,
    I64 = 0x03,
    U32 = 0x04,
    U64 = 0x05,
}

impl TensorDtype {
    #[inline]
    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0x00 => Some(TensorDtype::F32),
            0x01 => Some(TensorDtype::F64),
            0x02 => Some(TensorDtype::I32),
            0x03 => Some(TensorDtype::I64),
            0x04 => Some(TensorDtype::U32),
            0x05 => Some(TensorDtype::U64),
            _ => None,
        }
    }

    #[inline]
    pub fn element_size(self) -> usize {
        match self {
            TensorDtype::F32 | TensorDtype::I32 | TensorDtype::U32 => 4,
            TensorDtype::F64 | TensorDtype::I64 | TensorDtype::U64 => 8,
        }
    }
}

/// A tensor with shape and typed data.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub dtype: TensorDtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Tensor {
    /// Create a tensor from f32 slice with shape.
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Self {
        let byte_data: Vec<u8> = data.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        Self {
            dtype: TensorDtype::F32,
            shape,
            data: byte_data,
        }
    }

    /// Create a tensor from f64 slice with shape.
    pub fn from_f64(data: &[f64], shape: Vec<usize>) -> Self {
        let byte_data: Vec<u8> = data.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        Self {
            dtype: TensorDtype::F64,
            shape,
            data: byte_data,
        }
    }

    /// Create a tensor from i32 slice with shape.
    pub fn from_i32(data: &[i32], shape: Vec<usize>) -> Self {
        let byte_data: Vec<u8> = data.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        Self {
            dtype: TensorDtype::I32,
            shape,
            data: byte_data,
        }
    }

    /// Create a tensor from i64 slice with shape.
    pub fn from_i64(data: &[i64], shape: Vec<usize>) -> Self {
        let byte_data: Vec<u8> = data.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        Self {
            dtype: TensorDtype::I64,
            shape,
            data: byte_data,
        }
    }

    /// Create a tensor from a JSON array of numbers.
    /// Infers dtype from the values. Shape is flat (1D) unless specified.
    pub fn from_json_array(arr: &[serde_json::Value], shape: Option<Vec<usize>>) -> Result<Self, FluxPackError> {
        if arr.is_empty() {
            return Err(FluxPackError::ColumnarError("empty tensor".into()));
        }

        // Detect if all values are integers or floats
        let all_ints = arr.iter().all(|v| match v {
            serde_json::Value::Number(n) => n.is_i64() || n.is_u64(),
            _ => false,
        });
        let all_floats = arr.iter().all(|v| match v {
            serde_json::Value::Number(n) => n.is_f64(),
            _ => false,
        });

        let inferred_shape = shape.unwrap_or_else(|| vec![arr.len()]);

        if all_ints {
            let data: Vec<i64> = arr.iter().filter_map(|v| {
                if let serde_json::Value::Number(n) = v {
                    n.as_i64()
                } else {
                    None
                }
            }).collect();
            Ok(Self::from_i64(&data, inferred_shape))
        } else if all_floats || all_ints {
            // Mix or all floats → use f64
            let data: Vec<f64> = arr.iter().filter_map(|v| {
                if let serde_json::Value::Number(n) = v {
                    n.as_f64()
                } else {
                    None
                }
            }).collect();
            Ok(Self::from_f64(&data, inferred_shape))
        } else {
            // Use f32 for mixed or float values (smaller)
            let data: Vec<f32> = arr.iter().filter_map(|v| {
                if let serde_json::Value::Number(n) = v {
                    n.as_f64().map(|f| f as f32)
                } else {
                    None
                }
            }).collect();
            Ok(Self::from_f32(&data, inferred_shape))
        }
    }

    /// Total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether the tensor is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Decode this tensor back to a JSON array.
    pub fn to_json(&self) -> Result<serde_json::Value, FluxPackError> {
        match self.dtype {
            TensorDtype::F32 => {
                let values: Vec<serde_json::Value> = self.data.chunks(4)
                    .map(|chunk| {
                        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        let f = f32::from_bits(bits);
                        serde_json::Value::Number(serde_json::Number::from_f64(f as f64).unwrap_or(serde_json::Number::from(0)))
                    })
                    .collect();
                Ok(serde_json::Value::Array(values))
            }
            TensorDtype::F64 => {
                let values: Vec<serde_json::Value> = self.data.chunks(8)
                    .map(|chunk| {
                        let bits = u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]);
                        let f = f64::from_bits(bits);
                        serde_json::Value::Number(serde_json::Number::from_f64(f).unwrap_or(serde_json::Number::from(0)))
                    })
                    .collect();
                Ok(serde_json::Value::Array(values))
            }
            TensorDtype::I32 => {
                let values: Vec<serde_json::Value> = self.data.chunks(4)
                    .map(|chunk| {
                        let bits = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        serde_json::Value::Number(serde_json::Number::from(bits))
                    })
                    .collect();
                Ok(serde_json::Value::Array(values))
            }
            TensorDtype::I64 => {
                let values: Vec<serde_json::Value> = self.data.chunks(8)
                    .map(|chunk| {
                        let bits = i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]);
                        serde_json::Value::Number(serde_json::Number::from(bits))
                    })
                    .collect();
                Ok(serde_json::Value::Array(values))
            }
            TensorDtype::U32 => {
                let values: Vec<serde_json::Value> = self.data.chunks(4)
                    .map(|chunk| {
                        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        serde_json::Value::Number(serde_json::Number::from(bits))
                    })
                    .collect();
                Ok(serde_json::Value::Array(values))
            }
            TensorDtype::U64 => {
                let values: Vec<serde_json::Value> = self.data.chunks(8)
                    .map(|chunk| {
                        let bits = u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]);
                        serde_json::Value::Number(serde_json::Number::from(bits))
                    })
                    .collect();
                Ok(serde_json::Value::Array(values))
            }
        }
    }
}

/// Encode a tensor into a buffer.
pub fn encode_tensor(tensor: &Tensor, buffer: &mut Vec<u8>) {
    // dtype
    buffer.push(tensor.dtype as u8);

    // ndims
    encode_varint(tensor.shape.len() as u64, buffer);

    // shape
    for &dim in &tensor.shape {
        encode_varint(dim as u64, buffer);
    }

    // data (flat bytes)
    buffer.extend_from_slice(&tensor.data);
}

/// Decode a tensor from a buffer. Returns (tensor, bytes consumed).
pub fn decode_tensor(input: &[u8]) -> Result<(Tensor, usize), FluxPackError> {
    let mut cursor = 0;

    // dtype
    let dtype = TensorDtype::from_tag(input[cursor])
        .ok_or(FluxPackError::UnsupportedTensorDtype(input[cursor]))?;
    cursor += 1;

    // ndims
    let (ndims, consumed) = decode_varint(&input[cursor..])?;
    cursor += consumed;

    // shape
    let mut shape = Vec::with_capacity(ndims as usize);
    for _ in 0..ndims {
        let (dim, consumed) = decode_varint(&input[cursor..])?;
        cursor += consumed;
        shape.push(dim as usize);
    }

    // data
    let total_elements: usize = shape.iter().product();
    let data_len = total_elements * dtype.element_size();
    let data = input[cursor..cursor + data_len].to_vec();
    cursor += data_len;

    Ok((Tensor { dtype, shape, data }, cursor))
}

/// Check if a JSON array looks like a numeric tensor (all numbers, large enough).
pub fn is_tensor_candidate(arr: &[serde_json::Value]) -> bool {
    if arr.len() < 8 {
        return false;
    }
    arr.iter().all(|v| matches!(v, serde_json::Value::Number(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tensor_f32_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_f32(&data, vec![2, 3]);

        let mut buf = Vec::new();
        encode_tensor(&tensor, &mut buf);

        let (decoded, consumed) = decode_tensor(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(decoded.dtype, TensorDtype::F32);
        assert_eq!(decoded.shape, vec![2, 3]);
        assert_eq!(decoded.data, tensor.data);
    }

    #[test]
    fn test_tensor_i64_roundtrip() {
        let data = vec![100i64, 200, 300, 400];
        let tensor = Tensor::from_i64(&data, vec![4]);

        let mut buf = Vec::new();
        encode_tensor(&tensor, &mut buf);

        let (decoded, consumed) = decode_tensor(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(decoded.dtype, TensorDtype::I64);
        assert_eq!(decoded.shape, vec![4]);
    }

    #[test]
    fn test_tensor_from_json() {
        let arr = vec![json!(1.0), json!(2.0), json!(3.0), json!(4.0)];
        let tensor = Tensor::from_json_array(&arr, Some(vec![2, 2])).unwrap();

        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.len(), 4);

        let mut buf = Vec::new();
        encode_tensor(&tensor, &mut buf);

        let (decoded, _) = decode_tensor(&buf).unwrap();
        let back = decoded.to_json().unwrap();
        assert_eq!(back, serde_json::Value::Array(arr));
    }

    #[test]
    fn test_tensor_3d() {
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let tensor = Tensor::from_f64(&data, vec![2, 3, 4]);

        let mut buf = Vec::new();
        encode_tensor(&tensor, &mut buf);

        let (decoded, _) = decode_tensor(&buf).unwrap();
        assert_eq!(decoded.shape, vec![2, 3, 4]);
        assert_eq!(decoded.len(), 24);
    }

    #[test]
    fn test_tensor_json_roundtrip() {
        let arr: Vec<serde_json::Value> = (0..100).map(|i| json!(i as f64)).collect();
        let tensor = Tensor::from_json_array(&arr, Some(vec![10, 10])).unwrap();

        let mut buf = Vec::new();
        encode_tensor(&tensor, &mut buf);

        let (decoded, _) = decode_tensor(&buf).unwrap();
        let back = decoded.to_json().unwrap();

        match back {
            serde_json::Value::Array(vals) => {
                assert_eq!(vals.len(), 100);
                for (i, v) in vals.iter().enumerate() {
                    if let serde_json::Value::Number(n) = v {
                        assert_eq!(n.as_f64().unwrap(), i as f64);
                    }
                }
            }
            _ => panic!("Expected array"),
        }
    }

    #[test]
    fn test_tensor_size_comparison() {
        // Large tensor with large numbers where tensor encoding wins
        let arr: Vec<serde_json::Value> = (0..10000)
            .map(|i| json!(i as f64 * 12345.6789))
            .collect();

        let json_size = serde_json::to_vec(&arr).unwrap().len();

        let tensor = Tensor::from_f32(
            &arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect::<Vec<f32>>(),
            vec![10000],
        );
        let mut buf = Vec::new();
        encode_tensor(&tensor, &mut buf);
        let tensor_size = buf.len();

        // f32 tensor (4 bytes/element) vs JSON (~12 bytes/number for large floats)
        assert!(tensor_size < json_size,
            "Tensor f32 ({}) should be smaller than JSON ({})",
            tensor_size, json_size);
    }
}
