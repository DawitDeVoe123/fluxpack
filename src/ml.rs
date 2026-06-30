use crate::{FluxPackError, encode_varint, decode_varint, encode_signed_varint, decode_signed_varint};
use crate::tensor::{Tensor, TensorDtype};
use serde_json::{Value, Number};

/// Feature vector encoding for ML models.
///
/// Common in ML pipelines for representing feature vectors, embeddings,
/// and model outputs. More efficient than encoding as a generic array.
///
/// Wire format:
///   ndims(varint) | shape[ndims](varint×ndims) | dtype(byte) | data(flat bytes)
pub struct FeatureVector {
    pub shape: Vec<usize>,
    pub dtype: TensorDtype,
    pub data: Vec<u8>,
}

impl FeatureVector {
    /// Create from f32 slice.
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Self {
        let byte_data: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self { shape, dtype: TensorDtype::F32, data: byte_data }
    }

    /// Create from f64 slice.
    pub fn from_f64(data: &[f64], shape: Vec<usize>) -> Self {
        let byte_data: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self { shape, dtype: TensorDtype::F64, data: byte_data }
    }

    /// Create from i64 slice.
    pub fn from_i64(data: &[i64], shape: Vec<usize>) -> Self {
        let byte_data: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self { shape, dtype: TensorDtype::I64, data: byte_data }
    }

    /// Create from JSON array.
    pub fn from_json(arr: &[Value], shape: Option<Vec<usize>>) -> Result<Self, FluxPackError> {
        let tensor = Tensor::from_json_array(arr, shape)?;
        Ok(Self {
            shape: tensor.shape,
            dtype: tensor.dtype,
            data: tensor.data,
        })
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether the feature vector is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Convert to Tensor.
    pub fn to_tensor(&self) -> Tensor {
        Tensor { dtype: self.dtype, shape: self.shape.clone(), data: self.data.clone() }
    }

    /// Decode back to JSON array.
    pub fn to_json(&self) -> Result<Value, FluxPackError> {
        self.to_tensor().to_json()
    }
}

/// Encode a feature vector into a buffer.
pub fn encode_feature_vector(fv: &FeatureVector, buffer: &mut Vec<u8>) {
    buffer.push(fv.dtype as u8);
    encode_varint(fv.shape.len() as u64, buffer);
    for &dim in &fv.shape {
        encode_varint(dim as u64, buffer);
    }
    buffer.extend_from_slice(&fv.data);
}

/// Decode a feature vector from a buffer.
pub fn decode_feature_vector(input: &[u8]) -> Result<(FeatureVector, usize), FluxPackError> {
    let mut cursor = 0;

    let dtype = TensorDtype::from_tag(input[cursor])
        .ok_or(FluxPackError::UnsupportedTensorDtype(input[cursor]))?;
    cursor += 1;

    let (ndims, consumed) = decode_varint(&input[cursor..])?;
    cursor += consumed;

    let mut shape = Vec::with_capacity(ndims as usize);
    for _ in 0..ndims {
        let (dim, consumed) = decode_varint(&input[cursor..])?;
        cursor += consumed;
        shape.push(dim as usize);
    }

    let total_elements: usize = shape.iter().product();
    let data_len = total_elements * dtype.element_size();
    let data = input[cursor..cursor + data_len].to_vec();
    cursor += data_len;

    Ok((FeatureVector { dtype, shape, data }, cursor))
}

/// Sparse tensor for encoding ML data with many zeros.
///
/// Instead of storing all elements, stores only non-zero values with their indices.
/// Dramatically smaller for sparse data (e.g., one-hot encoded features, NLP embeddings).
///
/// Wire format:
///   dtype(byte) | ndims(varint) | shape[ndims] | nnz(varint) | indices[nnz](varint×nnz) | values[nnz]
pub struct SparseTensor {
    pub dtype: TensorDtype,
    pub shape: Vec<usize>,
    pub indices: Vec<Vec<usize>>,
    pub values: Vec<u8>,
}

impl SparseTensor {
    /// Create from dense f32 array, automatically extracting non-zero values.
    pub fn from_dense_f32(dense: &[f32], shape: Vec<usize>) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &val) in dense.iter().enumerate() {
            if val != 0.0 {
                indices.push(vec![i]);
                values.extend_from_slice(&val.to_le_bytes());
            }
        }

        Self { dtype: TensorDtype::F32, shape, indices, values }
    }

    /// Create from index-value pairs.
    pub fn from_sparse(dtype: TensorDtype, shape: Vec<usize>, indices: Vec<Vec<usize>>, values: Vec<u8>) -> Self {
        Self { dtype, shape, indices, values }
    }

    /// Total number of elements (including zeros).
    pub fn dense_size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Sparsity ratio (fraction of zeros).
    pub fn sparsity(&self) -> f64 {
        let total = self.dense_size();
        if total == 0 { return 0.0; }
        1.0 - (self.nnz() as f64 / total as f64)
    }

    /// Check if sparse encoding is beneficial (typically when sparsity > 50%).
    pub fn is_beneficial(&self) -> bool {
        self.sparsity() > 0.5
    }
}

/// Encode a sparse tensor into a buffer.
pub fn encode_sparse_tensor(sparse: &SparseTensor, buffer: &mut Vec<u8>) {
    buffer.push(sparse.dtype as u8);

    encode_varint(sparse.shape.len() as u64, buffer);
    for &dim in &sparse.shape {
        encode_varint(dim as u64, buffer);
    }

    let nnz = sparse.nnz();
    encode_varint(nnz as u64, buffer);

    for idx in &sparse.indices {
        for &i in idx {
            encode_varint(i as u64, buffer);
        }
    }

    buffer.extend_from_slice(&sparse.values);
}

/// Decode a sparse tensor from a buffer.
pub fn decode_sparse_tensor(input: &[u8]) -> Result<(SparseTensor, usize), FluxPackError> {
    let mut cursor = 0;

    let dtype = TensorDtype::from_tag(input[cursor])
        .ok_or(FluxPackError::UnsupportedTensorDtype(input[cursor]))?;
    cursor += 1;

    let (ndims, consumed) = decode_varint(&input[cursor..])?;
    cursor += consumed;

    let mut shape = Vec::with_capacity(ndims as usize);
    for _ in 0..ndims {
        let (dim, consumed) = decode_varint(&input[cursor..])?;
        cursor += consumed;
        shape.push(dim as usize);
    }

    let (nnz, consumed) = decode_varint(&input[cursor..])?;
    cursor += consumed;

    let mut indices = Vec::with_capacity(nnz as usize);
    for _ in 0..nnz {
        let mut idx = Vec::with_capacity(ndims as usize);
        for _ in 0..ndims {
            let (i, consumed) = decode_varint(&input[cursor..])?;
            cursor += consumed;
            idx.push(i as usize);
        }
        indices.push(idx);
    }

    let total_elements: usize = nnz as usize;
    let data_len = total_elements * dtype.element_size();
    let values = input[cursor..cursor + data_len].to_vec();
    cursor += data_len;

    Ok((SparseTensor { dtype, shape, indices, values }, cursor))
}

/// Convert sparse tensor to dense tensor.
pub fn sparse_to_dense(sparse: &SparseTensor) -> Tensor {
    let total_elements = sparse.dense_size();
    let elem_size = sparse.dtype.element_size();
    let mut dense_data = vec![0u8; total_elements * elem_size];

    for (idx_pos, idx) in sparse.indices.iter().enumerate() {
        let flat_idx = idx.iter().enumerate().fold(0usize, |acc, (dim, &i)| {
            let stride: usize = shape_stride(&sparse.shape, dim + 1);
            acc + i * stride
        });
        let byte_offset = flat_idx * elem_size;
        let value_offset = idx_pos * elem_size;
        dense_data[byte_offset..byte_offset + elem_size]
            .copy_from_slice(&sparse.values[value_offset..value_offset + elem_size]);
    }

    Tensor { dtype: sparse.dtype, shape: sparse.shape.clone(), data: dense_data }
}

fn shape_stride(shape: &[usize], dim: usize) -> usize {
    shape[dim..].iter().product()
}

/// Timestamp encoding for ML training logs.
///
/// Encodes millisecond timestamps efficiently. Uses varint encoding
/// for the timestamp value, which is compact for recent timestamps.
///
/// Wire format:
///   timestamp_ms(varint)
pub fn encode_timestamp(timestamp_ms: u64, buffer: &mut Vec<u8>) {
    encode_varint(timestamp_ms, buffer);
}

/// Decode a timestamp from bytes.
pub fn decode_timestamp(input: &[u8]) -> Result<(u64, usize), FluxPackError> {
    decode_varint(input)
}

/// Encode a relative timestamp (offset from a base time).
/// Useful for training logs where timestamps are close together.
///
/// Wire format:
///   base_ms(varint) | count(varint) | deltas[count-1](signed_varint × (count-1))
pub fn encode_timestamps_deltas(timestamps_ms: &[u64], buffer: &mut Vec<u8>) -> Result<(), FluxPackError> {
    if timestamps_ms.is_empty() {
        return Err(FluxPackError::ColumnarError("empty timestamps".into()));
    }

    encode_varint(timestamps_ms[0], buffer);
    encode_varint(timestamps_ms.len() as u64, buffer);

    let mut prev = timestamps_ms[0] as i64;
    for &ts in timestamps_ms.iter().skip(1) {
        let delta = ts as i64 - prev;
        encode_signed_varint(delta, buffer);
        prev = ts as i64;
    }

    Ok(())
}

/// Decode delta-encoded timestamps.
pub fn decode_timestamps_deltas(input: &[u8]) -> Result<(Vec<u64>, usize), FluxPackError> {
    let mut cursor = 0;

    let (base, consumed) = decode_varint(&input[cursor..])?;
    cursor += consumed;

    let (count, consumed) = decode_varint(&input[cursor..])?;
    cursor += consumed;

    let mut timestamps = Vec::with_capacity(count as usize);
    timestamps.push(base);

    let mut current = base as i64;
    for _ in 1..count {
        let (delta, consumed) = decode_signed_varint(&input[cursor..])?;
        cursor += consumed;
        current = current.wrapping_add(delta);
        timestamps.push(current as u64);
    }

    Ok((timestamps, cursor))
}

/// Hyperparameter encoding for ML training configs.
///
/// Encodes common hyperparameter types efficiently:
/// - Learning rate (float, often small decimals)
/// - Batch size (integer, often powers of 2)
/// - Epochs (integer)
/// - Optimizer name (interned string)
///
/// Wire format:
///   lr(float32) | batch_size(varint) | epochs(varint) | optimizer_token(varint) | custom_count(varint) | custom_key-value pairs
pub struct Hyperparams {
    pub learning_rate: f32,
    pub batch_size: u32,
    pub epochs: u32,
    pub optimizer: Option<String>,
    pub custom: Vec<(String, Value)>,
}

/// Encode hyperparameters into a buffer.
pub fn encode_hyperparams(params: &Hyperparams, buffer: &mut Vec<u8>) {
    buffer.extend_from_slice(&params.learning_rate.to_le_bytes());
    encode_varint(params.batch_size as u64, buffer);
    encode_varint(params.epochs as u64, buffer);

    // Optimizer name as interned string (would need symbol table integration)
    match &params.optimizer {
        Some(name) => {
            buffer.push(1); // has optimizer
            encode_varint(name.len() as u64, buffer);
            buffer.extend_from_slice(name.as_bytes());
        }
        None => buffer.push(0), // no optimizer
    }

    encode_varint(params.custom.len() as u64, buffer);
    for (key, value) in &params.custom {
        encode_varint(key.len() as u64, buffer);
        buffer.extend_from_slice(key.as_bytes());
        encode_value_flat(value, buffer);
    }
}

/// Decode hyperparameters from a buffer.
pub fn decode_hyperparams(input: &[u8]) -> Result<(Hyperparams, usize), FluxPackError> {
    let mut cursor = 0;

    let lr_bits = u32::from_le_bytes([
        input[cursor], input[cursor+1], input[cursor+2], input[cursor+3],
    ]);
    let learning_rate = f32::from_bits(lr_bits);
    cursor += 4;

    let (batch_size, consumed) = decode_varint(&input[cursor..])?;
    cursor += consumed;

    let (epochs, consumed) = decode_varint(&input[cursor..])?;
    cursor += consumed;

    let has_optimizer = input[cursor] == 1;
    cursor += 1;

    let optimizer = if has_optimizer {
        let (len, consumed) = decode_varint(&input[cursor..])?;
        cursor += consumed;
        let name = std::str::from_utf8(&input[cursor..cursor + len as usize])
            .map_err(|_| FluxPackError::InvalidUtf8)?
            .to_string();
        cursor += len as usize;
        Some(name)
    } else {
        None
    };

    let (custom_count, consumed) = decode_varint(&input[cursor..])?;
    cursor += consumed;

    let mut custom = Vec::with_capacity(custom_count as usize);
    for _ in 0..custom_count {
        let (key_len, consumed) = decode_varint(&input[cursor..])?;
        cursor += consumed;
        let key = std::str::from_utf8(&input[cursor..cursor + key_len as usize])
            .map_err(|_| FluxPackError::InvalidUtf8)?
            .to_string();
        cursor += key_len as usize;

        let (value, consumed) = decode_value_flat(&input[cursor..])?;
        cursor += consumed;
        custom.push((key, value));
    }

    Ok((Hyperparams { learning_rate, batch_size: batch_size as u32, epochs: epochs as u32, optimizer, custom }, cursor))
}

fn encode_value_flat(value: &Value, buffer: &mut Vec<u8>) {
    match value {
        Value::Null => buffer.push(0x00),
        Value::Bool(true) => buffer.push(0x01),
        Value::Bool(false) => buffer.push(0x02),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                buffer.push(0x03);
                encode_signed_varint(i, buffer);
            } else if let Some(f) = n.as_f64() {
                buffer.push(0x06);
                buffer.extend_from_slice(&f.to_bits().to_le_bytes());
            }
        }
        Value::String(s) => {
            buffer.push(0x05);
            encode_varint(s.len() as u64, buffer);
            buffer.extend_from_slice(s.as_bytes());
        }
        _ => buffer.push(0x00),
    }
}

fn decode_value_flat(input: &[u8]) -> Result<(Value, usize), FluxPackError> {
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
        _ => Err(FluxPackError::InvalidValueType(value_type)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_feature_vector_f32_roundtrip() {
        let fv = FeatureVector::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut buf = Vec::new();
        encode_feature_vector(&fv, &mut buf);

        let (decoded, consumed) = decode_feature_vector(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(decoded.shape, vec![2, 2]);
        assert_eq!(decoded.data, fv.data);
    }

    #[test]
    fn test_sparse_tensor_roundtrip() {
        let dense = vec![1.0f32, 0.0, 0.0, 2.0, 0.0, 3.0];
        let sparse = SparseTensor::from_dense_f32(&dense, vec![6]);

        assert_eq!(sparse.nnz(), 3);
        assert!(sparse.sparsity() > 0.4);

        let mut buf = Vec::new();
        encode_sparse_tensor(&sparse, &mut buf);

        let (decoded, consumed) = decode_sparse_tensor(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(decoded.nnz(), 3);

        let back_to_dense = sparse_to_dense(&decoded);
        assert_eq!(back_to_dense.data.len(), dense.len() * 4);
    }

    #[test]
    fn test_timestamp_deltas() {
        let timestamps = vec![1000, 1016, 1032, 1048, 1064];
        let mut buf = Vec::new();
        encode_timestamps_deltas(&timestamps, &mut buf).unwrap();

        let (decoded, consumed) = decode_timestamps_deltas(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(decoded, timestamps);
    }

    #[test]
    fn test_hyperparams_roundtrip() {
        let params = Hyperparams {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            optimizer: Some("adam".to_string()),
            custom: vec![
                ("weight_decay".to_string(), json!(0.01)),
                ("warmup_steps".to_string(), json!(1000)),
            ],
        };

        let mut buf = Vec::new();
        encode_hyperparams(&params, &mut buf);

        let (decoded, consumed) = decode_hyperparams(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert!((decoded.learning_rate - 0.001).abs() < 0.0001);
        assert_eq!(decoded.batch_size, 32);
        assert_eq!(decoded.epochs, 100);
        assert_eq!(decoded.optimizer, Some("adam".to_string()));
        assert_eq!(decoded.custom.len(), 2);
    }

    #[test]
    fn test_sparse_tensor_compression() {
        // 90% sparse tensor
        let mut dense = vec![0.0f32; 1000];
        dense[0] = 1.0;
        dense[100] = 2.0;
        dense[500] = 3.0;

        let sparse = SparseTensor::from_dense_f32(&dense, vec![1000]);

        let mut buf = Vec::new();
        encode_sparse_tensor(&sparse, &mut buf);

        let json_size = serde_json::to_vec(&dense).unwrap().len();
        let dense_fp_size = 1000 * 4 + 10; // rough estimate

        println!("Dense JSON: {} bytes", json_size);
        println!("Dense FluxPack: ~{} bytes", dense_fp_size);
        println!("Sparse FluxPack: {} bytes", buf.len());
        println!("Sparsity: {:.1}%", sparse.sparsity() * 100.0);

        // Sparse should be much smaller than dense for 90% sparsity
        assert!(buf.len() < dense_fp_size);
    }
}
