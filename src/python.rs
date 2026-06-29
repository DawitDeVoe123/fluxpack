use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyBytes, PyTuple};
use pyo3::exceptions::PyValueError;
use serde_json::{Value, Map, Number};

use crate::{Encoder, Decoder, StreamWriter as CoreStreamWriter, StreamReader as CoreStreamReader, Frame};

/// FluxPack: schema-free, Shannon-optimal serialization for ML pipelines.
///
/// # Quick Start
/// ```python
/// import fluxpack
///
/// data = {"epoch": 1, "loss": 0.5, "accuracy": 0.8}
/// encoded = fluxpack.encode(data)
/// decoded = fluxpack.decode(encoded)
/// assert decoded == data
/// ```
#[pymodule]
fn fluxpack(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_py, m)?)?;
    m.add_function(wrap_pyfunction!(decode_py, m)?)?;
    m.add_function(wrap_pyfunction!(encode_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(decode_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(encode_compressed_py, m)?)?;
    m.add_function(wrap_pyfunction!(decode_compressed_py, m)?)?;
    m.add_class::<PyEncoder>()?;
    m.add_class::<PyDecoder>()?;
    m.add_class::<PyStreamWriter>()?;
    m.add_class::<PyStreamReader>()?;
    m.add("__version__", "0.2.0")?;
    Ok(())
}

/// Encode a Python dict to FluxPack bytes.
///
/// >>> encoded = fluxpack.encode({"key": "value", "count": 42})
#[pyfunction]
fn encode_py(py: Python<'_>, data: &Bound<'_, PyDict>) -> PyResult<PyObject> {
    let value = dict_to_json(py, data)?;
    let mut encoder = Encoder::new();
    let encoded = encoder.encode(&value)
        .map_err(|e| PyValueError::new_err(format!("Encode error: {}", e)))?;
    Ok(PyBytes::new(py, encoded).into())
}

/// Decode FluxPack bytes to a Python dict.
///
/// >>> data = fluxpack.decode(encoded)
#[pyfunction]
fn decode_py(py: Python<'_>, data: &Bound<'_, PyBytes>) -> PyResult<PyObject> {
    let bytes = data.as_bytes();
    let mut decoder = Decoder::new();
    let value = decoder.decode(bytes)
        .map_err(|e| PyValueError::new_err(format!("Decode error: {}", e)))?;
    json_to_py(py, &value)
}

/// Encode multiple dicts to a single FluxPack batch.
///
/// >>> encoded = fluxpack.encode_batch([{"a": 1}, {"a": 2}, {"a": 3}])
#[pyfunction]
fn encode_batch_py(py: Python<'_>, messages: &Bound<'_, PyList>) -> PyResult<PyObject> {
    let mut encoder = Encoder::new();
    let mut values = Vec::new();

    for item in messages.iter() {
        if let Ok(dict) = item.downcast::<PyDict>() {
            values.push(dict_to_json(py, dict)?);
        } else {
            return Err(PyValueError::new_err("All items must be dicts"));
        }
    }

    let encoded = encoder.encode_batch(&values)
        .map_err(|e| PyValueError::new_err(format!("Batch encode error: {}", e)))?;
    Ok(PyBytes::new(py, encoded).into())
}

/// Decode a FluxPack batch to a list of Python dicts.
///
/// >>> messages = fluxpack.decode_batch(encoded)
#[pyfunction]
fn decode_batch_py(py: Python<'_>, data: &Bound<'_, PyBytes>) -> PyResult<PyObject> {
    let bytes = data.as_bytes();
    let mut decoder = Decoder::new();
    let values = decoder.decode_all(bytes)
        .map_err(|e| PyValueError::new_err(format!("Batch decode error: {}", e)))?;

    let list = PyList::empty(py);
    for value in &values {
        let py_val = json_to_py(py, value)?;
        list.append(py_val)?;
    }
    Ok(list.into())
}

/// Encode and compress in one step.
///
/// >>> compressed = fluxpack.encode_compressed(data, level=3)
#[pyfunction]
#[pyo3(signature = (data, level=3))]
fn encode_compressed_py(py: Python<'_>, data: &Bound<'_, PyDict>, level: i32) -> PyResult<PyObject> {
    let value = dict_to_json(py, data)?;
    let mut encoder = Encoder::new();
    let fluxpack_bytes = encoder.encode(&value)
        .map_err(|e| PyValueError::new_err(format!("Encode error: {}", e)))?;

    let compressed = crate::compress::compress_with_level(fluxpack_bytes, level)
        .map_err(|e| PyValueError::new_err(format!("Compression error: {}", e)))?;

    Ok(PyBytes::new(py, &compressed).into())
}

/// Decompress and decode in one step.
///
/// >>> data = fluxpack.decode_compressed(compressed)
#[pyfunction]
fn decode_compressed_py(py: Python<'_>, data: &Bound<'_, PyBytes>) -> PyResult<PyObject> {
    let bytes = data.as_bytes();
    let decompressed = crate::compress::decompress(bytes)
        .map_err(|e| PyValueError::new_err(format!("Decompression error: {}", e)))?;

    let mut decoder = Decoder::new();
    let value = decoder.decode(&decompressed)
        .map_err(|e| PyValueError::new_err(format!("Decode error: {}", e)))?;

    json_to_py(py, &value)
}

/// Encoder with state reuse for streaming ML pipelines.
///
/// >>> encoder = fluxpack.Encoder()
/// >>> for batch in training_batches:
/// ...     encoded = encoder.encode(batch)
#[pyclass]
struct PyEncoder {
    inner: Encoder,
}

#[pymethods]
impl PyEncoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Encoder::new(),
        }
    }

    /// Encode a dict to FluxPack bytes.
    fn encode(&mut self, py: Python<'_>, data: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        let value = dict_to_json(py, data)?;
        let encoded = self.inner.encode(&value)
            .map_err(|e| PyValueError::new_err(format!("Encode error: {}", e)))?;
        Ok(PyBytes::new(py, encoded).into())
    }

    /// Encode with columnar optimization for arrays of objects.
    fn encode_with_columnar(&mut self, py: Python<'_>, data: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        let value = dict_to_json(py, data)?;
        let encoded = self.inner.encode_with_columnar(&value)
            .map_err(|e| PyValueError::new_err(format!("Encode error: {}", e)))?;
        Ok(PyBytes::new(py, encoded).into())
    }

    /// Encode a batch of dicts.
    fn encode_batch(&mut self, py: Python<'_>, messages: &Bound<'_, PyList>) -> PyResult<PyObject> {
        let mut values = Vec::new();
        for item in messages.iter() {
            if let Ok(dict) = item.downcast::<PyDict>() {
                values.push(dict_to_json(py, dict)?);
            } else {
                return Err(PyValueError::new_err("All items must be dicts"));
            }
        }
        let encoded = self.inner.encode_batch(&values)
            .map_err(|e| PyValueError::new_err(format!("Batch encode error: {}", e)))?;
        Ok(PyBytes::new(py, encoded).into())
    }

    /// Reset the encoder state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get the symbol table size.
    fn symbol_table_size(&self) -> usize {
        self.inner.symbol_table_size()
    }
}

/// Decoder with state reuse.
///
/// >>> decoder = fluxpack.Decoder()
/// >>> data = decoder.decode(encoded)
#[pyclass]
struct PyDecoder {
    inner: Decoder,
}

#[pymethods]
impl PyDecoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Decoder::new(),
        }
    }

    /// Decode FluxPack bytes to a Python dict.
    fn decode(&mut self, py: Python<'_>, data: &Bound<'_, PyBytes>) -> PyResult<PyObject> {
        let bytes = data.as_bytes();
        let value = self.inner.decode(bytes)
            .map_err(|e| PyValueError::new_err(format!("Decode error: {}", e)))?;
        json_to_py(py, &value)
    }

    /// Decode a batch of messages.
    fn decode_all(&mut self, py: Python<'_>, data: &Bound<'_, PyBytes>) -> PyResult<PyObject> {
        let bytes = data.as_bytes();
        let values = self.inner.decode_all(bytes)
            .map_err(|e| PyValueError::new_err(format!("Decode error: {}", e)))?;

        let list = PyList::empty(py);
        for value in &values {
            list.append(json_to_py(py, value)?)?;
        }
        Ok(list.into())
    }

    /// Reset the decoder state.
    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Streaming writer for frame-level encoding.
#[pyclass]
struct PyStreamWriter {
    inner: CoreStreamWriter,
}

#[pymethods]
impl PyStreamWriter {
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreStreamWriter::new(),
        }
    }

    /// Write a DEF frame for a key.
    fn write_def(&mut self, key: &str) -> PyResult<u16> {
        self.inner.write_def(key)
            .map_err(|e| PyValueError::new_err(format!("Write DEF error: {}", e)))
    }

    /// Write a DATA frame.
    fn write_data(&mut self, py: Python<'_>, data: &Bound<'_, PyDict>) -> PyResult<()> {
        let value = dict_to_json(py, data)?;
        self.inner.write_data(&value)
            .map_err(|e| PyValueError::new_err(format!("Write DATA error: {}", e)))
    }

    /// Get the accumulated buffer.
    fn finish<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.buffer())
    }

    /// Reset the writer.
    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Streaming reader for frame-level decoding.
#[pyclass]
struct PyStreamReader {
    inner: CoreStreamReader,
}

#[pymethods]
impl PyStreamReader {
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreStreamReader::new(),
        }
    }

    /// Read all frames from bytes and return as a list of dicts.
    /// Each dict has a "type" field ("def", "data", "columnar") and relevant data.
    fn read_all(&mut self, py: Python<'_>, data: &Bound<'_, PyBytes>) -> PyResult<PyObject> {
        let bytes = data.as_bytes();
        let frames = self.inner.read_all(bytes)
            .map_err(|e| PyValueError::new_err(format!("Read error: {}", e)))?;

        let list = PyList::empty(py);
        for frame in frames {
            let frame_dict = PyDict::new(py);
            match frame {
                Frame::Def { token, key } => {
                    frame_dict.set_item("type", "def")?;
                    frame_dict.set_item("token", token)?;
                    frame_dict.set_item("key", &key)?;
                }
                Frame::Data(value) => {
                    frame_dict.set_item("type", "data")?;
                    frame_dict.set_item("value", json_to_py(py, &value)?)?;
                }
                Frame::Columnar { row_count, .. } => {
                    frame_dict.set_item("type", "columnar")?;
                    frame_dict.set_item("row_count", row_count)?;
                }
                Frame::Eos => {
                    frame_dict.set_item("type", "eos")?;
                }
            }
            list.append(frame_dict)?;
        }
        Ok(list.into())
    }

    /// Reset the reader.
    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Convert a serde_json::Value to a Python object.
fn json_to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.into_pyobject(py)?.into()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into())
            } else {
                Ok(py.None())
            }
        }
        Value::String(s) => Ok(s.into_pyobject(py)?.into()),
        Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into())
        }
        Value::Object(obj) => {
            let dict = PyDict::new(py);
            for (key, val) in obj {
                dict.set_item(key.as_str(), json_to_py(py, val)?)?;
            }
            Ok(dict.into())
        }
    }
}

/// Convert a Python dict to serde_json::Value.
fn dict_to_json(py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<Value> {
    let mut map = Map::new();

    for (key, value) in dict.iter() {
        let key_str: String = key.extract(py)?;
        map.insert(key_str, py_to_json(py, &value)?);
    }

    Ok(Value::Object(map))
}

/// Convert a Python object to serde_json::Value.
fn py_to_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        Ok(Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(Value::Number(Number::from(i)))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(Value::Number(Number::from_f64(f).unwrap_or(Number::from(0))))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(Value::String(s))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let mut arr = Vec::new();
        for item in list.iter() {
            arr.push(py_to_json(py, &item)?);
        }
        Ok(Value::Array(arr))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        dict_to_json(py, dict)
    } else if let Ok(tuple) = obj.downcast::<PyTuple>() {
        let mut arr = Vec::new();
        for item in tuple.iter() {
            arr.push(py_to_json(py, &item)?);
        }
        Ok(Value::Array(arr))
    } else {
        Ok(Value::Null)
    }
}
