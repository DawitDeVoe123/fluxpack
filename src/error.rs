use thiserror::Error;
use crate::MAX_TOKENS;
#[derive(Debug, Error, Clone, PartialEq)]
pub enum FluxPackError {
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u8),

    #[error("Unknown token: {0}")]
    UnknownToken(u16),

    #[error("Table overflow: max tokens {MAX_TOKENS}")]
    TableOverflow,

    #[error("Duplicate DEF for token: {0}")]
    DuplicateDef(u16),

    #[error("Invalid UTF-8 in key")]
    InvalidUtf8,

    #[error("Malformed frame")]
    MalformedFrame,

    #[error("Invalid value type: {0}")]
    InvalidValueType(u8),

    #[error("Unknown struct: {0}")]
    UnknownStruct(u16),

    #[error("Buffer overrun")]
    BufferOverrun,

    #[error("Varint overflow")]
    VarintOverflow,

    #[error("Expected object")]
    ExpectedObject,

    #[error("Internal error")]
    InternalError,

    #[error("Columnar encoding failed: {0}")]
    ColumnarError(String),

    #[error("Unsupported tensor dtype: {0}")]
    UnsupportedTensorDtype(u8),

    #[error("Shape mismatch: expected {expected} elements, got {got}")]
    TensorShapeMismatch { expected: usize, got: usize },

    #[error("Incomplete stream: expected {expected} more frames")]
    IncompleteStream { expected: usize },

    #[error("Schema mismatch: {0}")]
    SchemaMismatch(String),
}
