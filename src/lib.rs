//! FluxPack — A schema‑free, Shannon‑optimal serialisation format.
//!
//! This crate implements the FluxPack wire format specification v1.0.
//! For full details, see the spec at `/docs/spec.md`.

pub mod symbol_table;
pub mod varint;
pub mod error;

pub use symbol_table::SymbolTable;
pub use varint::{encode_varint, decode_varint, varint_len};
pub use error::FluxPackError;

/// Magic bytes that identify a FluxPack stream: F X P 0x01
pub const MAGIC: [u8; 4] = [0x46, 0x58, 0x50, 0x01];

/// Maximum number of tokens per session (14-bit space)
pub const MAX_TOKENS: u16 = 0x3FFF;

/// Frame types
#[repr(u8)]
pub enum FrameType {
    Def = 0x01,
    Data = 0x02,
    Struct = 0x03,
    Reset = 0x04,
    Debug = 0x05,
    Ack = 0x06,
}

/// Value types
#[repr(u8)]
pub enum ValueType {
    Null = 0x00,
    BoolTrue = 0x01,
    BoolFalse = 0x02,
    IntVar = 0x03,
    UintVar = 0x04,
    String = 0x05,
    Float64 = 0x06,
    Float32 = 0x07,
    Bytes = 0x08,
    Array = 0x09,
    Object = 0x0A,
    Interned = 0x0B,
    Timestamp = 0x0C,
}