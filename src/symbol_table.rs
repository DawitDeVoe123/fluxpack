use std::collections::HashMap;
use crate::MAX_TOKENS;

/// The shared symbol table that both encoder and decoder maintain.
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// Maps a token ID (1..=MAX_TOKENS) to its UTF-8 key string.
    id_to_key: HashMap<u16, String>,
    /// Maps a key string to its token ID for O(1) lookup during encoding.
    key_to_id: HashMap<String, u16>,
    /// The next available token ID to assign.
    next_token: u16,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            id_to_key: HashMap::new(),
            key_to_id: HashMap::new(),
            next_token: 1, // 0 is reserved for inline keys in debug mode
        }
    }

    /// Returns the token ID for a key. If the key is new, assigns a new token.
    pub fn intern(&mut self, key: &str) -> Result<u16, crate::FluxPackError> {
        if let Some(&id) = self.key_to_id.get(key) {
            return Ok(id);
        }

        if self.next_token > MAX_TOKENS {
            return Err(crate::FluxPackError::TableOverflow);
        }

        let id = self.next_token;
        self.next_token += 1;
        self.key_to_id.insert(key.to_string(), id);
        self.id_to_key.insert(id, key.to_string());
        Ok(id)
    }

    /// Looks up a key by token ID. Used during decoding.
    pub fn resolve(&self, id: u16) -> Option<&str> {
        self.id_to_key.get(&id).map(|s| s.as_str())
    }

    /// Returns the current size of the symbol table.
    pub fn size(&self) -> usize {
        self.id_to_key.len()
    }

    /// Returns whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_key.is_empty()
    }

    /// Clears the symbol table entirely.
    pub fn reset(&mut self) {
        self.id_to_key.clear();
        self.key_to_id.clear();
        self.next_token = 1;
    }

    /// Returns an iterator over all (token, key) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u16, &str)> {
        self.id_to_key.iter().map(|(&id, key)| (id, key.as_str()))
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}