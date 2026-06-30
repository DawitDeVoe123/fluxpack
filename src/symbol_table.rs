use ahash::AHashMap;
use crate::MAX_TOKENS;

/// Pre-defined common ML pipeline field names for zero-cost interning.
pub const COMMON_ML_KEYS: &[&str] = &[
    "epoch", "loss", "accuracy", "val_loss", "val_accuracy",
    "learning_rate", "batch_size", "step", "timestamp", "status",
    "model_type", "optimizer", "lr", "metrics", "config",
    "train_loss", "test_loss", "f1", "precision", "recall",
];

/// The shared symbol table that both encoder and decoder maintain.
///
/// Key optimization: `id_to_key` uses a `Vec<String>` indexed by token ID
/// for O(1) array access instead of O(1)-amortized hash lookup.
/// Tokens are sequential u16 values (1, 2, 3...), so direct indexing is safe.
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// Token-to-key mapping using Vec for O(1) direct indexing.
    /// Index 0 is unused; tokens start at 1.
    id_to_key: Vec<String>,
    /// Key-to-token mapping (still uses HashMap for O(1) insert/lookup).
    key_to_id: AHashMap<String, u16>,
    next_token: u16,
    /// Tracks which tokens have had DEF frames emitted (encoder only)
    emitted_defs: Vec<bool>,
    /// Schema fingerprint: hash of all keys in insertion order
    schema_fingerprint: u64,
    /// Cached sorted tokens for fingerprint calculation
    sorted_tokens_cache: Vec<u16>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self::with_predefined()
    }

    /// Create a new symbol table pre-loaded with common ML keys.
    pub fn with_predefined() -> Self {
        // Pre-allocate Vec with index 0 as empty (tokens start at 1)
        let mut id_to_key: Vec<String> = Vec::with_capacity(256);
        id_to_key.push(String::new()); // index 0 unused

        let mut key_to_id = AHashMap::with_capacity(64);
        let mut next_token: u16 = 1;

        // Pre-register common ML keys for zero-overhead interning
        for &key in COMMON_ML_KEYS {
            if next_token > MAX_TOKENS {
                break;
            }
            key_to_id.insert(key.to_string(), next_token);
            id_to_key.push(key.to_string());
            next_token += 1;
        }

        let emitted_count = (next_token - 1) as usize;
        let sorted_tokens_cache: Vec<u16> = (1..next_token).collect();
        Self {
            id_to_key,
            key_to_id,
            next_token,
            emitted_defs: vec![true; emitted_count],
            schema_fingerprint: 0,
            sorted_tokens_cache,
        }
    }

    /// Returns the token ID for a key. If the key is new, assigns a new token.
    #[inline]
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
        self.id_to_key.push(key.to_string());
        self.emitted_defs.push(false);
        self.sorted_tokens_cache.push(id);
        self.schema_fingerprint = 0;
        Ok(id)
    }

    /// Store a DEF frame mapping directly (used by decoder).
    #[inline]
    pub fn store_def(&mut self, token: u16, key: &str) -> Result<(), crate::FluxPackError> {
        if token > MAX_TOKENS {
            return Err(crate::FluxPackError::TableOverflow);
        }

        let idx = token as usize;

        // Check if already stored — O(1) Vec access
        if idx < self.id_to_key.len() && !self.id_to_key[idx].is_empty() {
            if self.id_to_key[idx] != key {
                return Err(crate::FluxPackError::DuplicateDef(token));
            }
            return Ok(());
        }

        self.key_to_id.insert(key.to_string(), token);

        // Ensure Vec is large enough for direct indexing
        if idx >= self.id_to_key.len() {
            self.id_to_key.resize(idx + 1, String::new());
        }
        self.id_to_key[idx] = key.to_string();

        if idx >= self.emitted_defs.len() {
            self.emitted_defs.resize(idx + 1, false);
        }

        if token >= self.next_token {
            self.next_token = token + 1;
        }

        self.sorted_tokens_cache.push(token);
        self.sorted_tokens_cache.sort_unstable();
        self.sorted_tokens_cache.dedup();
        self.schema_fingerprint = 0;
        Ok(())
    }

    /// Looks up a key by token ID. O(1) direct Vec indexing.
    #[inline]
    pub fn resolve(&self, id: u16) -> Option<&str> {
        let idx = id as usize;
        if idx < self.id_to_key.len() && !self.id_to_key[idx].is_empty() {
            Some(&self.id_to_key[idx])
        } else {
            None
        }
    }

    /// Returns true if this key has already been interned.
    #[inline]
    pub fn contains_key(&self, key: &str) -> bool {
        self.key_to_id.contains_key(key)
    }

    /// Check if a DEF frame has been emitted for this token.
    #[inline]
    pub fn def_emitted(&self, token: u16) -> bool {
        self.emitted_defs
            .get(token as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Mark a DEF frame as emitted for this token.
    #[inline]
    pub fn mark_def_emitted(&mut self, token: u16) {
        let idx = token as usize;
        if idx >= self.emitted_defs.len() {
            self.emitted_defs.resize(idx + 1, false);
        }
        self.emitted_defs[idx] = true;
    }

    /// Returns true if all keys in the table have had their DEFs emitted.
    #[inline]
    pub fn all_defs_emitted(&self) -> bool {
        self.emitted_defs.iter().all(|&b| b)
    }

    /// Compute a schema fingerprint (hash of all keys in token order).
    pub fn schema_fingerprint(&mut self) -> u64 {
        if self.schema_fingerprint != 0 {
            return self.schema_fingerprint;
        }
        let mut hasher = ahash::AHasher::default();
        use std::hash::{Hash, Hasher};
        for &token in &self.sorted_tokens_cache {
            let idx = token as usize;
            if idx < self.id_to_key.len() && !self.id_to_key[idx].is_empty() {
                token.hash(&mut hasher);
                self.id_to_key[idx].hash(&mut hasher);
            }
        }
        let fp = hasher.finish();
        self.schema_fingerprint = if fp == 0 { 1 } else { fp };
        self.schema_fingerprint
    }

    /// Compare schema fingerprints to determine if DEF frames can be skipped.
    #[inline]
    pub fn schema_matches(&self, other_fp: u64) -> bool {
        self.schema_fingerprint != 0 && self.schema_fingerprint == other_fp
    }

    /// Returns the current size of the symbol table.
    #[inline]
    pub fn size(&self) -> usize {
        self.id_to_key.len().saturating_sub(1) // subtract index 0
    }

    /// Returns whether the table is empty (excluding index 0).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.id_to_key.len() <= 1
    }

    /// Returns the next token that would be assigned.
    #[inline]
    pub fn next_token(&self) -> u16 {
        self.next_token
    }

    /// Clears the symbol table entirely.
    pub fn reset(&mut self) {
        self.id_to_key.clear();
        self.id_to_key.push(String::new()); // index 0 unused
        self.key_to_id.clear();
        self.next_token = 1;
        self.emitted_defs.clear();
        self.schema_fingerprint = 0;
        self.sorted_tokens_cache.clear();

        // Re-register common ML keys
        for &key in COMMON_ML_KEYS {
            if self.next_token > MAX_TOKENS {
                break;
            }
            self.key_to_id.insert(key.to_string(), self.next_token);
            self.id_to_key.push(key.to_string());
            self.emitted_defs.push(true);
            self.sorted_tokens_cache.push(self.next_token);
            self.next_token += 1;
        }
    }

    /// Returns an iterator over all (token, key) pairs in token order.
    pub fn iter(&self) -> impl Iterator<Item = (u16, &str)> {
        self.sorted_tokens_cache
            .iter()
            .filter_map(|&token| {
                let idx = token as usize;
                if idx < self.id_to_key.len() && !self.id_to_key[idx].is_empty() {
                    Some((token, self.id_to_key[idx].as_str()))
                } else {
                    None
                }
            })
    }

    /// Returns the list of tokens that need DEF frames emitted.
    pub fn pending_defs(&self) -> Vec<(u16, &str)> {
        self.sorted_tokens_cache
            .iter()
            .filter(|&&token| !self.def_emitted(token))
            .filter_map(|&token| {
                let idx = token as usize;
                if idx < self.id_to_key.len() && !self.id_to_key[idx].is_empty() {
                    Some((token, self.id_to_key[idx].as_str()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns the count of pending DEF frames.
    pub fn pending_defs_count(&self) -> usize {
        self.sorted_tokens_cache
            .iter()
            .filter(|&&token| !self.def_emitted(token))
            .count()
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_o1() {
        let table = SymbolTable::with_predefined();
        // Common ML keys should be at known token IDs
        assert_eq!(table.resolve(1), Some("epoch"));
        assert_eq!(table.resolve(2), Some("loss"));
        assert_eq!(table.resolve(20), Some("recall"));
    }

    #[test]
    fn test_intern_and_resolve() {
        let mut table = SymbolTable::new();
        let token = table.intern("my_custom_key").unwrap();
        assert_eq!(table.resolve(token), Some("my_custom_key"));
    }

    #[test]
    fn test_store_def() {
        let mut table = SymbolTable::new();
        table.store_def(100, "custom_key").unwrap();
        assert_eq!(table.resolve(100), Some("custom_key"));
    }

    #[test]
    fn test_resolve_nonexistent() {
        let table = SymbolTable::new();
        assert_eq!(table.resolve(9999), None);
    }
}
