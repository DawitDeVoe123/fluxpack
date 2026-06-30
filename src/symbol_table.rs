use ahash::AHashMap;
use crate::MAX_TOKENS;

/// Pre-defined common ML pipeline field names for zero-cost interning.
/// These get tokens 1-20 if present, avoiding DEF frame overhead entirely.
pub const COMMON_ML_KEYS: &[&str] = &[
    "epoch", "loss", "accuracy", "val_loss", "val_accuracy",
    "learning_rate", "batch_size", "step", "timestamp", "status",
    "model_type", "optimizer", "lr", "metrics", "config",
    "train_loss", "test_loss", "f1", "precision", "recall",
];

/// Pre-defined keys indexed by token for O(1) lookup during encoding.
pub const COMMON_ML_KEYS_BY_TOKEN: &[&str] = &[
    "", // token 0 unused
    "epoch", "loss", "accuracy", "val_loss", "val_accuracy",
    "learning_rate", "batch_size", "step", "timestamp", "status",
    "model_type", "optimizer", "lr", "metrics", "config",
    "train_loss", "test_loss", "f1", "precision", "recall",
];

/// The shared symbol table that both encoder and decoder maintain.
#[derive(Debug, Clone)]
pub struct SymbolTable {
    id_to_key: AHashMap<u16, String>,
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
    /// This gives a huge speedup for typical ML pipelines.
    pub fn with_predefined() -> Self {
        let mut id_to_key = AHashMap::with_capacity(64);
        let mut key_to_id = AHashMap::with_capacity(64);
        let mut next_token: u16 = 1;

        // Pre-register common ML keys for zero-overhead interning
        for &key in COMMON_ML_KEYS {
            if next_token > MAX_TOKENS {
                break;
            }
            key_to_id.insert(key.to_string(), next_token);
            id_to_key.insert(next_token, key.to_string());
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
        self.id_to_key.insert(id, key.to_string());
        self.emitted_defs.push(false);
        self.sorted_tokens_cache.push(id);
        self.schema_fingerprint = 0;
        Ok(id)
    }

    /// Store a DEF frame mapping directly (used by decoder).
    /// This ensures the decoder uses the EXACT token ID from the wire.
    #[inline]
    pub fn store_def(&mut self, token: u16, key: &str) -> Result<(), crate::FluxPackError> {
        if token > MAX_TOKENS {
            return Err(crate::FluxPackError::TableOverflow);
        }

        if self.id_to_key.contains_key(&token) {
            if let Some(existing) = self.id_to_key.get(&token) {
                if existing != key {
                    return Err(crate::FluxPackError::DuplicateDef(token));
                }
            }
            return Ok(());
        }

        self.key_to_id.insert(key.to_string(), token);
        self.id_to_key.insert(token, key.to_string());

        let idx = token as usize;
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

    /// Looks up a key by token ID. Used during decoding.
    #[inline]
    pub fn resolve(&self, id: u16) -> Option<&str> {
        self.id_to_key.get(&id).map(|s| s.as_str())
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
    /// Used by the encoder to skip redundant DEF frame emission.
    #[inline]
    pub fn all_defs_emitted(&self) -> bool {
        self.emitted_defs.iter().all(|&b| b)
    }

    /// Compute a schema fingerprint (hash of all keys in token order).
    /// Two symbol tables with the same keys will have the same fingerprint.
    /// Used to skip DEF frames when schema is unchanged.
    pub fn schema_fingerprint(&mut self) -> u64 {
        if self.schema_fingerprint != 0 {
            return self.schema_fingerprint;
        }
        let mut hasher = ahash::AHasher::default();
        use std::hash::{Hash, Hasher};
        for &token in &self.sorted_tokens_cache {
            if let Some(key) = self.id_to_key.get(&token) {
                token.hash(&mut hasher);
                key.hash(&mut hasher);
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
        self.id_to_key.len()
    }

    /// Returns whether the table is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.id_to_key.is_empty()
    }

    /// Returns the next token that would be assigned.
    #[inline]
    pub fn next_token(&self) -> u16 {
        self.next_token
    }

    /// Clears the symbol table entirely.
    pub fn reset(&mut self) {
        self.id_to_key.clear();
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
            self.id_to_key.insert(self.next_token, key.to_string());
            self.emitted_defs.push(true);
            self.sorted_tokens_cache.push(self.next_token);
            self.next_token += 1;
        }
    }

    /// Returns an iterator over all (token, key) pairs in token order.
    pub fn iter(&self) -> impl Iterator<Item = (u16, &str)> {
        self.sorted_tokens_cache
            .iter()
            .filter_map(|&token| self.id_to_key.get(&token).map(|key| (token, key.as_str())))
    }

    /// Returns the list of tokens that need DEF frames emitted.
    pub fn pending_defs(&self) -> Vec<(u16, &str)> {
        self.sorted_tokens_cache
            .iter()
            .filter(|&&token| !self.def_emitted(token))
            .filter_map(|&token| self.id_to_key.get(&token).map(|key| (token, key.as_str())))
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
