#![no_main]
use libfuzzer_sys::fuzz_target;
use serde_json::Value;

fuzz_target!(|data: &[u8]| {
    // Fuzz the decoder with random bytes — must never panic
    let mut decoder = fluxpack::Decoder::new();
    let _ = decoder.decode(data);

    // Also test zero-copy decoder
    let _ = fluxpack::decode_zero_copy(data);

    // Also test decode_all
    let mut decoder2 = fluxpack::Decoder::new();
    let _ = decoder2.decode_all(data);
});
