#![no_main]
use libfuzzer_sys::fuzz_target;
use serde_json::Value;

fuzz_target!(|data: &[u8]| {
    // Fuzz inline decoder with random bytes
    if !data.is_empty() && data[0] == 0xFE {
        let _ = fluxpack::inline::decode_inline(data);
    }

    // Also fuzz standard decoder with random bytes
    let mut decoder = fluxpack::Decoder::new();
    let _ = decoder.decode(data);

    // Fuzz zero-copy decoder
    let _ = fluxpack::decode_zero_copy(data);
});
