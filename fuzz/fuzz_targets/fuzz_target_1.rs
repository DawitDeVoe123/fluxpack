#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz the decoder with random bytes
    // This must NEVER panic — only return errors
    let mut decoder = fluxpack::Decoder::new();
    let _ = decoder.decode(data);
});
