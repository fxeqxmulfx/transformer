// Reuse the checked diagnostic accessors and local trainer, without mutation.
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../forward_alignment/src/instrumentation.rs"
));
