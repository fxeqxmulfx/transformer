// Appended to the pinned Engine by the shared build script. Original training
// equations are unchanged; extensions provide operand-only serving and accounting.
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../single_update/src/instrumentation.rs"
));
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/allocation.rs"));
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/prefix.rs"));
