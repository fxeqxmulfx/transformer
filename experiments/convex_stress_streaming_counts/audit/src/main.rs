use convex_streaming_raw_audit::{Result, run};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [root, binary, producer, case, output] = args.as_slice() else {
        return Err(
            "usage: convex-streaming-raw-audit RUN FROZEN_BINARY PRODUCER CASE FRESH_EXTERNAL_JSON"
                .into(),
        );
    };
    run::case(
        Path::new(root),
        Path::new(binary),
        producer,
        case,
        Path::new(output),
    )
}
