//! Complete dense recovery cost control; no stress data or accuracy credit.
use convex_bilinear_sequence::{
    C, Result, algebra,
    domain::Shape,
    guard,
    model::{Joint, Model},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: recovery_bench WORKSPACE STORAGE".into());
    }
    let shape = Shape::new(100, 10)?;
    let count = 3255;
    let reserved = shape.reservation(count)?;
    guard::admit(reserved, Path::new(&args[0]), Path::new(&args[1]))?;
    algebra::set_threads(1)?;
    let mut joint = Joint::zero(shape);
    for d in 0..count {
        joint.columns.insert(
            d,
            (0..shape.objects * shape.queries)
                .map(|i| {
                    C::new(
                        ((i * 773 + d * 991) % 1009) as f64 / 1024.,
                        ((i * 577 + d * 787) % 1013) as f64 / 1024.,
                    )
                })
                .collect(),
        );
    }
    let started = Instant::now();
    let model = Model::recover(joint, 80_000_000_000)?;
    let seconds = started.elapsed().as_secs_f64();
    let mut digest = Sha256::new();
    // Every original class is read at each of these input-only controls.
    for length in 0..=11 {
        for start in 0..100 {
            let prefix: Vec<_> = (0..length)
                .map(|i| ((start + i * 17) % 100) as i64)
                .collect();
            for value in model.complex(&shape.addresses(&prefix)?)? {
                digest.update(value.re.to_bits().to_le_bytes());
                digest.update(value.im.to_bits().to_le_bytes());
            }
        }
    }
    let raw: String = digest
        .finalize()
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect();
    println!(
        "{}",
        json!({"shape":shape,"stored_columns":count,"reserved_bytes":reserved,
        "recovery_seconds":seconds,"all_complex_output_digest":raw,
        "reconstruction_error":model.reconstruction_error,"balance_error":model.balance_error})
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("REFUSED: {error}");
        std::process::exit(1);
    }
}
