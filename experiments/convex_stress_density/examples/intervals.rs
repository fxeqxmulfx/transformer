//! Exact/tie and genuinely perturbed interval fixtures, never original tasks.
use convex_stress_density::{
    C, Result,
    interval::{Input, Options},
    interval_audit,
};
use std::{fs, path::PathBuf};

fn main() -> Result<()> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: intervals FRESH_DIRECTORY")?,
    );
    if dir.exists() {
        return Err("interval fixture destination exists".into());
    }
    fs::create_dir_all(&dir)?;
    for (name, n, frequencies, scale, epsilon, iterations) in [
        ("exact", 4, vec![0, 1], 1., 0., 16_384),
        ("tie", 13, vec![1, 12], 1., 0., 16_384),
        ("perturbed_exact_refusal", 4, vec![1], 1.1, 0., 64),
        ("perturbed_interval", 4, vec![1], 1.1, 0.21, 64),
    ] {
        let input = Input {
            n,
            rank: frequencies.len(),
            rows: [1, 0],
            features: frequencies
                .iter()
                .flat_map(|k| {
                    (0..n).map(move |x| {
                        C::from_polar(scale, std::f64::consts::TAU * (x * k) as f64 / n as f64)
                    })
                })
                .collect(),
            options: Options {
                model_tolerance: epsilon,
                iterations,
                ..Options::default()
            },
        };
        let out = interval_audit::run(input.clone())?;
        interval_audit::replay(&input, &out)?;
        fs::write(
            dir.join(format!("{name}.input.json")),
            serde_json::to_vec_pretty(&input)?,
        )?;
        fs::write(
            dir.join(format!("{name}.artifact.json")),
            serde_json::to_vec_pretty(&out)?,
        )?;
        println!(
            "{}",
            serde_json::json!({"fixture":name,"audit":out.raw_audit,
            "buffer_bound_bytes":out.outcome.buffer_bound_bytes})
        );
    }
    Ok(())
}
