//! Exact-geometry fixtures only. Latent frequencies are used to generate the
//! input, then withheld from the numerical solver and its replay/audit.
use convex_stress_density::{
    C, Result,
    artifact::{self, Input},
    solver::Options,
};
use std::{fs, path::PathBuf};

fn main() -> Result<()> {
    let output = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: fixtures FRESH_DIRECTORY")?,
    );
    if output.exists() {
        return Err("fixture destination exists".into());
    }
    fs::create_dir_all(&output)?;
    for (name, frequencies, permuted) in [
        ("rotated", vec![1, 3, 9], true),
        ("real_tie", vec![1, 12], false),
    ] {
        let (n, r) = (13, frequencies.len());
        let mut features = vec![C::default(); n * r];
        for j in 0..r {
            for x in 0..n {
                let state = if permuted { (5 * x + 3) % n } else { x };
                for (i, &frequency) in frequencies.iter().enumerate() {
                    let rotation = match (i, j) {
                        (0, 0) | (1, 1) => 0.6,
                        (1, 0) => 0.8,
                        (0, 1) => -0.8,
                        _ => f64::from(i == j),
                    };
                    features[x + n * j] += C::from_polar(
                        rotation,
                        std::f64::consts::TAU * (state * frequency) as f64 / n as f64
                            + 0.37 * (j + 1) as f64,
                    );
                }
            }
        }
        let input = Input {
            n,
            rank: r,
            features,
            rows: [0, 1],
            options: Options::default(),
        };
        let out = artifact::run(input.clone())?;
        artifact::replay(&input, &out)?;
        fs::write(
            output.join(format!("{name}.input.json")),
            serde_json::to_vec_pretty(&input)?,
        )?;
        fs::write(
            output.join(format!("{name}.artifact.json")),
            serde_json::to_vec_pretty(&out)?,
        )?;
        println!(
            "{}",
            serde_json::json!({"fixture":name,"accepted":out.extraction.accepted,
            "stage_iterations":out.extraction.stages.iter().map(|s|s.trace.len()).collect::<Vec<_>>(),
            "purity_error":out.extraction.purity_error,"diagonal_residual":out.extraction.diagonal_residual,
            "raw_audit":out.raw_audit})
        );
    }
    Ok(())
}
