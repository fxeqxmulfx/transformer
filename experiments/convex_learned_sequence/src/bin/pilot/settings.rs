//! Explicit small complete-family pilot. These are reproducibility settings,
//! not a stress-suite architecture or a task program supplied to inference.

use convex_learned_sequence::domain::DomainLimits;
use convex_learned_sequence::geometry::GeometryLimits;
use convex_learned_sequence::model::forward_learning::{ForwardConfig, ForwardLimits};
use convex_learned_sequence::model::learning::{MiniLimits, Record};
use convex_learned_sequence::model::{ComplexModel, ModelLimits, Renaming};
use convex_learned_sequence::{ProjectionLimits, Rational};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const EPOCHS: usize = 8;
pub const BATCH: usize = 2;
pub const MODES: [&str; 2] = ["mini", "forward"];

pub fn ratio(n: i64, d: i64) -> Rational {
    Rational::new(n.into(), d.into())
}
pub fn rename() -> Renaming {
    Renaming::new(vec![0, 2, 1]).expect("declared full involution")
}
pub fn model() -> Result<ComplexModel> {
    Ok(ComplexModel::zero(
        3,
        1,
        ModelLimits {
            domain: DomainLimits {
                max_prefixes: 4,
                max_patterns: 16,
                max_real_parameters: 768,
                max_row_entries: 16,
            },
            projection: ProjectionLimits {
                max_active_sets: 16,
                max_matrix_entries: 1024,
                max_coefficient_bits: 4096,
            },
            max_working_bytes: 1 << 30,
        },
    )?)
}
pub fn mini_limits() -> MiniLimits {
    MiniLimits {
        max_records: BATCH,
        max_pullback_terms: 3072,
        max_projection_candidates: 3072,
        max_working_bytes: 1 << 30,
    }
}
pub fn forward_config() -> ForwardConfig {
    ForwardConfig {
        rate: ratio(1, 32),
        mixture: ratio(1, 1024),
        query_cap: ratio(1, 16384),
    }
}
pub fn forward_limits() -> ForwardLimits {
    ForwardLimits {
        geometry: GeometryLimits {
            max_normal_projections: 4,
            max_active_candidates: 64,
            max_rational_entries: 1024,
            max_working_bytes: 1 << 28,
        },
        max_records: BATCH,
        max_geometry_builds: 32,
        max_active_candidates: 5120,
        max_batch_evaluations: 1539,
        max_forward_passes: 6156,
        max_forward_terms: 4_727_808,
        max_constraint_terms: 4_724_736,
        max_working_bytes: 1 << 30,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub tokens: Vec<usize>,
    pub target: usize,
}
pub fn records(inputs: &[Input]) -> Result<Vec<Record>> {
    if inputs.is_empty() || inputs.len() > 100 {
        return Err("empty or oversized input cohort".into());
    }
    let checked = model()?;
    inputs
        .iter()
        .map(|x| {
            checked.domain().input_index(&x.tokens)?;
            if x.target >= 3 {
                return Err("invalid original target object".into());
            }
            Ok(Record {
                tokens: x.tokens.clone(),
                target: x.target,
            })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub version: usize,
    pub revision: String,
    pub source_sha: String,
    pub objects: usize,
    pub context: usize,
    pub epochs: usize,
    pub batch: usize,
    pub renaming: Vec<usize>,
    pub rate: String,
    pub mixture: String,
    pub query_cap: String,
}
impl Config {
    pub fn new(revision: &str, source_sha: &str) -> Result<Self> {
        if !hex(revision, 40) || !hex(source_sha, 64) {
            return Err("invalid source identity".into());
        }
        Ok(Self {
            version: 1,
            revision: revision.into(),
            source_sha: source_sha.into(),
            objects: 3,
            context: 1,
            epochs: EPOCHS,
            batch: BATCH,
            renaming: vec![0, 2, 1],
            rate: "1/32".into(),
            mixture: "1/1024".into(),
            query_cap: "1/16384".into(),
        })
    }
}
pub fn hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}
pub fn load_bytes(path: &Path, max: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(max + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max {
        return Err("artifact exceeds admitted size".into());
    }
    Ok(bytes)
}
pub fn load<T: DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&load_bytes(path, 64 << 20)?)?)
}
pub fn save<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    if bytes.len() > 64 << 20 {
        return Err("artifact exceeds admitted size".into());
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}
pub fn hash(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(load_bytes(path, 64 << 20)?)))
}
fn command(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    if !output.status.success() {
        return Err(format!("resource query failed: {program}").into());
    }
    Ok(String::from_utf8(output.stdout)?)
}
pub fn guard() -> Result<()> {
    for (path, minimum) in [
        ("/home/misha/lean_projects/transformer", 2u128 << 30),
        ("/run/media/misha/storage", 16u128 << 30),
    ] {
        let result = command("df", &["-B1", "--output=avail", path])?;
        let available: u128 = result
            .lines()
            .nth(1)
            .ok_or("missing disk capacity")?
            .trim()
            .parse()?;
        if available < minimum {
            return Err(format!("insufficient free disk space: {path}").into());
        }
    }
    let rss = command("ps", &["-e", "-o", "rss="])?;
    let mut bytes = 0u128;
    for line in rss.lines() {
        bytes = bytes
            .checked_add(
                line.trim()
                    .parse::<u128>()?
                    .checked_mul(1024)
                    .ok_or("RSS overflow")?,
            )
            .ok_or("RSS overflow")?;
    }
    if bytes + (4u128 << 30) > 110_000_000_000 {
        return Err("aggregate RAM reservation exceeds 110 GB".into());
    }
    Ok(())
}
pub fn mean_loss(model: &ComplexModel, records: &[Record]) -> Result<f64> {
    let mut total = 0.0;
    for record in records {
        total += model
            .paired_forward(&record.tokens, &rename())?
            .cross_entropy(record.target)?;
    }
    if records.is_empty() || !total.is_finite() {
        return Err("invalid mean CE".into());
    }
    Ok(total / records.len() as f64)
}
