//! Self-contained extraction artifacts bind original observable inputs to the
//! saved endpoint. Replay also compares a separately supplied original input.
use crate::{
    C, Result, audit,
    solver::{self, Extraction, Options},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Input {
    pub n: usize,
    pub rank: usize,
    pub features: Vec<C>,
    pub rows: [usize; 2],
    pub options: Options,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Artifact {
    pub format: u32,
    pub input: Input,
    pub extraction: Extraction,
    pub raw_audit: Value,
}
pub fn run(input: Input) -> Result<Artifact> {
    let extraction = solver::extract(
        &input.features,
        input.n,
        input.rank,
        input.rows,
        input.options,
    )?;
    let raw_audit = audit::endpoint(&input.features, &extraction)?;
    Ok(Artifact {
        format: 1,
        input,
        extraction,
        raw_audit,
    })
}
pub fn replay(original: &Input, saved: &Artifact) -> Result<()> {
    if saved.format != 1 || original != &saved.input {
        return Err("artifact input differs from supplied original".into());
    }
    let e = &saved.extraction;
    if (e.n, e.rank, e.rows, e.options)
        != (original.n, original.rank, original.rows, original.options)
    {
        return Err("artifact extraction envelope differs from original".into());
    }
    solver::replay(&original.features, e)?;
    if audit::endpoint(&original.features, e)? != saved.raw_audit {
        return Err("raw audit replay mismatch".into());
    }
    Ok(())
}
