//! Actual head checks remain kernel/audit.rs at ec5f839d. Record-weighted
//! raw CE is separately audited with complex products, so repeated original
//! records are not silently deduplicated by the algebraic equation dictionary.
use crate::*;
pub use kernel::audit::complex_logits;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Audit {
    pub head: kernel::audit::Audit,
    pub original_records: usize,
    pub record_ce: f64,
    pub record_errors: usize,
}
pub fn run(
    model: &Model,
    input: &kernel::discovery::Input,
    records: &[records::Record],
) -> Result<Audit> {
    records::validate(input, records)?;
    if model.n != input.n || model.vocabulary != input.vocabulary {
        return Err("record audit dimensions differ from original input".into());
    }
    let head = model.audit(&input.observations)?;
    let encoded = records::encode(model, records)?;
    let (mut record_ce, mut record_errors) = (0., 0);
    for s in &encoded {
        let z = complex_logits(model, (s.block, s.state));
        let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        record_ce += z.iter().map(|x| (x - top).exp()).sum::<f64>().ln() + top - z[s.label];
        let best = (1..z.len()).fold(0, |best, c| if z[c] > z[best] { c } else { best });
        record_errors += usize::from(best != s.label);
    }
    record_ce /= encoded.len() as f64;
    if !close(record_ce, training::score(model, &encoded), 1e-8) {
        return Err("actual complex and realified record CE differ".into());
    }
    Ok(Audit {
        head,
        original_records: records.len(),
        record_ce,
        record_errors,
    })
}
pub(crate) fn close(a: f64, b: f64, tolerance: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance
}
/// Independent conjugate transform ported from kernel/audit.rs at ec5f839d.
pub(crate) fn transformed(model: &Model, table: &[f64]) -> Vec<f64> {
    let mut out = vec![0.; model.coefficients.len()];
    for b in &model.blocks {
        for f in 0..b.states() {
            for c in 0..model.vocabulary {
                let value: C = (0..b.states())
                    .map(|s| {
                        b.code[s * b.states() + f].conj()
                            * table[(b.offset + s) * model.vocabulary + c]
                    })
                    .sum();
                let k = 2 * (b.offset + f) * model.vocabulary;
                out[k + c] = value.re;
                out[k + model.vocabulary + c] = value.im;
            }
        }
    }
    out
}
