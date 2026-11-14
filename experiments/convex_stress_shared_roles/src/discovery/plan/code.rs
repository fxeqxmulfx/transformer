//! Port of kernel/discovery/plan.rs and readout_kernel/discovery.rs at
//! ec5f839d: transpose every complete basis direction and expand declared
//! answer groups back to the original raw observed class names.
use super::*;
impl Plan {
    pub fn code(&self, r: &Request, basis: &[Vec<u16>]) -> Result<Code> {
        if basis.len() > 8
            || basis
                .iter()
                .any(|v| v.len() != self.shape[1] || v.iter().any(|&x| x >= r.prime))
        {
            return Err("invalid restricted complete-basis shape".into());
        }
        let (nr, nc) = (self.names.rows.len(), self.names.columns.len());
        let vectors: Vec<Vec<u16>> = (0..self.shape[1])
            .map(|k| basis.iter().map(|v| v[k]).collect())
            .collect();
        let code = Code {
            prime: r.prime,
            dimension: basis.len(),
            names: self.names.clone(),
            rows: vectors[..nr].into(),
            columns: vectors[nr..nr + nc].into(),
            classes: vectors[nr + nc..].into(),
        };
        code.validate(r.input.n, r.answers)?;
        Ok(code)
    }
    pub fn expand(&self, r: &Request, code: &Code) -> Result<Code> {
        let raw = Code {
            prime: code.prime,
            dimension: code.dimension,
            names: Names {
                rows: code.names.rows.clone(),
                columns: code.names.columns.clone(),
                classes: self.raw_names.clone(),
            },
            rows: code.rows.clone(),
            columns: code.columns.clone(),
            classes: self
                .raw_names
                .iter()
                .map(|&c| {
                    code.label(r.readout[c])
                        .map(<[u16]>::to_vec)
                        .ok_or("raw class absent from restricted grouped code")
                })
                .collect::<std::result::Result<_, _>>()?,
        };
        raw.validate(r.input.n, r.input.vocabulary)?;
        Ok(raw)
    }
}
