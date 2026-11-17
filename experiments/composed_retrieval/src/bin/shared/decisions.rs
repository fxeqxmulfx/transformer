//! Compare original-vocabulary decisions directly from sealed raw f64 logits.
use baseline::channel::argmax;
use composed_retrieval::Result;

#[derive(Default, serde::Serialize)]
pub struct Decisions {
    pub rows: usize,
    pub different: usize,
    pub max_logit_delta: f64,
}
impl Decisions {
    pub fn add(&mut self, a: &[u8], b: &[u8], v: usize) -> Result<()> {
        if v == 0 || a.is_empty() || a.len() != b.len() || !a.len().is_multiple_of(8 * v) {
            return Err("unaligned raw logits".into());
        }
        let decode = |row: &[u8]| {
            row.as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b))
                .collect::<Vec<_>>()
        };
        for (a, b) in a.chunks_exact(v * 8).zip(b.chunks_exact(v * 8)) {
            let (a, b) = (decode(a), decode(b));
            if a.iter().chain(&b).any(|x| !x.is_finite()) {
                return Err("nonfinite raw score".into());
            }
            self.rows += 1;
            self.different += usize::from(argmax(&a) != argmax(&b));
            for (x, y) in a.iter().zip(b) {
                self.max_logit_delta = self.max_logit_delta.max((x - y).abs());
            }
        }
        Ok(())
    }
}
