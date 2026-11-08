//! Fixed local boundaries and prototypes from PROTOCOL.md, following the
//! pairwise reformulation in ForwardForward.Loss (37c9c711). Prototypes are
//! independent random constants; no learned initialization or task structure.
use crate::domain::{
    Result,
    data::Random,
    model::{Arm, W},
};
use kernite_cuda::{Buffer, Cuda};

pub const STEPS: usize = 4000;
pub const CADENCE: usize = 100;

/// First moments have the signed weight ties; second moments have unsigned
/// ties. This is the source coordinate AdamW storage contract.
pub fn valid_state(arm: Arm, state: &crate::application::State) -> bool {
    use crate::domain::model;
    model::valid(arm, &state.parameters)
        && model::valid(arm, &state.first)
        && state.second.len() == arm.count()
        && state.second.iter().all(|&v| v.is_finite() && v >= 0.)
        && (arm == Arm::Swiglu
            || model::pairs().iter().all(|&[ar, bi, br, ai, kind]| {
                kind == 1
                    || (state.second[ar as usize] == state.second[ai as usize]
                        && state.second[bi as usize] == state.second[br as usize])
            }))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Ce,
    Pair,
    Auxiliary,
    Local,
}
impl Method {
    pub const ALL: [Self; 4] = [Self::Ce, Self::Pair, Self::Auxiliary, Self::Local];
    pub fn name(self) -> &'static str {
        match self {
            Self::Ce => "ce-bp",
            Self::Pair => "pair-bp",
            Self::Auxiliary => "auxiliary-bp",
            Self::Local => "local-pair",
        }
    }
    pub fn auxiliary(self) -> bool {
        matches!(self, Self::Auxiliary | Self::Local)
    }
}

pub struct Prototypes {
    pub vocabulary: Vec<f32>,
    pub hidden: Vec<f32>,
}
fn codes(seed: u64, width: usize) -> Vec<f32> {
    let mut values = Random(seed).normals(97 * width, 1.);
    for row in values.chunks_exact_mut(width) {
        let norm = row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        for value in row {
            *value = (*value as f64 / norm) as f32;
        }
    }
    values
}
pub fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * cols);
    (0..cols)
        .flat_map(|c| (0..rows).map(move |r| x[r * cols + c]))
        .collect()
}
impl Prototypes {
    pub fn new(arm: Arm, seed: u64) -> Self {
        Self {
            vocabulary: codes(seed ^ 0x666670726f746f30, W),
            hidden: codes(seed ^ 0x666670726f746f31, arm.hidden()),
        }
    }
}

pub struct Aux {
    pub r0: Buffer<f32>,
    pub r0t: Buffer<f32>,
    pub r1: Buffer<f32>,
    pub r1t: Buffer<f32>,
    pub norm: Buffer<f32>,
    pub grad: Buffer<f32>,
    pub saved_a: Option<Buffer<f32>>,
}
impl Aux {
    pub fn new(cuda: &Cuda, arm: Arm, p: &Prototypes, method: Method) -> Result<Self> {
        assert!(method.auxiliary());
        Ok(Self {
            r0: cuda.upload(&p.vocabulary)?,
            r0t: cuda.upload(&transpose(&p.vocabulary, 97, W))?,
            r1: cuda.upload(&p.hidden)?,
            r1t: cuda.upload(&transpose(&p.hidden, 97, arm.hidden()))?,
            norm: cuda.alloc(512 * arm.hidden())?,
            grad: cuda.alloc(512 * arm.hidden())?,
            saved_a: if method == Method::Auxiliary {
                Some(cuda.alloc(2 * arm.hidden() * W)?)
            } else {
                None
            },
        })
    }
    pub fn bytes(&self) -> usize {
        4 * (self.r0.len()
            + self.r0t.len()
            + self.r1.len()
            + self.r1t.len()
            + self.norm.len()
            + self.grad.len()
            + self.saved_a.as_ref().map_or(0, |x| x.len()))
    }
}

pub struct Kernels {
    pub loss: crate::kernel::kernels::Kernels,
    pub norm: crate::local_norm::kernels::Kernels,
}
impl Kernels {
    pub fn new(cuda: &Cuda) -> Result<Self> {
        Ok(Self {
            loss: crate::kernel::kernels::Kernels::load(cuda)?,
            norm: crate::local_norm::kernels::Kernels::load(cuda)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_heads_are_reproducible_unit_rows_and_transpose_exactly() {
        for arm in Arm::ALL {
            let a = Prototypes::new(arm, 7);
            let b = Prototypes::new(arm, 7);
            assert_eq!(a.vocabulary, b.vocabulary);
            assert_eq!(a.hidden, b.hidden);
            assert_ne!(a.hidden, Prototypes::new(arm, 11).hidden);
            for (p, w) in [(&a.vocabulary, W), (&a.hidden, arm.hidden())] {
                for row in p.chunks_exact(w) {
                    let n = row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
                    assert!((n - 1.).abs() < 1e-7);
                }
                assert_eq!(transpose(&transpose(p, 97, w), w, 97), *p);
            }
        }
    }
}
