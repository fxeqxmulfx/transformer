//! Immutable-version cache handles and original CUDA module loading.
use crate::{
    domain::Result,
    infrastructure::{gemm, ops, product, rotate},
    inputs::require,
};
use kernite_cuda::{Buffer, Cuda};
pub struct Modules {
    pub ops: ops::kernels::Kernels,
    pub gemm: gemm::kernels::Kernels,
    pub product: product::kernels::Kernels,
    pub rotate: rotate::kernels::Kernels,
}
impl Modules {
    pub fn load(cuda: &Cuda) -> Result<Self> {
        Ok(Self {
            ops: ops::kernels::Kernels::load(cuda)?,
            gemm: gemm::kernels::Kernels::load(cuda)?,
            product: product::kernels::Kernels::load(cuda)?,
            rotate: rotate::kernels::Kernels::load(cuda)?,
        })
    }
}
pub struct Weights {
    pub version: String,
    pub first: Buffer<f32>,
    pub second: Buffer<f32>,
    pub readout: Buffer<f32>,
}
impl Weights {
    pub fn bytes(&self) -> usize {
        4 * (self.first.len() + self.second.len() + self.readout.len())
    }
    pub fn validate(&self, version: &str) -> Result<()> {
        same_version(&self.version, version)
    }
}
pub fn same_version(stored: &str, current: &str) -> Result<()> {
    require(
        stored == current,
        "cache belongs to another parameter version",
    )
}
pub fn valid_pairs(pairs: &[[u32; 2]]) -> Result<()> {
    require(
        !pairs.is_empty() && pairs.len() <= 512 && pairs.iter().flatten().all(|&v| v < 97),
        "invalid serving operands",
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stale_versions_and_invalid_operands_are_rejected() {
        assert!(same_version("old", "new").is_err());
        assert!(same_version("new", "new").is_ok());
        assert!(valid_pairs(&[]).is_err());
        assert!(valid_pairs(&[[97, 0]]).is_err());
        assert!(valid_pairs(&vec![[1, 2]; 513]).is_err());
        assert!(valid_pairs(&[[96, 96]]).is_ok());
    }
}
