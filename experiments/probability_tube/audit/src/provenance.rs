//! Content fingerprints are independent of the trajectory runner's fingerprints.
use crate::{
    Result,
    evidence::{hash, require},
    infrastructure,
};
use serde_json::{Value, json};
pub fn source_hash() -> String {
    env!("REPRO_SOURCE_SHA256").to_owned()
}
pub fn metadata() -> Result<Value> {
    let cuda = kernite_cuda::Cuda::new()?;
    let ptx = hash(
        [
            infrastructure::ops::kernels::PTX,
            infrastructure::rotate::kernels::PTX,
            infrastructure::gemm::kernels::PTX,
            infrastructure::product::kernels::PTX,
        ]
        .concat()
        .as_bytes(),
    );
    require(
        ptx == "dc4e40d319937fb5175b041a0da5edca8613e9310ec7875e7e61f149131199ac",
        "source PTX changed",
    )?;
    Ok(
        json!({"gpu":cuda.name(),"rustc":env!("REPRO_RUSTC"),"source_sha256":source_hash(),
        "source_ledger":include_str!(concat!(env!("OUT_DIR"),"/sources.tsv")),"source_ptx_sha256":ptx,
        "executable_sha256":hash(&std::fs::read(std::env::current_exe()?)?),
        "trainer_sha256":env!("TRAINER_SOURCE_SHA256"),"protocol":include_str!("../PROTOCOL.md")}),
    )
}
