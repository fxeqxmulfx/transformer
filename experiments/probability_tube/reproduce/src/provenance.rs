//! Execution identity and streaming artifact hashes for a new reproduction.
use crate::{
    Result,
    infrastructure::{self, storage},
    inputs::require,
    schedule,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Read,
    path::Path,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};
pub fn source_hash() -> String {
    storage::hash(
        &[
            env!("REPRO_SOURCE_SHA256").as_bytes(),
            &schedule::PEAK.to_le_bytes(),
            schedule::NAME.as_bytes(),
        ]
        .concat(),
    )
}
pub fn cohort() -> Result<&'static str> {
    match (schedule::NAME, schedule::PEAK) {
        ("main_wsd", 0.001) => Ok("lr001"),
        ("main_wsd", 0.01) => Ok("lr01"),
        ("early_settle_1000_4000_hold_1e-4", 0.01) => Ok("settle"),
        _ => Err("undeclared reproduction cohort".into()),
    }
}
pub fn file_hash(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
pub fn unix_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
pub fn metadata() -> Result<Value> {
    let cuda = kernite_cuda::Cuda::new()?;
    let gpu = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ])
        .output()?;
    require(gpu.status.success(), "nvidia-smi provenance query failed")?;
    let ptx = storage::hash(
        [
            infrastructure::ops::kernels::PTX,
            infrastructure::rotate::kernels::PTX,
            infrastructure::gemm::kernels::PTX,
            infrastructure::product::kernels::PTX,
        ]
        .concat()
        .as_bytes(),
    );
    // Unchanged workspace/kernel namespaces retain the pinned source PTX too.
    require(
        ptx == "dc4e40d319937fb5175b041a0da5edca8613e9310ec7875e7e61f149131199ac",
        "original PTX mismatch",
    )?;
    Ok(
        json!({"source_sha256":source_hash(),"source_ledger":include_str!(concat!(env!("OUT_DIR"),"/sources.tsv")),
        "trainer_sha256":env!("TRAINER_SOURCE_SHA256"),"trainer_commit":"b70811e0e0e7ebdc4a8a2e48044c180332f7c40d",
        "kernite_commit":"a1b6bc2c956e1a5f35960e26b209e1b6c213096c","rustc":env!("REPRO_RUSTC"),
        "gpu":cuda.name(),"nvidia_smi":String::from_utf8(gpu.stdout)?.trim(),"source_ptx_sha256":ptx,
        "optimizer_ptx_sha256":storage::hash(crate::optimizer_kernels::kernels::PTX.as_bytes()),
        "executable_sha256":file_hash(&std::env::current_exe()?)?,"cohort":cohort()?,"schedule":schedule::NAME,
        "peak_lr":schedule::PEAK,"protocol":include_str!("../PROTOCOL.md"),"parent_protocol":include_str!("../../PROTOCOL.md")}),
    )
}
