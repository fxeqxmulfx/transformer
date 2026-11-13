//! Serial observed-only catalogue. Original full35 integration is a separate step.
use convex_stress_quotient::{
    self as component, Result, binding, certificate, row_program, training,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path};
pub mod artifacts;
mod execute;
mod ledger;
#[cfg(test)]
mod tests;
mod sources {
    include!(concat!(env!("OUT_DIR"), "/sources.rs"));
}
use artifacts::{digest, hash, read, write};
pub use execute::{replay, run};
use ledger::{Attempt, Ledger};
const PROTOCOL: &str = include_str!("../protocol.json");
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Configuration {
    pub input: component::Input,
    pub training: [training::Policy; 3],
    pub prefix: row_program::Policy,
    pub certificate: certificate::Policy,
    pub readout: Option<Vec<usize>>,
}
impl Configuration {
    fn request(&self, prime: u16) -> binding::Request {
        binding::Request {
            discovery: component::Request {
                input: self.input.clone(),
                prime,
            },
            training: self.training.clone(),
            prefix_policy: self.prefix.clone(),
            certificate_policy: self.certificate.clone(),
            readout: self.readout.clone(),
        }
    }
}
pub fn fingerprint() -> String {
    let mut bytes = vec![];
    for &(name, source) in sources::SOURCES {
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    for source in [
        convex_stress_decode::cohort::fingerprint(),
        convex_stress_relabel::cohort::fingerprint(),
    ] {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source.as_bytes());
    }
    hash(&bytes)
}
fn commit(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|c| c.is_ascii_hexdigit())
}
fn metadata(config: &Configuration, producer: &str) -> Result<Value> {
    if !commit(producer) {
        return Err("catalogue producer must be a full commit".into());
    }
    Ok(
        json!({"producer_commit":producer,"source_fingerprint":fingerprint(),
        "binary_sha256":digest(&std::env::current_exe()?)?,"protocol":serde_json::from_str::<Value>(PROTOCOL)?,
        "configuration_sha256":hash(&serde_json::to_vec(config)?),"primes":component::catalogue()}),
    )
}
