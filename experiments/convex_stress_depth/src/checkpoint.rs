//! Streaming coefficient persistence; no JSON-sized copy of a dense tensor.
use crate::{Result, features::Bank, head::Head};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

#[derive(Serialize, Deserialize)]
struct Header {
    bank: Bank,
    radius: f64,
}

pub fn save(model: &Head, mut out: impl Write) -> Result<()> {
    model.validate()?;
    let header = serde_json::to_vec(&Header {
        bank: model.bank.clone(),
        radius: model.radius,
    })?;
    out.write_all(b"CDHEAD01")?;
    out.write_all(&(header.len() as u64).to_le_bytes())?;
    out.write_all(&header)?;
    for &w in &model.coefficients {
        out.write_all(&w.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

pub fn load(mut input: impl Read) -> Result<Head> {
    let mut bytes = [0u8; 8];
    input.read_exact(&mut bytes)?;
    if &bytes != b"CDHEAD01" {
        return Err("invalid head checkpoint magic".into());
    }
    input.read_exact(&mut bytes)?;
    let n = u64::from_le_bytes(bytes);
    if n > 16384 {
        return Err("oversized checkpoint header".into());
    }
    let mut header = vec![0u8; n as usize];
    input.read_exact(&mut header)?;
    let Header { bank, radius } = serde_json::from_slice(&header)?;
    let mut model = Head::zero(bank, radius)?;
    for w in &mut model.coefficients {
        input.read_exact(&mut bytes)?;
        *w = f64::from_le_bytes(bytes);
    }
    if input.read(&mut bytes[..1])? != 0 {
        return Err("trailing head checkpoint bytes".into());
    }
    model.validate()?;
    Ok(model)
}

pub fn digest(mut input: impl Read) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{Mask, Transport};
    #[test]
    fn binary_roundtrip_is_exact_and_corrupted_models_are_rejected() -> Result<()> {
        let bank = Bank {
            vocab: 3,
            depth: 0,
            history_slots: 8,
            masks: vec![Mask::All],
            transport: Transport::Fourier,
        };
        let mut model = Head::zero(bank, 3.)?;
        model.coefficients[0] = 0.1;
        model.coefficients[4] = -0.7;
        let mut bytes = vec![];
        save(&model, &mut bytes)?;
        assert_eq!(load(bytes.as_slice())?.coefficients, model.coefficients);
        assert!(load(&bytes[..bytes.len() - 1]).is_err());
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(load(extra.as_slice()).is_err());
        let header = u64::from_le_bytes(bytes[8..16].try_into()?) as usize;
        bytes[16 + header..24 + header].copy_from_slice(&4f64.to_le_bytes());
        assert!(load(bytes.as_slice()).is_err());
        bytes[16 + header..24 + header].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(load(bytes.as_slice()).is_err());
        Ok(())
    }
}
