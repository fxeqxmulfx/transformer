//! Streaming coefficient persistence; no JSON-sized copy of a dense tensor.
use crate::independent::Independent;
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

#[derive(Serialize, Deserialize)]
struct FactorHeader {
    bank: Bank,
    radius: f64,
    probabilities: Vec<f64>,
}

pub fn save_independent(model: &Independent, mut out: impl Write) -> Result<()> {
    model.validate()?;
    let header = serde_json::to_vec(&FactorHeader {
        bank: model.bank.clone(),
        radius: model.radius,
        probabilities: model.probabilities.clone(),
    })?;
    out.write_all(b"CDINDP01")?;
    out.write_all(&(header.len() as u64).to_le_bytes())?;
    out.write_all(&header)?;
    for d in &model.decoder {
        out.write_all(&d.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

pub fn load_independent(mut input: impl Read) -> Result<Independent> {
    let mut bytes = [0u8; 8];
    input.read_exact(&mut bytes)?;
    if &bytes != b"CDINDP01" {
        return Err("invalid independent checkpoint magic".into());
    }
    input.read_exact(&mut bytes)?;
    let n = u64::from_le_bytes(bytes);
    if n > 16384 {
        return Err("oversized independent checkpoint header".into());
    }
    let mut raw = vec![0; n as usize];
    input.read_exact(&mut raw)?;
    let FactorHeader {
        bank,
        radius,
        probabilities,
    } = serde_json::from_slice(&raw)?;
    let mut model = Independent::new(bank, radius)?;
    model.probabilities = probabilities;
    // Reject invalid routing before reading a potentially large decoder.
    model.validate()?;
    for d in &mut model.decoder {
        input.read_exact(&mut bytes)?;
        *d = f64::from_le_bytes(bytes);
    }
    if input.read(&mut bytes[..1])? != 0 {
        return Err("trailing independent checkpoint bytes".into());
    }
    model.validate()?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{Mask, Transport};
    #[test]
    fn independent_checkpoint_preserves_factors_and_rejects_invalid_routing() -> Result<()> {
        let bank = Bank {
            vocab: 3,
            depth: 2,
            history_slots: 8,
            masks: vec![Mask::All],
            transport: Transport::Fourier,
            execution: crate::features::Execution::Composed,
        };
        let mut model = Independent::new(bank, 10.)?;
        model.probabilities = vec![0.2, 0.7];
        model.decoder[4] = -0.125;
        let mut bytes = vec![];
        save_independent(&model, &mut bytes)?;
        let restored = load_independent(bytes.as_slice())?;
        assert_eq!(restored.decoder, model.decoder);
        assert_eq!(restored.probabilities, model.probabilities);
        assert!(load_independent(&bytes[..bytes.len() - 1]).is_err());
        bytes.push(0);
        assert!(load_independent(bytes.as_slice()).is_err());
        model.probabilities[0] = 1.01;
        assert!(save_independent(&model, Vec::new()).is_err());
        Ok(())
    }
    #[test]
    fn binary_roundtrip_is_exact_and_corrupted_models_are_rejected() -> Result<()> {
        let bank = Bank {
            vocab: 3,
            depth: 0,
            history_slots: 8,
            masks: vec![Mask::All],
            transport: Transport::Fourier,
            execution: crate::features::Execution::Composed,
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
