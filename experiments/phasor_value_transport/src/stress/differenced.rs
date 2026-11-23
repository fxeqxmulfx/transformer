//! Data-only port of DiffGPT 5a37c515's div97.division_rows and make_tensors.
//! Both positive-order LGT grids have unit scale and a fixed zero prefix.
//! See Transformer.Phasor.DifferencedDivisionScore for the mask and inverse.
use arithmetic::reference::{SplitMix64, fingerprint, power};

pub(super) struct Source {
    pub vocabulary: usize,
    pub train: Vec<[u32; 3]>,
    pub validation: Vec<[u32; 3]>,
    pub fingerprints: [String; 2],
}

pub(super) fn build(order: u32, seed: u64) -> arithmetic::Result<Source> {
    // tests/differenced.rs reconstructs every token with the source codec.
    let (vocabulary, expected) = match order {
        1 => (
            256,
            [
                "b9e2f8d0763634c34b210d82b1b52dcf2812da9f6bdeefb64e53a67faa7bffd5",
                "e32d62e6a6dbd96f0183f0cd50d7832b4ae447f9b815f7d97affd6af0fe72fcb",
            ],
        ),
        2 => (
            512,
            [
                "d300b8724625d08429cf7792a53ba26388a7444bd0ecb45d42cbdc39a094900f",
                "536d02dea65e740ea7105604371a61e896a414b11adb0140cbef3f79a1643b3f",
            ],
        ),
        _ => return Err("expected first- or second-order differenced division".into()),
    };
    if seed != 1 {
        return Err("differenced division requires original data seed one".into());
    }
    let offset = vocabulary / 2;
    let mut rows = Vec::with_capacity(97 * 96);
    for a in 0..97 {
        for b in 1..97 {
            let q = a * power(b, 95, 97) % 97;
            rows.push([
                offset + a,
                offset + b - order * a,
                offset + q + order * (order - 1) / 2 * a - order * b,
            ]);
        }
    }
    SplitMix64(seed ^ 0x7370_6c69_7473_0002).shuffle(&mut rows);
    let validation = rows.split_off(rows.len() / 2);
    let fingerprints = [fingerprint(&rows), fingerprint(&validation)];
    if fingerprints != expected {
        return Err("differenced-division tensors differ from the source codec".into());
    }
    Ok(Source {
        vocabulary: vocabulary as usize,
        train: rows,
        validation,
        fingerprints,
    })
}
