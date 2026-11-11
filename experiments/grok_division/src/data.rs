//! The division equations of openai/grok `grok/data.py` at 3d64b1d8: the vocabulary of 239 tokens
//! (`<|eos|>`, `=`, the 20 operator names in sorted order, the 97 residues, the 120 elements of
//! S5), the 9312 equations `<|eos|> a / b = c <|eos|>` in the author's order, and the split
//! by NumPy's legacy shuffle.

use crate::mt::Mt;
use sha2::{Digest, Sha256};

/// The modulus.
pub const P: usize = 97;
/// The token `<|eos|>`.
pub const EOS: usize = 0;
/// The token `=`.
pub const EQ: usize = 1;
/// The token `/`: the ninth of the sorted operator names, which start at 2.
pub const DIV: usize = 10;
/// The token of the residue 0; the residue `r` is `NUM0 + r`.
pub const NUM0: usize = 22;
/// The position whose output predicts the answer `c`.
pub const ANSWER_POS: usize = 4;
/// The position whose output predicts the closing `<|eos|>`.
pub const EOS_POS: usize = 5;

/// The equation `num / den = quo` in the field of 97 elements, `den` nonzero.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Equation {
    /// The numerator `a`.
    pub num: u8,
    /// The denominator `b`, 1 to 96.
    pub den: u8,
    /// The answer `c = a / b`.
    pub quo: u8,
}

impl Equation {
    /// The seven tokens of the equation.
    pub fn tokens(&self) -> [usize; 7] {
        [EOS, NUM0 + self.num as usize, DIV, NUM0 + self.den as usize, EQ, NUM0 + self.quo as usize, EOS]
    }

    /// The token the output at `pos` is trained to give: the next token.
    pub fn target(&self, pos: usize) -> usize {
        self.tokens()[pos + 1]
    }
}

/// Every equation, in the order `_make_binary_operation_data("/")` lists them: the answer in
/// the outer loop, the denominator in the inner one, and the numerator their product.
pub fn equations() -> Vec<Equation> {
    let mut out = Vec::with_capacity(P * (P - 1));
    for quo in 0..P {
        for den in 1..P {
            out.push(Equation { num: (den * quo % P) as u8, den: den as u8, quo: quo as u8 });
        }
    }
    out
}

/// The split of `ArithmeticDataset.splits`: the equations shuffled by
/// `np.random.RandomState(seed).shuffle`, the first `train` of them the training set and the
/// rest the held-out set. The author's code always uses the seed 0.
pub fn split(seed: u32, train: usize) -> (Vec<Equation>, Vec<Equation>) {
    let mut all = equations();
    Mt::new(seed).shuffle(&mut all);
    let heldout = all.split_off(train);
    (all, heldout)
}

/// `round(len * (pct / 100))` as Python rounds it, to the even integer at a tie:
/// `calc_split_len`.
pub fn train_len(len: usize, pct: f64) -> usize {
    (len as f64 * (pct / 100.0)).round_ties_even() as usize
}

/// The SHA-256 of the token rows as little-endian 64-bit integers, in hex: the fingerprint the
/// repository's earlier port of the author's generator recorded.
pub fn fingerprint(rows: &[Equation]) -> String {
    let mut hasher = Sha256::new();
    for row in rows {
        for token in row.tokens() {
            hasher.update((token as i64).to_le_bytes());
        }
    }
    hex(&hasher.finalize())
}

/// Bytes in lowercase hex.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
