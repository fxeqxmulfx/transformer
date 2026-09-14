//! The released artefacts: the schedule behind them, and what they hash to.
//!
//! `plan.yaml` is an input — the one file of the release that is not a build
//! product — so it is in this repository and compiled in here.  `model.bin`
//! and the eighteen `transformer_vm/data/*.txt` are outputs, ten megabytes of
//! them, and are not.  What stands in for them is their SHA-256: a test builds
//! the artefact itself and compares the digest, which is byte identity by
//! another name and costs 1.6 KB.  Nothing here needs the vendored checkout.

/// The released schedule, `transformer-vm/plan.yaml`, verbatim.
pub const PLAN: &str = include_str!("../../reference/plan.yaml");

/// `sha256sum` over the release's build products, one `digest  name` a line.
const SUMS: &str = include_str!("../../reference/sha256sums");

/// What the released `name` hashes to: `"model.bin"`, `"data/hello.txt"`.
pub fn released_digest(name: &str) -> Option<&'static str> {
    SUMS.lines().filter_map(|l| l.split_once("  ")).find(|&(_, n)| n == name).map(|(d, _)| d)
}

/// The round constants: the first thirty-two bits of the fractional parts of
/// the cube roots of the first sixty-four primes (FIPS 180-4 section 4.2.2).
#[rustfmt::skip]
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// The initial hash: the same of the square roots of the first eight primes.
#[rustfmt::skip]
const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// One block through the compression function (FIPS 180-4 section 6.2.2).
fn compress(h: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (i, c) in block.chunks_exact(4).enumerate() {
        w[i] = u32::from_be_bytes(c.try_into().unwrap());
    }
    for i in 16..64 {
        let (a, b) = (w[i - 15], w[i - 2]);
        let s0 = a.rotate_right(7) ^ a.rotate_right(18) ^ (a >> 3);
        let s1 = b.rotate_right(17) ^ b.rotate_right(19) ^ (b >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }
    let mut v = *h;
    for i in 0..64 {
        let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
        let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
        let t1 = v[7]
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
        let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
        let t2 = s0.wrapping_add(maj);
        v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
    }
    for (x, y) in h.iter_mut().zip(v) {
        *x = x.wrapping_add(y);
    }
}

/// SHA-256 of `bytes`, in lowercase hex — the same string `sha256sum` prints.
pub fn sha256(bytes: &[u8]) -> String {
    let mut h = H0;
    let mut blocks = bytes.chunks_exact(64);
    for block in blocks.by_ref() {
        compress(&mut h, block.try_into().unwrap());
    }
    // The tail: a one bit, zeroes, and the length in bits, which needs a
    // second block when the remainder leaves no eight bytes for it.
    let rest = blocks.remainder();
    let mut tail = [0u8; 128];
    tail[..rest.len()].copy_from_slice(rest);
    tail[rest.len()] = 0x80;
    let n = if rest.len() < 56 { 64 } else { 128 };
    tail[n - 8..n].copy_from_slice(&(8 * bytes.len() as u64).to_be_bytes());
    for block in tail[..n].chunks_exact(64) {
        compress(&mut h, block.try_into().unwrap());
    }
    h.iter().fold(String::with_capacity(64), |mut s, x| {
        use std::fmt::Write;
        let _ = write!(s, "{x:08x}");
        s
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three vectors of FIPS 180-4 appendix B — the last of them a
    /// million bytes, checked by its prefix — plus a string of exactly one
    /// block, where the padding has to spill into a second.
    #[test]
    fn the_digest_is_the_one_in_the_standard() {
        let cases = [
            ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            ("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            (
                "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
            (
                "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno",
                "2ff100b36c386c65a1afc462ad53e25479bec9498ed00aa5a04de584bc25301b",
            ),
        ];
        for (input, want) in cases {
            assert_eq!(sha256(input.as_bytes()), want, "{input:?}");
        }
        assert_eq!(sha256(&vec![b'a'; 1_000_000])[..16], *"cdc76e5c9914fb92");
    }

    /// The manifest covers what the release builds, and nothing is malformed.
    #[test]
    fn the_manifest_names_the_model_and_the_eighteen_data_files() {
        let names: Vec<&str> = SUMS.lines().filter_map(|l| l.split_once("  ")).map(|(_, n)| n).collect();
        assert_eq!(names.len(), 19, "model.bin and six programs times three files");
        assert_eq!(SUMS.lines().count(), 19, "every line parses");
        for (d, _) in SUMS.lines().filter_map(|l| l.split_once("  ")) {
            assert_eq!(d.len(), 64);
            assert!(d.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
        }
        assert!(released_digest("model.bin").is_some());
        for p in ["hello", "addition", "collatz", "fibonacci", "min_cost_matching", "sudoku"] {
            for s in [".txt", "_spec.txt", "_ref.txt"] {
                assert!(released_digest(&format!("data/{p}{s}")).is_some(), "{p}{s}");
            }
        }
    }

    /// `plan.yaml` is compiled in, so this cannot skip: the released schedule
    /// is seven layers wide and the port reads it without the checkout.
    #[test]
    fn the_released_plan_is_compiled_in() {
        assert!(PLAN.starts_with("summary:\n  layers: 7\n"), "{:?}", &PLAN[..PLAN.len().min(40)]);
        assert_eq!(PLAN.matches("\n- layer: ").count(), 7);
    }
}
