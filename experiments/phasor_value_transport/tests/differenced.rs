//! Compare both original dense splits with DiffGPT's public LGT encoder and decoder.
#[path = "../src/stress/differenced.rs"]
mod differenced;
use arithmetic::{hash, reference};
use source_codec::{Difference, Domain, Matrix, Tokenizer, decode, encode};
use std::{fs, path::Path};

/// Enumerate the division graph by testing the defining equation directly.
fn equations(seed: u64) -> Vec<[u32; 3]> {
    let mut rows = Vec::with_capacity(9312);
    for a in 0..97 {
        for b in 1..97 {
            let q = (0..97).find(|&q| b * q % 97 == a).unwrap();
            rows.push([a, b, q]);
        }
    }
    reference::SplitMix64(seed ^ 0x73706c6974730002).shuffle(&mut rows);
    rows
}

fn raw_hash(rows: &[[u32; 3]]) -> String {
    let bytes: Vec<_> = rows
        .iter()
        .flatten()
        .flat_map(|&x| i64::from(x).to_le_bytes())
        .collect();
    hash(&bytes)
}

#[test]
fn original_browser_checksums_fix_both_halves_of_the_shuffle() {
    // BROWSER_ROWS in DiffGPT tests/diff_gpt/application/grokking/test_div97.py.
    let fixtures = [
        (
            11,
            "af1518d3d6d9f078a11a6cfcdd0ed6318c2386ea41a9b6d9826d4064e1c1b01e",
            "fabbebaf7ab9a960308e775938abe42b8f5ced24da85f69f9890c078467c94ec",
        ),
        (
            17,
            "0ae63eaf37fd8c32ecf9ccff2790788fd3db7de4d2da6c253094ba1f19362bc8",
            "3bc9435f860665c066037288312ef09fbc6399e6bc1ff1fdd970be9242f72044",
        ),
        (
            23,
            "bfc8141de1293be4f60493ae24169a4b691cf46c1c86d9be3ecdd3e32fee08da",
            "08af43e6607898c94c01940c819b357b201ee74bdc019c776c0be6f4e50fc90e",
        ),
    ];
    for (seed, train, validation) in fixtures {
        let rows = equations(seed);
        let (a, b) = rows.split_at(4656);
        assert_eq!(
            (raw_hash(a), raw_hash(b)),
            (train.into(), validation.into())
        );
    }
}

#[test]
fn public_codec_round_trips_every_ordered_catalog_equation() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.lake/reproduction/gpt");
    for (name, digest) in [
        (
            "src/native_transformer/codec/src/lib.rs",
            "d9c808ec03fdffe86cc0cd79c81cb456440c03f4b2e6b1f2f0881827ef6b5dd8",
        ),
        (
            "src/native_transformer/codec/src/codec.rs",
            "2e021f13692ef166a8a0c7b8fa893d41298b6a48a48cde0f81c35dd396f952bf",
        ),
        (
            "src/diff_gpt/application/grokking/div97.py",
            "8c4d0123bdc0240b07f40e8f787d8603908ef73f5b69af4526a0d4cb88deb0e2",
        ),
        (
            "tests/diff_gpt/application/grokking/test_div97.py",
            "80f9003d1771473064606cd8aa310866842ec1c2c4b0879ff345d213fe6f6618",
        ),
    ] {
        assert_eq!(
            hash(&fs::read(source.join(name)).unwrap()),
            digest,
            "{name}"
        );
    }
    let rows = equations(1);
    let n = rows.len();
    for order in [1usize, 2] {
        let vocab = 256 << (order - 1);
        let domains = vec![Domain::symmetric(((vocab - (1 << order)) / 2) as f64); n];
        let operators = vec![Difference::order(order).unwrap(); n];
        let mut levels = vec![0.; order * n];
        for column in 0..3 {
            levels.extend(rows.iter().map(|r| f64::from(r[column])));
        }
        let input = Matrix::new(n, levels).unwrap();
        let encoded = encode(&input, vocab, &domains, &operators, Tokenizer::Lgt).unwrap();
        assert_eq!(encoded.clipped_tokens, 0);
        assert!(encoded.grid.scale.iter().all(|&s| s == 1.));
        assert_eq!(
            decode(
                &encoded.prefix,
                &encoded.grid,
                &encoded.tokens,
                vocab,
                &operators,
                Tokenizer::Lgt
            )
            .unwrap(),
            input
        );
        let tokens = encoded.tokens.values();
        assert_eq!(tokens.len(), 3 * n);
        let expected: Vec<[u32; 3]> = (0..n)
            .map(|i| {
                [
                    tokens[i] as u32,
                    tokens[i + n] as u32,
                    tokens[i + 2 * n] as u32,
                ]
            })
            .collect();
        let (train, validation) = expected.split_at(4656);
        let data = differenced::build(order as u32, 1).unwrap();
        assert_eq!(data.vocabulary, vocab as usize);
        assert_eq!(data.train, train);
        assert_eq!(data.validation, validation);
        assert_eq!(
            data.fingerprints,
            [
                reference::fingerprint(train),
                reference::fingerprint(validation)
            ]
        );
    }
}
