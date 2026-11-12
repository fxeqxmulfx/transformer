//! The checks. No expected value here comes from the code it checks: the generator is held to
//! published outputs of MT19937 and NumPy and to fingerprints that NumPy itself produced; the
//! reference model to closed forms and to finite differences of its own forward pass; the fast
//! model to the reference; the optimizer to the recurrence written out in doubles.

use crate::data::{self, Equation};
use crate::layout::Dims;
use crate::mt::Mt;
use crate::reference::{self, Example};

/// Pseudo-random parameters in doubles: every tensor filled, gains near 1.
pub fn random_params(dims: Dims, seed: u32, scale: f64) -> Vec<f64> {
    let mut rng = Mt::new(seed);
    let mut p: Vec<f64> = (0..dims.len()).map(|_| scale * rng.normal()).collect();
    for l in 0..dims.layers {
        let at = dims.block(l);
        for i in 0..dims.d {
            p[at.g1 + i] += 1.0;
            p[at.g2 + i] += 1.0;
        }
    }
    p
}

/// Pseudo-random examples over the whole vocabulary.
pub fn random_batch(dims: Dims, seed: u32, n: usize) -> Vec<Example> {
    let mut rng = Mt::new(seed);
    let token = |rng: &mut Mt| rng.interval(dims.vocab as u32 - 1) as usize;
    (0..n)
        .map(|_| Example {
            tokens: (0..dims.ctx).map(|_| token(&mut rng)).collect(),
            targets: (0..dims.ctx).map(|_| token(&mut rng)).collect(),
        })
        .collect()
}

const SMALL: Dims = Dims { vocab: 11, d: 8, heads: 2, layers: 2, dff: 12, ctx: 6 };

#[test]
fn mt19937_matches_published_outputs() {
    // The C++ standard's check value: the 10000th output of `mt19937` seeded with 5489.
    let mut rng = Mt::new(5489);
    assert_eq!(rng.next_u32(), 3_499_211_612);
    for _ in 1..9999 {
        rng.next_u32();
    }
    assert_eq!(rng.next_u32(), 4_123_659_995);
}

#[test]
fn numpy_legacy_draws_match() {
    // `np.random.seed(0); np.random.rand(4)`, as NumPy prints it to 16 digits.
    let mut rng = Mt::new(0);
    let expected = [0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969];
    for value in expected {
        assert_eq!(rng.double(), value);
    }
    // `np.random.seed(0); np.random.permutation(10)`.
    let mut order: Vec<usize> = (0..10).collect();
    Mt::new(0).shuffle(&mut order);
    assert_eq!(order, [2, 8, 4, 9, 1, 6, 7, 3, 0, 5]);
}

#[test]
fn state_round_trips() {
    let mut rng = Mt::new(7);
    for _ in 0..1000 {
        rng.next_u32();
    }
    let mut copy = Mt::from_words(&rng.words());
    assert_eq!(copy, rng);
    assert_eq!(copy.next_u32(), rng.next_u32());
}

#[test]
fn equations_are_the_division_table() {
    let all = data::equations();
    assert_eq!(all.len(), 9312);
    let mut seen = vec![false; 97 * 97];
    for Equation { num, den, quo } in all {
        assert!(den != 0 && (num as usize) < 97 && (quo as usize) < 97);
        // The answer times the denominator is the numerator: checked without the generator's formula.
        assert_eq!((quo as usize * den as usize) % 97, num as usize);
        assert!(!std::mem::replace(&mut seen[num as usize * 97 + den as usize], true), "an equation twice");
    }
    assert_eq!(seen.iter().filter(|&&s| s).count(), 9312);
    assert_eq!(data::train_len(9312, 50.0), 4656);
}

#[test]
fn tokens_are_the_authors() {
    let eq = Equation { num: 3, den: 5, quo: 20 };
    assert_eq!(eq.tokens(), [0, 25, 10, 27, 1, 42, 0]);
    assert_eq!(eq.target(data::ANSWER_POS), 42);
    assert_eq!(eq.target(data::EOS_POS), 0);
}

#[test]
fn split_matches_numpy_fingerprints() {
    // Recorded by the repository's Python port of the author's generator, which shuffled with
    // NumPy itself: `experiments/grokking_internals/internal_suite_results.json`.
    let (train, heldout) = data::split(1, 4656);
    assert_eq!(data::fingerprint(&train), "ee3506fe908abba46d259aa0f889acf6a0aa3fc5040ab62f46d966216ff2aa62");
    assert_eq!(data::fingerprint(&heldout), "b69fde22ceee4d5bc3c56298a91eacb9548d0851b74c1108d3faf1ebe92ff412");
    let (train, heldout) = data::split(0, 1862);
    assert_eq!(data::fingerprint(&train), "492c6e09d5c8d97dde0f9e71344689edf00a19a6aca1f5ef3d7de31ab78376dc");
    assert_eq!(data::fingerprint(&heldout), "dfeb50f291c27a7e385d90f5c747da6b0a2347eb305c226d55d7cefa5dd5b0f0");
}

#[test]
fn positions_are_the_sinusoids() {
    let pe = Dims::PAPER.positions();
    assert_eq!(pe[0], 0.0);
    assert_eq!(pe[1], 1.0);
    // Position 3, channels 0 and 1: sin 3 and cos 3; channels 2 and 3: the angle 3 / 10000^(2/128).
    assert_eq!(pe[3 * 128], 3f64.sin() as f32);
    assert_eq!(pe[3 * 128 + 1], 3f64.cos() as f32);
    let angle = 3.0 / 10000f64.powf(2.0 / 128.0);
    assert_eq!(pe[3 * 128 + 2], angle.sin() as f32);
    assert_eq!(pe[3 * 128 + 3], angle.cos() as f32);
}

#[test]
fn parameter_count_is_the_papers() {
    let dims = Dims::PAPER;
    // Per block: 12 head maps of 32 by 128, the output map, two feed-forward maps, two normalizations.
    let block = 12 * 32 * 128 + 128 * 128 + 2 * 128 * 512 + 4 * 128;
    assert_eq!(dims.block_len(), block);
    assert_eq!(dims.len(), 239 * 128 + 2 * block + 128 * 239);
    // "about 4 * 10^5 non-embedding parameters".
    assert_eq!(dims.len() - 239 * 128, 424_832);
}

/// With the attention output map and the second feed-forward map zero, a block is two layer
/// normalizations: the logits have a closed form, written out here.
#[test]
fn reference_reduces_to_normalizations() {
    let dims = SMALL;
    let mut p = random_params(dims, 3, 0.5);
    for l in 0..dims.layers {
        let at = dims.block(l);
        p[at.wo..at.wo + dims.d * dims.d].fill(0.0);
        p[at.w2..at.w2 + dims.dff * dims.d].fill(0.0);
    }
    let tokens = [1, 4, 2, 9, 0, 7];
    let got = reference::logits(dims, &p, &tokens);
    let pe = dims.positions();
    let norm = |z: &[f64], g: &[f64], b: &[f64]| -> Vec<f64> {
        let mean = z.iter().sum::<f64>() / z.len() as f64;
        let var = z.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / z.len() as f64;
        (0..z.len()).map(|i| (z[i] - mean) / (var + 1e-5).sqrt() * g[i] + b[i]).collect()
    };
    for t in 0..dims.ctx {
        let mut x: Vec<f64> = (0..dims.d).map(|i| p[tokens[t] * dims.d + i] + pe[t * dims.d + i] as f64).collect();
        for l in 0..dims.layers {
            let at = dims.block(l);
            x = norm(&x, &p[at.g1..at.g1 + dims.d], &p[at.b1..at.b1 + dims.d]);
            x = norm(&x, &p[at.g2..at.g2 + dims.d], &p[at.b2..at.b2 + dims.d]);
        }
        for c in 0..dims.vocab {
            let want: f64 = (0..dims.d).map(|i| x[i] * p[dims.wout() + i * dims.vocab + c]).sum();
            assert!((got[t * dims.vocab + c] - want).abs() < 1e-12);
        }
    }
}

/// Causal masking: the logits at a position do not change with later tokens, and do change
/// with an earlier one.
#[test]
fn reference_is_causal() {
    let dims = SMALL;
    let p = random_params(dims, 5, 0.5);
    let a = reference::logits(dims, &p, &[1, 4, 2, 9, 0, 7]);
    let b = reference::logits(dims, &p, &[1, 4, 2, 9, 3, 5]);
    assert_eq!(a[..4 * dims.vocab], b[..4 * dims.vocab]);
    assert!(a[4 * dims.vocab..] != b[4 * dims.vocab..]);
    let c = reference::logits(dims, &p, &[6, 4, 2, 9, 0, 7]);
    assert!((0..dims.vocab).any(|v| (a[5 * dims.vocab + v] - c[5 * dims.vocab + v]).abs() > 1e-6));
}

/// With zero query maps every head averages the values of the positions up to its own: with
/// one block and the feed-forward network off, the logits have a closed form.
#[test]
fn reference_attention_averages_under_zero_queries() {
    let dims = Dims { layers: 1, ..SMALL };
    let mut p = random_params(dims, 9, 0.5);
    let at = dims.block(0);
    p[at.wq..at.wq + dims.d * dims.d].fill(0.0);
    p[at.w2..at.w2 + dims.dff * dims.d].fill(0.0);
    let tokens = [1, 4, 2, 9, 0, 7];
    let got = reference::logits(dims, &p, &tokens);
    let (d, pe) = (dims.d, dims.positions());
    let x: Vec<Vec<f64>> =
        (0..dims.ctx).map(|t| (0..d).map(|i| p[tokens[t] * d + i] + pe[t * d + i] as f64).collect()).collect();
    for t in 0..dims.ctx {
        // The values are columns d..2d of the key-value map; the mean runs over positions 0..=t.
        let mut mean_v = vec![0.0; d];
        for xj in x.iter().take(t + 1) {
            for o in 0..d {
                for i in 0..d {
                    mean_v[o] += xj[i] * p[at.wkv + i * 2 * d + d + o] / (t + 1) as f64;
                }
            }
        }
        let mut z: Vec<f64> = x[t].clone();
        for o in 0..d {
            for i in 0..d {
                z[o] += mean_v[i] * p[at.wo + i * d + o];
            }
        }
        let norm = |z: &[f64], g: usize, b: usize| -> Vec<f64> {
            let mean = z.iter().sum::<f64>() / d as f64;
            let var = z.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / d as f64;
            (0..d).map(|i| (z[i] - mean) / (var + 1e-5).sqrt() * p[g + i] + p[b + i]).collect()
        };
        let y = norm(&norm(&z, at.g1, at.b1), at.g2, at.b2);
        for c in 0..dims.vocab {
            let want: f64 = (0..d).map(|i| y[i] * p[dims.wout() + i * dims.vocab + c]).sum();
            assert!((got[t * dims.vocab + c] - want).abs() < 1e-12, "position {t}, class {c}");
        }
    }
}

/// The reference gradient against central differences of the reference loss, in every
/// parameter of a small model, for both sets of supervised positions.
#[test]
fn reference_gradient_matches_finite_differences() {
    let dims = SMALL;
    let p = random_params(dims, 11, 0.4);
    let batch = random_batch(dims, 13, 3);
    for sup in [vec![4, 5], vec![4]] {
        let (value, grad) = reference::loss_and_gradient(dims, &p, &batch, &sup);
        assert!((value - reference::loss(dims, &p, &batch, &sup)).abs() < 1e-13);
        let h = 1e-6;
        let mut worst = 0f64;
        for i in 0..dims.len() {
            let (mut up, mut down) = (p.clone(), p.clone());
            up[i] += h;
            down[i] -= h;
            let slope =
                (reference::loss(dims, &up, &batch, &sup) - reference::loss(dims, &down, &batch, &sup)) / (2.0 * h);
            worst = worst.max((slope - grad[i]).abs());
        }
        assert!(worst < 2e-8, "the largest difference is {worst}");
        assert!(grad.iter().map(|g| g * g).sum::<f64>().sqrt() > 1e-3);
    }
}

#[test]
fn cross_entropy_is_the_log_of_the_classes_at_equal_logits() {
    let (value, probabilities) = reference::cross_entropy(&[0.25; 5], 2);
    assert!((value - 5f64.ln()).abs() < 1e-15);
    assert!(probabilities.iter().all(|p| (p - 0.2).abs() < 1e-15));
    // Two classes a logit apart: log(1 + e^-1) for the larger.
    let (value, _) = reference::cross_entropy(&[1.0, 0.0], 0);
    assert!((value - (1.0 + (-1f64).exp()).ln()).abs() < 1e-15);
}

use crate::adam::{self, Hyper};
use crate::gemm::Gemm;
use crate::run;
use crate::train::{self, Sampler, Trainer};

/// The kernel against the product written out in doubles, at sizes around its tile of 6 by 16,
/// its strips of 64 columns and its blocks of 768 steps.
#[test]
fn kernel_matches_the_product_in_doubles() {
    let mut gemm = Gemm::default();
    let mut rng = Mt::new(21);
    for (m, n, k) in [(1, 1, 1), (5, 17, 3), (6, 16, 8), (7, 65, 129), (384, 128, 128), (13, 239, 128), (128, 70, 800)]
    {
        let a: Vec<f32> = (0..m * k).map(|_| rng.normal() as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.normal() as f32).collect();
        let mut c = vec![f32::NAN; m * n];
        gemm.mul(&mut c, &a, &b, (m, n, k));
        for i in 0..m {
            for j in 0..n {
                let want: f64 = (0..k).map(|s| a[i * k + s] as f64 * b[s * n + j] as f64).sum();
                let size: f64 = (0..k).map(|s| (a[i * k + s] as f64 * b[s * n + j] as f64).abs()).sum();
                assert!((c[i * n + j] as f64 - want).abs() <= 1e-6 * size + 1e-30, "{m} by {n} by {k} at ({i}, {j})");
            }
        }
    }
}

#[test]
fn transpose_swaps_the_indices() {
    let x: Vec<f32> = (0..35 * 70).map(|i| i as f32).collect();
    let mut out = vec![0f32; 35 * 70];
    crate::gemm::transpose(&mut out, &x, 35, 70);
    for r in 0..35 {
        for c in 0..70 {
            assert_eq!(out[c * 35 + r], x[r * 70 + c]);
        }
    }
}

/// The fast model's loss and gradient against the reference's, on random examples over the
/// whole vocabulary and with every tensor random, for both sets of supervised positions and
/// with the batch cut over several threads.
fn fast_against_reference(dims: Dims, examples: usize, threads: usize, tolerance: f64) {
    for sup in [vec![4, 5], vec![4], vec![2, 5]] {
        let p64 = random_params(dims, 31, 0.3);
        let p32: Vec<f32> = p64.iter().map(|&v| v as f32).collect();
        let p64: Vec<f64> = p32.iter().map(|&v| v as f64).collect();
        let batch = random_batch(dims, 37, examples);
        let (want_loss, want_grad) = reference::loss_and_gradient(dims, &p64, &batch, &sup);
        let pe = dims.positions();
        let mut wt = crate::model::Transposed::new(dims);
        wt.update(dims, &p32);
        let model = crate::model::Model { dims, p: &p32, pe: &pe, wt: &wt, sup: &sup };
        let scale = 1.0 / (examples * sup.len()) as f32;
        let (mut loss, mut grad) = (0f64, vec![0f64; dims.len()]);
        for part in batch.chunks(examples.div_ceil(threads)) {
            let mut work = crate::model::Work::default();
            let tokens: Vec<usize> = part.iter().flat_map(|e| e.tokens.clone()).collect();
            let targets: Vec<usize> =
                part.iter().flat_map(|e| sup.iter().map(|&s| e.targets[s]).collect::<Vec<_>>()).collect();
            model.forward(&mut work, &tokens);
            for (b, example) in part.iter().enumerate() {
                let all = reference::logits(dims, &p64, &example.tokens);
                for (si, &pos) in sup.iter().enumerate() {
                    for c in 0..dims.vocab {
                        let (got, want) =
                            (work.logits[(b * sup.len() + si) * dims.vocab + c] as f64, all[pos * dims.vocab + c]);
                        assert!((got - want).abs() < tolerance * (1.0 + want.abs()), "logit {got} against {want}");
                    }
                }
            }
            loss += model.backward(&mut work, &tokens, &targets, scale).iter().sum::<f64>();
            grad.iter_mut().zip(&work.grad).for_each(|(g, &part)| *g += part as f64);
        }
        loss /= (examples * sup.len()) as f64;
        assert!((loss - want_loss).abs() < tolerance * want_loss, "loss {loss} against {want_loss}");
        let norm = want_grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        let error = grad.iter().zip(&want_grad).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
        assert!(norm > 1e-3 && error < tolerance * norm, "gradient error {error} of norm {norm}, positions {sup:?}");
        // Tensor by tensor too, so that a small tensor's gradient cannot hide in the norm of the rest.
        let mut bounds = vec![0, dims.vocab * dims.d];
        for l in 0..dims.layers {
            let at = dims.block(l);
            bounds.extend([at.wkv, at.wo, at.g1, at.b1, at.w1, at.w2, at.g2, at.b2, at.b2 + dims.d]);
        }
        bounds.push(dims.len());
        for pair in bounds.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            let norm = want_grad[lo..hi].iter().map(|g| g * g).sum::<f64>().sqrt();
            let error = grad[lo..hi].iter().zip(&want_grad[lo..hi]).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
            assert!(error <= 10.0 * tolerance * norm + 1e-9, "tensor at {lo}: error {error} of norm {norm}");
        }
    }
}

#[test]
fn fast_model_matches_reference_small() {
    fast_against_reference(SMALL, 7, 3, 2e-5);
}

#[test]
fn fast_model_matches_reference_at_the_papers_sizes() {
    fast_against_reference(Dims::PAPER, 5, 2, 5e-5);
}

/// The activation at its kink, where the source's ReLU has derivative 0. With the first map of
/// every feed-forward network zero, every hidden value is exactly 0. The loss has a kink there:
/// its slope in an entry of that map, taken from one side, is not zero. Both models give exactly
/// zero for the gradient in that map, as PyTorch does, and not the slope of either side.
#[test]
fn hidden_units_at_zero_pass_no_gradient() {
    let (dims, sup) = (SMALL, vec![4, 5]);
    let maps: Vec<std::ops::Range<usize>> =
        (0..dims.layers).map(|l| dims.block(l).w1..dims.block(l).w1 + dims.d * dims.dff).collect();
    let mut p32: Vec<f32> = random_params(dims, 41, 0.3).iter().map(|&v| v as f32).collect();
    for map in &maps {
        p32[map.clone()].fill(0.0);
    }
    let p64: Vec<f64> = p32.iter().map(|&v| v as f64).collect();
    let batch = random_batch(dims, 43, 4);
    let (value, want) = reference::loss_and_gradient(dims, &p64, &batch, &sup);
    let h = 1e-6;
    let slope = |i: usize, h: f64| {
        let mut moved = p64.clone();
        moved[i] += h;
        (reference::loss(dims, &moved, &batch, &sup) - value) / h
    };
    for map in &maps {
        let steepest = map.clone().map(|i| slope(i, h).abs().max(slope(i, -h).abs())).fold(0f64, f64::max);
        assert!(steepest > 1e-4, "the loss has no kink in the map at {}: slope {steepest}", map.start);
    }
    let pe = dims.positions();
    let mut wt = crate::model::Transposed::new(dims);
    wt.update(dims, &p32);
    let model = crate::model::Model { dims, p: &p32, pe: &pe, wt: &wt, sup: &sup };
    let mut work = crate::model::Work::default();
    let tokens: Vec<usize> = batch.iter().flat_map(|e| e.tokens.clone()).collect();
    let targets: Vec<usize> = batch.iter().flat_map(|e| sup.iter().map(|&s| e.targets[s])).collect();
    model.forward(&mut work, &tokens);
    model.backward(&mut work, &tokens, &targets, 1.0 / (batch.len() * sup.len()) as f32);
    for map in &maps {
        assert!(want[map.clone()].iter().all(|g| *g == 0.0), "the reference passes a gradient at 0");
        assert!(work.grad[map.clone()].iter().all(|g| *g == 0.0), "the fast model passes a gradient at 0");
    }
    // The other tensors have their gradients, in both models.
    let norm = |g: &mut dyn Iterator<Item = f64>| g.map(|g| g * g).sum::<f64>().sqrt();
    assert!(norm(&mut want.iter().copied()) > 1e-3 && norm(&mut work.grad.iter().map(|&g| g as f64)) > 1e-3);
}

#[test]
fn schedule_is_the_sources() {
    let h = Hyper::source(1.0);
    // `LambdaLR` has stepped `t - 1` times at update `t`: rates 0, 1e-4, ..., 9e-4, then 1e-3.
    assert_eq!(h.lr(1), 0.0);
    assert_eq!(h.lr(2), 1.0 / 10.0 * 1e-3);
    assert_eq!(h.lr(10), 9.0 / 10.0 * 1e-3);
    assert_eq!(h.lr(11), 1e-3);
    assert_eq!(h.lr(12), 1e-3);
    assert_eq!(h.lr(1_000_000), 1e-3);
}

#[test]
fn first_update_moves_nothing_and_starts_the_moments() {
    let h = Hyper::source(1.0);
    let (mut p, mut m, mut v) = (vec![0.5f32, -2.0], vec![0f32; 2], vec![0f32; 2]);
    adam::update(&h, 1, &mut p, &mut m, &mut v, &[3.0, -0.25]);
    assert_eq!(p, [0.5, -2.0]);
    assert!((m[0] - 0.3).abs() < 1e-7 && (m[1] + 0.025).abs() < 1e-8);
    assert!((v[0] - 0.18).abs() < 1e-7 && (v[1] - 0.00125).abs() < 1e-9);
}

/// At a constant gradient `g` the corrected moments are `g` and `g^2`, so an update without
/// decay moves the parameter by `lr g / (|g| + eps)`.
#[test]
fn constant_gradient_moves_by_the_rate() {
    let h = Hyper::source(0.0);
    for g in [0.37f32, -4.0, 1e-3] {
        let (mut p, mut m, mut v) = (vec![1f32], vec![0f32], vec![0f32]);
        let mut want = 1f64;
        for t in 1..=40 {
            adam::update(&h, t, &mut p, &mut m, &mut v, &[g]);
            want -= h.lr(t) * g as f64 / ((g as f64).abs() + 1e-8);
            assert!((p[0] as f64 - want).abs() < 2e-6, "update {t}: {} against {want}", p[0]);
        }
    }
}

/// Without a gradient only the decay acts, before the step: the parameter is multiplied by
/// `1 - lr_t` at the weight decay 1.
#[test]
fn decay_is_decoupled_and_follows_the_schedule() {
    let h = Hyper::source(1.0);
    let (mut p, mut m, mut v) = (vec![2f32], vec![0f32], vec![0f32]);
    for t in 1..=12 {
        adam::update(&h, t, &mut p, &mut m, &mut v, &[0.0]);
    }
    let rates = [0.0, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 1e-3];
    let want: f64 = rates.iter().fold(2.0, |p, lr| p * (1.0 - lr));
    assert!((p[0] as f64 - want).abs() < 1e-6, "{} against {want}", p[0]);
    // The decay is not in the moments: they stay zero.
    assert_eq!((m[0], v[0]), (0.0, 0.0));
}

/// The update against Adam with decoupled decay written out in doubles in its textbook form:
/// `p <- p (1 - lr wd) - lr mhat / (sqrt(vhat) + eps)`.
#[test]
fn update_matches_the_recurrence_in_doubles() {
    for wd in [0.0, 1.0] {
        let h = Hyper::source(wd);
        let mut rng = Mt::new(41);
        let n = 16;
        let (mut p, mut m, mut v) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
        let (mut p64, mut m64, mut v64) = (vec![0f64; n], vec![0f64; n], vec![0f64; n]);
        for i in 0..n {
            p[i] = rng.normal() as f32;
            p64[i] = p[i] as f64;
        }
        for t in 1..=25u64 {
            let g: Vec<f32> = (0..n).map(|_| (rng.normal() * 0.1) as f32).collect();
            adam::update(&h, t, &mut p, &mut m, &mut v, &g);
            let lr = if t <= 11 { (t - 1) as f64 * 1e-4 } else { 1e-3 };
            for i in 0..n {
                let gi = g[i] as f64;
                m64[i] = 0.9 * m64[i] + 0.1 * gi;
                v64[i] = 0.98 * v64[i] + 0.02 * gi * gi;
                let (mhat, vhat) = (m64[i] / (1.0 - 0.9f64.powi(t as i32)), v64[i] / (1.0 - 0.98f64.powi(t as i32)));
                p64[i] = p64[i] * (1.0 - lr * wd) - lr * mhat / (vhat.sqrt() + 1e-8);
                assert!((p[i] as f64 - p64[i]).abs() < 3e-6, "update {t}, decay {wd}: {} against {}", p[i], p64[i]);
            }
        }
    }
}

#[test]
fn sampler_cuts_fresh_permutations() {
    assert_eq!(train::batch_size(4656), 512);
    assert_eq!(train::batch_size(100), 50);
    let mut sampler = Sampler::new(5, 4656, 512);
    let mut epochs = Vec::new();
    for _ in 0..2 {
        let mut seen = vec![0u32; 4656];
        let mut order = Vec::new();
        for b in 0..10 {
            let batch = sampler.next();
            assert_eq!(batch.len(), if b < 9 { 512 } else { 48 });
            batch.iter().for_each(|&i| seen[i as usize] += 1);
            order.extend(batch);
        }
        assert!(seen.iter().all(|&count| count == 1), "an epoch is a permutation");
        epochs.push(order);
    }
    assert!(epochs[0] != epochs[1], "each epoch has a permutation of its own");
}

fn small_trainer(threads: usize) -> (Trainer, Vec<Equation>) {
    let dims = Dims { vocab: 239, d: 16, heads: 2, layers: 2, dff: 32, ctx: 6 };
    let params = train::initialize(dims, &mut Mt::new(train::init_seed(3)));
    let (train, _) = data::split(3, 300);
    (Trainer::new(dims, vec![4, 5], Hyper::source(1.0), params, threads), train)
}

#[test]
fn initialization_has_the_laws() {
    let dims = Dims::PAPER;
    let p = train::initialize(dims, &mut Mt::new(train::init_seed(0)));
    let moments = |x: &[f32]| {
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64;
        (mean, x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / x.len() as f64)
    };
    // The embedding is standard normal; a uniform on [-b, b] has variance b^2 / 3.
    let (mean, var) = moments(&p[..239 * 128]);
    assert!(mean.abs() < 0.02 && (var - 1.0).abs() < 0.03);
    let at = dims.block(1);
    for (lo, len, fan_in) in [
        (at.wq, 128 * 128, 128),
        (at.wkv, 2 * 128 * 128, 128),
        (at.w1, 128 * 512, 128),
        (at.w2, 512 * 128, 512),
        (dims.wout(), 128 * 239, 128),
    ] {
        let (mean, var) = moments(&p[lo..lo + len]);
        let bound = 1.0 / (fan_in as f64).sqrt();
        assert!(mean.abs() < 0.02 * bound && (var - bound * bound / 3.0).abs() < 0.03 * bound * bound);
        assert!(p[lo..lo + len].iter().all(|v| v.abs() as f64 <= bound));
    }
    assert!(p[at.g1..at.g1 + 128].iter().all(|&g| g == 1.0) && p[at.b2..at.b2 + 128].iter().all(|&b| b == 0.0));
}

/// A run stopped at a checkpoint and resumed from its bytes is, bit for bit, the run that was
/// not stopped: parameters, both moments and the batches drawn.
#[test]
fn resumed_run_is_the_same_run() {
    let (mut straight, train) = small_trainer(2);
    let mut sampler = Sampler::new(9, train.len(), train::batch_size(train.len()));
    let batch =
        |sampler: &mut Sampler| -> Vec<Equation> { sampler.next().iter().map(|&i| train[i as usize]).collect() };
    for _ in 0..7 {
        straight.update(&batch(&mut sampler));
    }
    let bytes = run::checkpoint_bytes(&[7; 32], 1.5, &straight, &sampler);
    for _ in 0..7 {
        straight.update(&batch(&mut sampler));
    }
    let (mut resumed, _) = small_trainer(2);
    let mut resumed_sampler = Sampler::new(0, train.len(), 1);
    assert!(run::restore(&bytes, &[8; 32], &mut resumed, &mut resumed_sampler).is_err(), "another configuration");
    let mut broken = bytes.clone();
    broken[200] ^= 1;
    assert!(run::restore(&broken, &[7; 32], &mut resumed, &mut resumed_sampler).is_err(), "a flipped bit");
    assert_eq!(run::restore(&bytes, &[7; 32], &mut resumed, &mut resumed_sampler), Ok(1.5));
    assert_eq!(resumed.step, 7);
    for _ in 0..7 {
        resumed.update(&batch(&mut resumed_sampler));
    }
    assert_eq!(resumed_sampler, sampler);
    assert!(resumed.p == straight.p && resumed.m == straight.m && resumed.v == straight.v);
    assert_eq!(resumed.step, 14);
}

/// Training lowers the loss of the batch it trains on, and an evaluation agrees with the
/// reference model's decisions.
#[test]
fn training_fits_a_batch_and_evaluation_agrees_with_reference() {
    let (mut trainer, train) = small_trainer(3);
    let batch = &train[..40];
    let first = trainer.update(batch);
    for _ in 0..60 {
        trainer.update(batch);
    }
    let last = trainer.gradient(batch);
    assert!(last < 0.8 * first, "the loss went from {first} to {last}");
    let eval = trainer.evaluate(batch);
    let p64: Vec<f64> = trainer.p.iter().map(|&v| v as f64).collect();
    let (mut ce_answer, mut ce_eos) = (0f64, 0f64);
    for (b, eq) in batch.iter().enumerate() {
        let all = reference::logits(trainer.dims, &p64, &eq.tokens()[..6]);
        let row = &all[4 * 239..5 * 239];
        let target = eq.target(4);
        let rival = (0..239).filter(|&c| c != target).map(|c| row[c]).fold(f64::NEG_INFINITY, f64::max);
        assert!((eval.margins[b] as f64 - (row[target] - rival)).abs() < 1e-4);
        if (row[target] - rival).abs() > 1e-3 {
            assert_eq!(eval.right[b], row[target] > rival);
        }
        ce_answer += reference::cross_entropy(row, target).0;
        ce_eos += reference::cross_entropy(&all[5 * 239..6 * 239], eq.target(5)).0;
    }
    assert!((eval.ce_answer - ce_answer / 40.0).abs() < 1e-4 && (eval.ce_eos - ce_eos / 40.0).abs() < 1e-4);
}

#[test]
fn decisions_take_the_first_of_the_largest() {
    assert_eq!(crate::model::decide(&[1.0, 3.0, 3.0, 2.0], 2), (1, 0.0));
    assert_eq!(crate::model::decide(&[1.0, 3.0, 5.0, 2.0], 2), (2, 2.0));
    assert_eq!(crate::model::decide(&[1.0, 3.0, 5.0, 2.0], 0), (2, -4.0));
}

#[test]
fn evaluation_steps_are_the_protocols() {
    let steps: Vec<u64> = (0..=12_000).filter(|&s| run::evaluated(s, 12_000)).collect();
    assert_eq!(steps[..6], [0, 10, 20, 50, 100, 200]);
    assert_eq!(steps.len(), 4 + 100 + 2);
    assert!(run::evaluated(1_000_000, 1_000_000) && !run::evaluated(10_100, 1_000_000));
    assert_eq!(run::field("{\"step\":1200,\"heldout_acc\":0.25}", "heldout_acc"), Some(0.25));
}

/// The chunks of a batch fix the order of the sums: one thread and five give the same bits.
#[test]
fn threads_do_not_change_the_result() {
    let (mut one, train) = small_trainer(1);
    let (mut five, _) = small_trainer(5);
    for step in 0..4 {
        let batch = &train[step * 40..step * 40 + 150];
        assert_eq!(one.update(batch).to_bits(), five.update(batch).to_bits());
    }
    assert!(one.p == five.p && one.m == five.m && one.v == five.v);
    let (a, b) = (one.evaluate(&train[..90]), five.evaluate(&train[..90]));
    assert!(a.margins == b.margins && a.predictions == b.predictions && a.ce_answer == b.ce_answer);
}

/// A run on disk that is stopped twice and resumed, on other numbers of threads, leaves the
/// files of the run that was not stopped: history without its times, predictions, margins and
/// the last checkpoint's parameters.
#[test]
fn run_on_disk_resumes_exactly() {
    let dir = std::env::temp_dir().join(format!("grok-division-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let config = |threads| run::Config {
        protocol: "adamw".into(),
        seed: 1,
        weight_decay: 1.0,
        budget: 23,
        sup: vec![4, 5],
        threads,
        checkpoint_every: 5,
        flush_from: None,
    };
    let (straight, stopped) = (dir.join("straight"), dir.join("stopped"));
    run::run(&config(8), &straight, None).expect("the straight run");
    run::run(&config(3), &stopped, Some(7)).expect("the first part");
    run::run(&config(16), &stopped, Some(9)).expect("the second part");
    run::run(&config(5), &stopped, None).expect("the last part");
    let timeless = |path: &std::path::Path| -> Vec<String> {
        let text = std::fs::read_to_string(path.join("history.jsonl")).expect("a history");
        text.lines().map(|l| l.split(",\"seconds\"").next().expect("a line").to_string()).collect()
    };
    assert_eq!(timeless(&straight), timeless(&stopped));
    assert_eq!(timeless(&straight).len(), 4, "the starts, updates 10 and 20, and the end");
    for name in ["heldout_predictions.bin", "heldout_margins.bin", "run.json", "heldout.txt"] {
        assert_eq!(
            std::fs::read(straight.join(name)).expect(name),
            std::fs::read(stopped.join(name)).expect(name),
            "{name}"
        );
    }
    let last =
        |path: &std::path::Path| std::fs::read(path.join("checkpoint-0000023.bin")).expect("the last checkpoint");
    let (a, b) = (last(&straight), last(&stopped));
    // Past the header with the seconds trained: parameters, moments and the sampler.
    assert_eq!(a[56..a.len() - 32], b[56..b.len() - 32]);
    assert_eq!(std::fs::read(straight.join("heldout_predictions.bin")).expect("predictions").len(), 4 * 4656);
    let report = run::report(&stopped).expect("a report");
    assert!(report.contains("\"complete\":true") && report.contains("\"last_step\":23"));
    // Another configuration does not continue the run.
    let mut other = config(8);
    other.weight_decay = 0.0;
    assert!(run::run(&other, &stopped, None).is_err());
    std::fs::remove_dir_all(&dir).expect("the directory is removed");
}

/// The loss gradient is PyTorch's in single precision. Away from saturation it is the softmax
/// less the indicator; at a margin of 30 the other classes' probabilities sum to `3 e^-30`,
/// below half a unit in the last place of 1, so the loss and the target's entry are exactly 0
/// while each other class keeps `e^-30` times the weight.
#[test]
fn loss_gradient_is_single_precision_softmax() {
    let mut row = [1.0f32, 0.0, -2.0];
    let loss = crate::model::loss_gradient(&mut row, 0, 0.5);
    let total = 1.0 + (-1f64).exp() + (-3f64).exp();
    assert!((loss - total.ln()).abs() < 1e-7);
    let want = [0.5 * (1.0 / total - 1.0), 0.5 * (-1f64).exp() / total, 0.5 * (-3f64).exp() / total];
    for c in 0..3 {
        assert!((row[c] as f64 - want[c]).abs() < 1e-7, "class {c}: {} against {}", row[c], want[c]);
    }
    let mut row = [0.0f32, 30.0, 0.0, 0.0];
    let loss = crate::model::loss_gradient(&mut row, 1, 0.25);
    assert_eq!(loss, 0.0);
    assert_eq!(row[1], 0.0);
    for c in [0, 2, 3] {
        let want = 0.25 * (-30f64).exp();
        assert!(row[c] > 0.0 && (row[c] as f64 - want).abs() < 1e-6 * want, "class {c}: {} against {want}", row[c]);
    }
    // In doubles the target's entry would be minus the sum of the others: not zero.
    assert!(3.0 * 0.25 * (-30f64).exp() > 0.0);
    // A wrong prediction at the same margin: the loss is the margin and the target's entry minus the weight.
    let mut row = [0.0f32, 30.0, 0.0, 0.0];
    let loss = crate::model::loss_gradient(&mut row, 0, 0.25);
    assert!((loss - 30.0).abs() < 1e-5 && (row[0] + 0.25).abs() < 1e-7 && (row[1] - 0.25).abs() < 1e-7);
}

use crate::arith;
use std::hint::black_box;

/// With the mode on, a subnormal result is zero and a subnormal operand counts as zero, while a
/// stored value keeps its bits. The mode is the thread's own, and off it is gradual underflow.
#[test]
fn flushing_replaces_subnormals_by_zero() {
    let tiny = f32::MIN_POSITIVE;
    let subnormal = f32::from_bits(1 << 20);
    let times = |x: f32, by: f32| black_box(x) * black_box(by);
    assert!(!arith::flushes_subnormals());
    assert!(times(tiny, 0.5) > 0.0 && times(times(tiny, 0.5), 4.0) == 2.0 * tiny, "gradual underflow");
    assert_eq!(times(subnormal, 4.0).to_bits(), 1 << 22);
    arith::flush_subnormals(true);
    assert!(arith::flushes_subnormals());
    assert_eq!(times(tiny, 0.5), 0.0, "a subnormal result");
    assert_eq!(times(subnormal, 4.0), 0.0, "a subnormal operand");
    assert_eq!(black_box(subnormal).to_bits(), 1 << 20, "a stored value");
    assert_eq!(times(3.0, 0.25), 0.75, "ordinary numbers");
    let other = std::thread::scope(|scope| {
        let thread = scope.spawn(|| {
            arith::flush_subnormals(false);
            times(tiny, 0.5).to_bits()
        });
        thread.join().expect("a thread")
    });
    // Read by its bits: with the mode on a comparison counts a subnormal as zero too.
    assert_eq!(other, 1 << 22, "a thread with the mode off");
    assert_eq!(times(tiny, 0.5), 0.0, "this thread still flushes");
    drop(arith::Restore);
    assert!(!arith::flushes_subnormals() && times(tiny, 0.5) > 0.0);
}

/// The optimizer's update at moments below the least normal number: the update is at most
/// `lr |m| / eps`, some `1e-34`, so parameters of ordinary size stay bit for bit in both modes.
/// Gradual underflow keeps the moments; the flush zeroes them.
#[test]
fn subnormal_moments_move_no_parameter_in_either_mode() {
    let hyper = Hyper::source(0.0);
    let start = [0.3f32, -1.7e-3, 2.5, -1e-6];
    let moment = f32::from_bits(1 << 21);
    let updates = |flush: bool| {
        arith::flush_subnormals(flush);
        let (mut p, mut m, mut v) = (start, [moment, -moment, moment, moment], [moment; 4]);
        let g = [0.0, 0.0, f32::from_bits(1 << 19), -f32::from_bits(3 << 18)];
        for t in 50_000..50_003 {
            adam::update(&hyper, t, &mut p, &mut m, &mut v, &g);
        }
        arith::flush_subnormals(false);
        (p, m, v)
    };
    let (gradual, flushed) = (updates(false), updates(true));
    assert!(gradual.0 == start && flushed.0 == start);
    assert!(gradual.1.iter().all(|m| *m != 0.0 && m.abs() < f32::MIN_POSITIVE));
    assert!(flushed.1 == [0.0; 4] && flushed.2 == [0.0; 4]);
}

/// At ordinary sizes the mode changes nothing: a trainer that flushes and one that does not
/// make the same updates and the same evaluation, bit for bit.
#[test]
fn flushing_changes_no_ordinary_update() {
    let (mut gradual, train) = small_trainer(3);
    let (mut flushed, _) = small_trainer(3);
    flushed.set_flush(true);
    for step in 0..12 {
        let batch = &train[step * 20..step * 20 + 64];
        assert_eq!(gradual.update(batch).to_bits(), flushed.update(batch).to_bits());
    }
    let (a, b) = (gradual.evaluate(&train[..90]), flushed.evaluate(&train[..90]));
    drop(arith::Restore);
    assert!(gradual.p == flushed.p && gradual.m == flushed.m && gradual.v == flushed.v);
    assert!(a.margins == b.margins && a.predictions == b.predictions && a.ce_answer == b.ce_answer);
    assert!(gradual.m.iter().chain(&gradual.v).all(|x| *x == 0.0 || x.abs() >= f32::MIN_POSITIVE), "no subnormal");
}

/// A run that flushes after update 12 and adopts the checkpoint of update 10 of the run that
/// never flushes leaves the files of the flushing run made from update 0, which is itself
/// stopped and resumed past the switch. A parent of another configuration, or without a
/// checkpoint that early, is refused.
#[test]
fn adopted_run_is_the_run_from_scratch() {
    let dir = std::env::temp_dir().join(format!("grok-division-adopt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let config = |seed, flush_from| run::Config {
        protocol: "adam".into(),
        seed,
        weight_decay: 0.0,
        budget: 23,
        sup: vec![4, 5],
        threads: 8,
        checkpoint_every: 5,
        flush_from,
    };
    let (parent, adopted, scratch) = (dir.join("parent"), dir.join("adopted"), dir.join("scratch"));
    run::run(&config(2, None), &parent, None).expect("the parent");
    assert!(run::adopt(&config(2, None), &parent, &adopted).is_err(), "a run that never flushes adopts nothing");
    assert!(run::adopt(&config(1, Some(12)), &parent, &adopted).is_err(), "another seed");
    assert!(run::adopt(&config(2, Some(3)), &parent, &adopted).is_err(), "no checkpoint up to update 3");
    assert!(!adopted.join("run.json").exists() && !adopted.join("checkpoint-0000010.bin").exists());
    run::adopt(&config(2, Some(12)), &parent, &adopted).expect("the adoption");
    let note = std::fs::read_to_string(adopted.join("adopted.json")).expect("the note");
    assert!(
        note.contains(&format!("\"parent_config\":\"{}\"", config(2, None).line())) && note.contains("\"step\":10")
    );
    run::run(&config(2, Some(12)), &adopted, None).expect("the adopted run");
    run::adopt(&config(2, Some(12)), &parent, &adopted).expect("a run that has begun adopts nothing");
    run::run(&config(2, Some(12)), &scratch, Some(17)).expect("the first part from scratch");
    run::run(&config(2, Some(12)), &scratch, None).expect("the rest from scratch");
    assert!(!arith::flushes_subnormals(), "the run leaves the thread with gradual underflow");
    let timeless = |path: &std::path::Path| -> Vec<String> {
        let text = std::fs::read_to_string(path.join("history.jsonl")).expect("a history");
        text.lines().map(|l| l.split(",\"seconds\"").next().expect("a line").to_string()).collect()
    };
    assert_eq!(timeless(&adopted), timeless(&scratch));
    assert_eq!(timeless(&adopted).len(), 4);
    // No number of these 23 updates is subnormal, so the parent's records are the same too.
    assert_eq!(timeless(&parent), timeless(&scratch));
    for name in ["heldout_predictions.bin", "heldout_margins.bin", "run.json", "heldout.txt"] {
        assert_eq!(
            std::fs::read(adopted.join(name)).expect(name),
            std::fs::read(scratch.join(name)).expect(name),
            "{name}"
        );
    }
    assert!(std::fs::read_to_string(adopted.join("run.json")).expect("run.json").contains(" flush_from=12\""));
    for step in [10, 23] {
        let bytes = |path: &std::path::Path| {
            std::fs::read(path.join(format!("checkpoint-{step:07}.bin"))).expect("a checkpoint")
        };
        let (a, b) = (bytes(&adopted), bytes(&scratch));
        // The configuration's hash, and past the seconds trained: parameters, moments, sampler.
        assert!(a[8..40] == b[8..40] && a[56..a.len() - 32] == b[56..b.len() - 32], "update {step}");
    }
    std::fs::remove_dir_all(&dir).expect("the directory is removed");
}
