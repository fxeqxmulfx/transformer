//! Independent checks for public forward, backward and update entry points.
use super::*;

fn mean_loss(e: &Engine, rows: usize, targets: usize) -> Result<f64> {
    Ok(e.losses(rows)?.iter().map(|&x| f64::from(x)).sum::<f64>() / targets as f64)
}

fn successor(prefix: &[u32], query: u32, vocab: u32) -> u32 {
    if pair_memory::CLOSED_RELATION
        && let Some(end) = crate::closed_checks::reference(prefix, query, vocab)
    {
        return end;
    }
    prefix
        .windows(2)
        .rev()
        .find(|pair| pair[0] == query)
        .map_or(query, |pair| pair[1])
}

// Independent transcription of PairKey.address, using the reference field
// exponentiation and reading the stored tables directly in double precision.
fn residual(prefix: &[u32], b: u32, p: u32, d: usize, tables: &[f32]) -> Vec<f64> {
    let a = prefix.last().copied().unwrap_or(b);
    let power = |x, i| arithmetic_data::power(x, i, p);
    let inverse = if b == 0 { 0 } else { power(b, p - 2) };
    let mut keys = vec![b, (a + p - b) % p, a.abs_diff(b), a * inverse % p];
    if pair_memory::SUCCESSOR {
        keys[0] = successor(prefix, b, p);
    }
    keys.extend((1..=3).flat_map(|i| (1..=3).map(move |j| (power(a, i) + power(b, j)) % p)));
    if !prefix.is_empty() {
        if pair_memory::XOR_PREFIX {
            let left = prefix.iter().fold(0, |x, y| x ^ y);
            if cfg!(feature = "prefix-xor-cubic") {
                keys[12] = (power(left, 3) + power(b, 3)) % p;
            } else {
                keys[4] = (left + b) % p;
            }
        } else if cfg!(feature = "prefix-sum") {
            keys[4] = (prefix.iter().sum::<u32>() + b) % p;
        }
    }
    keys.extend((1..=2).flat_map(|i| (1..=2).map(move |j| power(a, i) * power(b, j) % p)));
    if cfg!(any(
        feature = "transition-count",
        feature = "transition-table"
    )) {
        let transitions = prefix.windows(2).filter(|edge| edge[0] != edge[1]).count();
        let bucket = if cfg!(feature = "inclusive-transitions") {
            let last_edge = usize::from(prefix.last().is_some_and(|&a| a != b));
            (transitions + last_edge).min(p as usize - 1)
        } else {
            transitions % p as usize
        } as u32;
        if cfg!(feature = "transition-table") {
            keys.push(bucket);
        } else {
            keys[16] = (keys[16] + bucket) % p;
        }
    }
    let mut result = vec![0.; d];
    for (table, key) in keys.into_iter().enumerate() {
        let offset = (table * p as usize + key as usize) * d;
        for (x, &v) in result.iter_mut().zip(&tables[offset..offset + d]) {
            *x += f64::from(v);
        }
    }
    result
}

fn profile(p: usize, length: usize, width: usize) -> Result<()> {
    let batch = 3;
    let mut cfg = config(batch)?;
    cfg.max_length = length;
    cfg.vocab = p;
    cfg.width = width;
    cfg.heads = cfg.heads.max(width / 32);
    cfg.gain = cfg.heads as f32 / width as f32;
    let d = cfg.width;
    let mut base = Engine::new(cfg.clone(), &layout::initialize_model(&cfg, 0))?;
    let mut tokens: Vec<_> = (0..batch * length)
        .map(|i| {
            let r = i / length;
            [0, p - 1, 9, 0, 6, 4][(i % length + if r == 1 { 2 } else { 0 }) % 6] as u32
        })
        .collect();
    if cfg!(feature = "transition-table") && length == 17 {
        for (row, sequence) in tokens.chunks_mut(length).enumerate() {
            for (i, token) in sequence.iter_mut().enumerate() {
                *token = if row == 0 || (row == 1 && i < 5) {
                    (i % 2) as u32
                } else {
                    0
                };
            }
        }
    } else if cfg!(feature = "inclusive-transitions") && length == 6 {
        for row in tokens.chunks_exact_mut(length) {
            row.copy_from_slice(&[0, 1, 1, 2, 2, 0]);
        }
    } else if cfg!(feature = "single-write-gain") && length == 9 {
        let patterns = [
            [1, 7, 2, 4, 5, 6, 8, 9, 1],
            [1, 7, 2, 1, 7, 6, 8, 9, 1],
            [1, 7, 2, 1, 8, 6, 8, 9, 1],
        ];
        for (row, pattern) in tokens.chunks_mut(length).zip(patterns) {
            row.copy_from_slice(&pattern);
        }
    } else if pair_memory::CLOSED_RELATION && length == 8 {
        let patterns = [
            [1, 0, 2, 0, 3, 0, 1, 1],
            [1, 0, 2, 0, 3, 0, 2, 1],
            [3, 0, 1, 0, 2, 0, 3, 3],
        ];
        for (row, pattern) in tokens.chunks_mut(length).zip(patterns) {
            row.copy_from_slice(&pattern);
        }
    } else if cfg!(feature = "relation-support") && length == 18 {
        let patterns = [
            [2, 0, 3, 4, 1, 0, 2, 1, 1, 0, 2, 1, 7, 0, 8, 7, 0, 7],
            [1, 0, 2, 0, 3, 0, 4, 0, 5, 9, 1, 6, 7, 6, 8, 9, 9, 1],
            [1, 0, 2, 4, 1, 0, 5, 1, 5, 0, 3, 1, 5, 0, 5, 1, 5, 1],
        ];
        for (row, pattern) in tokens.chunks_mut(length).zip(patterns) {
            row.copy_from_slice(&pattern);
        }
    } else if cfg!(feature = "relation-paths") && length >= 8 {
        let patterns = [
            [2, 0, 3, 4, 1, 0, 2, 1, 1, 0, 2, 1],
            [1, 0, 2, 0, 3, 1, 4, 2, 4, 3, 5, 1],
            [1, 0, 2, 4, 1, 0, 5, 1, 5, 0, 3, 1],
        ];
        for (row, pattern) in tokens.chunks_mut(length).zip(patterns) {
            row.copy_from_slice(&pattern[..length]);
        }
    }
    let mut labels: Vec<_> = (0..tokens.len())
        .map(|i| ((7 * i + 3) % p) as u32)
        .collect();
    labels[1] = u32::MAX;
    let check_count_bucket = (cfg!(feature = "transition-table") && length == 17)
        || (cfg!(feature = "inclusive-transitions") && length == 6);
    if check_count_bucket {
        labels.fill(u32::MAX);
        labels[length - 1] = 7;
    }
    if pair_memory::CLOSED_RELATION && length == 8 {
        // This prefix has ordinary successor 0 and unique closed endpoint 3.
        labels.fill(u32::MAX);
        labels[length - 1] = 3;
    }
    if cfg!(feature = "relation-paths") && matches!(length, 12 | 18) {
        // Isolate a tie at length 12 and a strict support winner at length 18.
        labels.fill(u32::MAX);
        labels[2 * length - 1] = 3;
    }
    let targets = labels.iter().filter(|&&label| label != u32::MAX).count();
    base.forward(&tokens, &labels, batch, length)?;
    let scores = base.logits.read()?;
    cfg.head = crate::head::Head::Relations;
    let start = pair_memory::start(&cfg);
    let gain_at = pair_memory::gain(&cfg);
    let mut theta = layout::initialize_model(&cfg, 0);
    let selected = pair_memory::DEFAULT_MODEL;
    let expected_gain = if selected || cfg!(feature = "successor-gain-three-quarters") {
        0.75
    } else if pair_memory::MODEL == "phasor-prefix-successor-half-v1"
        || cfg!(feature = "successor-displacement")
        || cfg!(feature = "transition-count")
    {
        0.5
    } else {
        0.25
    };
    if theta[gain_at] != expected_gain || theta[start..gain_at].iter().any(|&v| v != 0.) {
        return Err("shared residual initialization differs from specification".into());
    }
    if cfg!(feature = "successor-bonus") {
        if theta.get(gain_at + 1) != Some(&0.5) {
            return Err(
                "independent successor bonus initialization differs from specification".into(),
            );
        }
        // Exercise an independently signed bonus in the reference and adjoint.
        theta[gain_at + 1] = -0.2;
    }
    if cfg!(feature = "relation-paths") {
        let amplitudes = &mut theta[gain_at + 1..layout::frequency_offset(&cfg)];
        let initial = if cfg!(feature = "relation-path-mean") {
            1.
        } else {
            0.
        };
        if amplitudes.len() != p || amplitudes.iter().any(|&x| x != initial) {
            return Err("relation amplitude initialization differs from specification".into());
        }
        for (h, x) in amplitudes.iter_mut().enumerate() {
            *x = 0.07 * ((h % 5) as f32 - 1.25);
        }
    }
    let mut rng = random::Random::new(301);
    for x in &mut theta[start..gain_at] {
        *x = rng.gauss(0., 0.005) as f32;
    }
    theta[gain_at] = 0.7;
    let mut e = Engine::new(cfg, &theta)?;
    e.forward(&tokens, &labels, batch, length)?;
    let actual = e.logits.read()?;
    let mut worst = 0_f64;
    for (i, &b) in tokens.iter().enumerate() {
        let prev = if i % length == 0 { i } else { i - 1 };
        let a = tokens[prev];
        let prefix = &tokens[i - i % length..i];
        let mut v = residual(prefix, b, p as u32, d, &theta[start..gain_at]);
        let mut gain = f64::from(theta[gain_at]);
        if selected || cfg!(feature = "successor-displacement") {
            let r = successor(prefix, b, p as u32);
            let agreed = !cfg!(feature = "successor-agreement")
                || prefix
                    .windows(2)
                    .filter(|edge| edge[0] == b)
                    .all(|edge| edge[1] == r);
            let single = cfg!(feature = "single-write-gain")
                && r != b
                && prefix.windows(2).filter(|pair| pair[0] == b).count() == 1;
            if pair_memory::CLOSED_GAIN
                && (single || crate::closed_checks::reference(prefix, b, p as u32).is_some())
            {
                gain *= 2.;
            } else if cfg!(feature = "successor-bonus") && r != b {
                gain += f64::from(theta[gain_at + 1]);
            } else if cfg!(feature = "successor-hit-gain") && r != b && agreed {
                gain *= if cfg!(feature = "successor-hit-gain-seven-quarters") {
                    7. / 4.
                } else {
                    2.
                };
            }
            let strength = if cfg!(feature = "successor-displacement-quarter") {
                0.25
            } else {
                1.0
            };
            for (col, x) in v.iter_mut().enumerate() {
                *x += strength
                    * (f64::from(theta[r as usize * d + col])
                        - f64::from(theta[b as usize * d + col]));
            }
        }
        if cfg!(feature = "prefix-frequency") && prefix.len() >= 2 {
            let history = &prefix[..prefix.len() - 1];
            for (col, value) in v.iter_mut().enumerate() {
                let current = f64::from(theta[b as usize * d + col]);
                let offset: f64 = history
                    .iter()
                    .map(|&token| f64::from(theta[token as usize * d + col]) - current)
                    .sum();
                *value += 0.5 * offset / history.len() as f64;
            }
        }
        if cfg!(feature = "relation-paths") {
            let next = |key: u32, h: usize| {
                prefix
                    .windows(3)
                    .rev()
                    .find(|triple| triple[0] == key && triple[1] == h as u32)
                    .map(|triple| triple[2])
            };
            let walk = |key, h| next(key, h).and_then(|middle| next(middle, h));
            let mut ends: Vec<_> = (0..p).map(|h| walk(b, h).unwrap_or(b)).collect();
            if cfg!(feature = "relation-support") {
                let counts: Vec<_> = (0..p)
                    .map(|h| (0..p as u32).filter(|&key| walk(key, h).is_some()).count())
                    .collect();
                let maximum = (0..p)
                    .filter(|&h| ends[h] != b)
                    .map(|h| counts[h])
                    .max()
                    .unwrap_or(0);
                for (end, count) in ends.iter_mut().zip(counts) {
                    if count < maximum {
                        *end = b;
                    }
                }
            }
            let divisor = if cfg!(feature = "relation-path-mean") {
                ends.iter().filter(|&&end| end != b).count().max(1) as f64
            } else {
                1.
            };
            for (end, &amplitude) in ends.into_iter().zip(&theta[gain_at + 1..gain_at + 1 + p]) {
                for (col, value) in v.iter_mut().enumerate() {
                    *value += f64::from(amplitude) / divisor
                        * (f64::from(theta[end as usize * d + col])
                            - f64::from(theta[b as usize * d + col]));
                }
            }
        }
        let read = checks::reference(
            &scores[prev * p..(prev + 1) * p],
            &theta[..p * d],
            &v,
            (a * b) as usize % p,
        );
        for (c, increment) in read.into_iter().enumerate() {
            let expected = f64::from(scores[i * p + c]) + gain * increment;
            let difference = (expected - f64::from(actual[i * p + c])).abs();
            if !difference.is_finite() {
                return Err("nonfinite shared-value logit".into());
            }
            worst = worst.max(difference);
        }
    }
    println!("vocab={p} length={length} maximum FP64 logit difference={worst:.3e}");
    if worst > 2e-5 {
        return Err("shared forward failed".into());
    }
    e.backward(batch, length)?;
    let gradients = e.gradients.read()?;
    let mut directions = vec![
        ("tied embedding".to_owned(), 0..p * d),
        ("residual gain".to_owned(), gain_at..gain_at + 1),
    ];
    if check_count_bucket {
        // The six-token fixture has three changes including the final edge.
        // In twelve buckets the alternating fixture either wraps to 3 or caps at 11.
        let offset = start + (pair_memory::TABLES - 1) * p * d;
        let bucket = if cfg!(feature = "inclusive-transitions") && length == 17 {
            11
        } else {
            3
        };
        let active = offset + bucket * d..offset + (bucket + 1) * d;
        let unused = offset + 4 * d..offset + 5 * d;
        if gradients[offset..offset + p * d]
            .chunks_exact(d)
            .enumerate()
            .any(|(i, row)| row.iter().any(|&g| g != 0.) != (i == bucket))
        {
            return Err("transition table gradient reached the wrong count bucket".into());
        }
        directions.push(("active transition count".to_owned(), active));
        directions.push(("unused transition count".to_owned(), unused));
    }
    if pair_memory::CLOSED_RELATION && length == 8 {
        let endpoint = start + 3 * d..start + 4 * d;
        let ordinary = start..start + d;
        if gradients[endpoint.clone()].iter().all(|&v| v == 0.)
            || gradients[ordinary.clone()].iter().any(|&v| v != 0.)
        {
            return Err(
                "address replacement sent a gradient to the wrong successor table row".into(),
            );
        }
        directions.push(("closed endpoint table row".to_owned(), endpoint));
        directions.push(("ordinary successor table row".to_owned(), ordinary));
    }
    if cfg!(feature = "successor-bonus") {
        directions.push(("successor bonus".to_owned(), gain_at + 1..gain_at + 2));
    }
    if cfg!(feature = "relation-paths") {
        let range = gain_at + 1..gain_at + 1 + p;
        let active = gradients[range.clone()].iter().any(|&x| x.abs() > 1e-6);
        if active != (length >= 8) {
            return Err("relation gradient disagrees with the exercised paths".into());
        }
        directions.push(("relation amplitudes".to_owned(), range));
        if length == 12 {
            directions.push((
                "one of two agreeing relations".to_owned(),
                gain_at + 1..gain_at + 2,
            ));
        }
        if length == 18 {
            if gradients[gain_at + 1].abs() <= 1e-6 || gradients[gain_at + 7] != 0. {
                return Err("strict relation winner did not isolate its amplitude gradient".into());
            }
            directions.push(("highest support".to_owned(), gain_at + 1..gain_at + 2));
            directions.push(("lower support".to_owned(), gain_at + 7..gain_at + 8));
        }
    }
    for k in 0..pair_memory::TABLES {
        directions.push((
            format!("memory {k}"),
            start + k * p * d..start + (k + 1) * p * d,
        ));
    }
    for (label, range) in directions {
        let signs: Vec<f32> = range
            .clone()
            .map(|_| if rng.random() < 0.5 { -1. } else { 1. })
            .collect();
        let derivative: f64 = gradients[range.clone()]
            .iter()
            .zip(&signs)
            .map(|(&x, &s)| f64::from(x) * f64::from(s))
            .sum();
        let delta = 0.0002;
        let mut losses = [0.; 2];
        for (i, sign) in [-1., 1.].into_iter().enumerate() {
            let mut v = theta.clone();
            for (x, &s) in v[range.clone()].iter_mut().zip(&signs) {
                *x += sign * delta * s;
            }
            e.parameters.write(&v)?;
            e.forward(&tokens, &labels, batch, length)?;
            losses[i] = mean_loss(&e, tokens.len(), targets)?;
        }
        let numerical = (losses[1] - losses[0]) / (2. * f64::from(delta));
        println!(
            "vocab={p} length={length} {label} analytic={derivative:.6} numeric={numerical:.6}"
        );
        if !derivative.is_finite()
            || !numerical.is_finite()
            || (derivative - numerical).abs() > 0.01 * (1. + numerical.abs())
        {
            return Err(format!("shared {label} adjoint mismatch").into());
        }
    }
    Ok(())
}

// Rebuild the adjoint after intervening inference at a different batch shape.
fn check_table_transpose() -> Result<()> {
    let mut cfg = config(2)?;
    cfg.head = crate::head::Head::Relations;
    cfg.max_length = 4;
    let (start, gain) = (pair_memory::start(&cfg), pair_memory::gain(&cfg));
    let weights = layout::initialize_model(&cfg, 0);
    let mut engine = Engine::new(cfg, &weights)?;
    if engine.backward(2, 2).is_ok() {
        return Err("table adjoint accepted a batch before its forward pass".into());
    }
    let tokens = [2, 3, 2, 3];
    let labels = [u32::MAX, 9, u32::MAX, 11];
    engine.forward(&tokens, &labels, 2, 2)?;
    if engine.backward(1, 4).is_ok() {
        return Err("table adjoint accepted a different sequence boundary".into());
    }
    engine.backward(2, 2)?;
    let first = engine.gradients.read()?;
    engine.forward(&[4, 1, 4, 1], &labels, 1, 4)?;
    engine.forward(&tokens, &labels, 2, 2)?;
    engine.backward(2, 2)?;
    let repeated = engine.gradients.read()?;
    if first[start..gain] != repeated[start..gain] || !first[start..gain].iter().any(|&x| x != 0.) {
        return Err("table adjoint changed after intervening inference".into());
    }
    Ok(())
}

pub fn run(width: Option<usize>) -> Result<()> {
    if !precision::FP32_AFFINE_REFERENCE {
        return Err("finite differences require FP32 affines".into());
    }
    if cfg!(feature = "relation-support") || pair_memory::CLOSED_RELATION {
        crate::relation_support::check_counts()?;
    }
    if pair_memory::CLOSED_RELATION {
        crate::closed_checks::run()?;
    }
    if let Some(width) = width {
        if ![64, 128, 256].contains(&width) {
            return Err("shared checks support widths 64, 128 and 256".into());
        }
        println!("width={width}");
        return profile(36, 8, width);
    }
    check_table_transpose()?;
    // Include a composite vocabulary, multi-position batches, a one-token
    // sequence, repeated histories and supervised initial positions.
    for (p, l) in [(97, 2), (100, 4), (36, 1)] {
        profile(p, l, 128)?;
    }
    if pair_memory::SUCCESSOR {
        profile(36, 8, 128)?;
    }
    if cfg!(feature = "single-write-gain") {
        profile(36, 9, 128)?;
    }
    if cfg!(feature = "transition-table") {
        profile(12, 17, 128)?;
    }
    if cfg!(feature = "inclusive-transitions") {
        profile(12, 6, 128)?;
    }
    if cfg!(feature = "relation-paths") {
        profile(36, 12, 128)?;
    }
    if cfg!(feature = "relation-support") {
        profile(36, 18, 128)?;
    }
    let mut cfg = config(8)?;
    cfg.head = crate::head::Head::Relations;
    let (p, d) = (cfg.vocab, cfg.width);
    let mut theta = layout::initialize_model(&cfg, 0);
    theta[..p * d].fill(0.);
    let start = pair_memory::start(&cfg);
    for b in 0..p {
        theta[b * d + b] = 1.;
        theta[start + b * d + b] = 2.;
    }
    theta[pair_memory::gain(&cfg)] = 8.;
    let mut e = Engine::new(cfg, &theta)?;
    for lo in (0..p).step_by(8) {
        let words: Vec<_> = (lo..(lo + 8).min(p)).flat_map(|b| [0, b as u32]).collect();
        e.forward(&words, &vec![u32::MAX; words.len()], words.len() / 2, 2)?;
        let labels = e.predictions.read()?;
        if words
            .as_chunks::<2>()
            .0
            .iter()
            .zip(labels.as_chunks::<2>().0)
            .any(|(tokens, got)| tokens[1] != got[1])
        {
            return Err("shared zero-row witness failed".into());
        }
    }
    println!("one table and gain, zero-row witness {p}/{p}");
    Ok(())
}
