//! Exact order of source random_formula and program_with_witnesses draws.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Selected {
    pub formula: Formula,
    pub witnesses: [Vec<i64>; 2],
    pub candidate_count: usize,
}
fn draw(depth: usize, rng: &mut Random) -> Formula {
    if depth == 0 {
        let f = Formula::Sym(SYMBOLS[rng.below(3) as usize]);
        return if rng.random() < 0.25 {
            Formula::Not(Box::new(f))
        } else {
            f
        };
    }
    let mut left = Term::Count(Box::new(draw(depth - 1, rng)));
    let mut right = Term::Count(Box::new(draw(depth - 1, rng)));
    if rng.random() < 0.5 {
        right = Term::Add(
            Box::new(right),
            Box::new(Term::Count(Box::new(Formula::Sym(
                SYMBOLS[rng.below(3) as usize],
            )))),
        );
    }
    if rng.random() < 0.25 {
        left = Term::Add(Box::new(left), Box::new(Term::Neg(Box::new(right))));
        right = Term::Const(rng.below(5) as i64 - 2);
    } else {
        right = Term::Add(
            Box::new(right),
            Box::new(Term::Const(rng.below(5) as i64 - 2)),
        );
    }
    let mut f = Formula::Lt(left, right);
    if rng.random() < 0.3 {
        f = Formula::Not(Box::new(f));
    }
    if rng.random() < 0.3 {
        f = Formula::And(Box::new(f), Box::new(draw(0, rng)));
    }
    f
}
pub fn select(depth: usize, seed: u32) -> Result<Selected> {
    if !(1..=5).contains(&depth) {
        return Err("formula count depth outside1..5".into());
    }
    let mut rng = Random::new(seed);
    let mut words = vec![];
    for size in 1..=4u32 {
        for index in 0..3usize.pow(size) {
            let mut rest = index;
            let mut word = vec![0; size as usize];
            for symbol in word.iter_mut().rev() {
                *symbol = SYMBOLS[rest % 3];
                rest /= 3;
            }
            words.push(word);
        }
    }
    for _ in 0..32 {
        let size = 5 + rng.below(8) as usize;
        words.push((0..size).map(|_| SYMBOLS[rng.below(3) as usize]).collect());
    }
    for attempt in 1..=64 {
        let formula = draw(depth, &mut rng);
        let mut witnesses: [Option<Vec<i64>>; 2] = [None, None];
        for word in &words {
            let result = *formula.evaluate(word)?.last().ok_or("empty witness word")? as usize;
            witnesses[result].get_or_insert_with(|| word.clone());
            if witnesses.iter().all(Option::is_some) {
                if formula.depth() != depth {
                    return Err("sampled syntactic depth differs".into());
                }
                return Ok(Selected {
                    formula,
                    witnesses: witnesses.map(Option::unwrap),
                    candidate_count: attempt,
                });
            }
        }
    }
    Err("source witness-bank program selection exhausted64 attempts".into())
}
