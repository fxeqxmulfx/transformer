//! CM12 proposal search, after Lean 0c89afcc. Only observed categorical
//! rectangles propose a permutation of the fixed Fourier class coordinates.
//! This discrete search has no global optimization or recovery certificate.
use crate::{Result, Sample};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Coding {
    /// Original class -> class coordinate used by the unchanged spectral solver.
    pub encode: Vec<usize>,
}

impl Coding {
    pub fn identity(n: usize) -> Self {
        Self {
            encode: (0..n).collect(),
        }
    }
    pub fn validate(&self, n: usize) -> Result<()> {
        let mut sorted = self.encode.clone();
        sorted.sort_unstable();
        if sorted != (0..n).collect::<Vec<_>>() {
            return Err("class coding is not a full-vocabulary permutation".into());
        }
        Ok(())
    }
    pub fn samples(&self, observations: &[Sample]) -> Result<Vec<Sample>> {
        observations
            .iter()
            .map(|s| {
                Ok(Sample {
                    cell: s.cell,
                    label: *self.encode.get(s.label).ok_or("label outside coding")?,
                })
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Discovery {
    pub coding: Option<Coding>,
    pub reason: String,
    pub field_order: usize,
    pub constraints: usize,
    pub rank_including_constant_gauge: usize,
    pub observed_classes: usize,
}

// The field concerns class phases only. Input tokens and target labels are
// categorical indices; no arithmetic function of input-token values is used.
fn prime(n: usize) -> bool {
    n >= 2
        && !(2..)
            .take_while(|d| d * d <= n)
            .any(|d| n.is_multiple_of(d))
}
fn power(mut x: usize, mut e: usize, p: usize) -> usize {
    let mut value = 1;
    while e > 0 {
        if e % 2 == 1 {
            value = value * x % p;
        }
        x = x * x % p;
        e /= 2;
    }
    value
}

struct Echelon {
    p: usize,
    rows: Vec<Option<Vec<usize>>>,
    rank: usize,
}
impl Echelon {
    fn new(p: usize) -> Self {
        Self {
            p,
            rows: vec![None; p],
            rank: 0,
        }
    }
    fn add(&mut self, mut equation: Vec<usize>) {
        for pivot in 0..self.p {
            let factor = equation[pivot];
            if factor == 0 {
                continue;
            }
            if let Some(row) = &self.rows[pivot] {
                for j in pivot..self.p {
                    equation[j] = (equation[j] + self.p - factor * row[j] % self.p) % self.p;
                }
            } else {
                let inverse = power(factor, self.p - 2, self.p);
                for entry in &mut equation[pivot..] {
                    *entry = *entry * inverse % self.p;
                }
                self.rows[pivot] = Some(equation);
                self.rank += 1;
                return;
            }
        }
    }
    fn unique_direction(&self) -> Option<Vec<usize>> {
        if self.rank + 1 != self.p {
            return None;
        }
        let free = self.rows.iter().position(Option::is_none)?;
        let mut values = vec![0; self.p];
        values[free] = 1;
        for pivot in (0..self.p).rev() {
            if let Some(row) = &self.rows[pivot] {
                let sum = row[pivot + 1..]
                    .iter()
                    .zip(&values[pivot + 1..])
                    .fold(0, |sum, (a, b)| (sum + a * b) % self.p);
                values[pivot] = (self.p - sum) % self.p;
            }
        }
        Some(values)
    }
}

/// Verify all observed edges admit row/column phase potentials, including
/// longer graph cycles absent from the rectangle equations. These potentials
/// are discarded; inference always reads the trained probability table.
fn consistent_edges(n: usize, labels: &[Option<usize>], coding: &Coding) -> bool {
    let mut values = vec![None; 2 * n];
    for start in 0..2 * n {
        if values[start].is_some() {
            continue;
        }
        values[start] = Some(0);
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(node) = queue.pop_front() {
            let known = values[node].unwrap();
            for other in 0..n {
                let (cell, target) = if node < n {
                    (node + n * other, n + other)
                } else {
                    (other + n * (node - n), other)
                };
                if let Some(label) = labels[cell] {
                    let next = (coding.encode[label] + n - known) % n;
                    if let Some(old) = values[target] {
                        if old != next {
                            return false;
                        }
                    } else {
                        values[target] = Some(next);
                        queue.push_back(target);
                    }
                }
            }
        }
    }
    true
}

/// Stream rectangle constraints z[a]+z[d]-z[b]-z[c]=0 over class phases.
/// With prime vocabulary V, fix z[0]=0 and require exactly one remaining
/// direction, a bijection, and consistency on every observed edge. Otherwise
/// retain an explicit rejected proposal; never substitute a guessed coding.
pub fn discover(n: usize, observations: &[Sample]) -> Result<Discovery> {
    if !(2..=193).contains(&n) {
        return Err("coding vocabulary exceeds frozen 193-class cap".into());
    }
    let mut labels = vec![None; n * n];
    let mut classes = vec![false; n];
    for s in observations {
        if s.cell >= n * n || s.label >= n || labels[s.cell].is_some() {
            return Err("invalid or repeated categorical input cell".into());
        }
        labels[s.cell] = Some(s.label);
        classes[s.label] = true;
    }
    if observations.is_empty() {
        return Err("empty coding observations".into());
    }
    let mut result = Discovery {
        coding: None,
        reason: String::new(),
        field_order: n,
        constraints: 0,
        rank_including_constant_gauge: 0,
        observed_classes: classes.iter().filter(|x| **x).count(),
    };
    if !prime(n) {
        result.reason = "composite_vocabulary_unsupported".into();
        return Ok(result);
    }
    if result.observed_classes != n {
        result.reason = "unobserved_output_classes".into();
        return Ok(result);
    }
    let mut system = Echelon::new(n);
    let mut gauge = vec![0; n];
    gauge[0] = 1;
    system.add(gauge);
    for row in 0..n {
        for other in row + 1..n {
            let mut anchor = None;
            for column in 0..n {
                let (Some(a), Some(b)) = (labels[row + n * column], labels[other + n * column])
                else {
                    continue;
                };
                if let Some((c, d)) = anchor {
                    let mut equation = vec![0; n];
                    for (label, coefficient) in [(a, 1), (d, 1), (b, n - 1), (c, n - 1)] {
                        equation[label] = (equation[label] + coefficient) % n;
                    }
                    system.add(equation);
                    result.constraints += 1;
                } else {
                    anchor = Some((a, b));
                }
            }
        }
    }
    result.rank_including_constant_gauge = system.rank;
    let Some(encode) = system.unique_direction() else {
        result.reason = if system.rank == n {
            "no_nonconstant_phase_code"
        } else {
            "ambiguous_phase_code"
        }
        .into();
        return Ok(result);
    };
    let coding = Coding { encode };
    if coding.validate(n).is_err() {
        result.reason = "nonbijective_phase_code".into();
    } else if !consistent_edges(n, &labels, &coding) {
        result.reason = "inconsistent_observed_cycle".into();
    } else {
        result.reason = "unique_bijective_phase_code".into();
        result.coding = Some(coding);
    }
    Ok(result)
}
