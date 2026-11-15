//! CM61 at822bb5e8: selected exact anchors, with every observed group kept
//! in CE. Projection is ported from CM57 atdf9344f4. Unlike the old domain
//! constructor, an explicit selection can have only its identity-copy prior;
//! no artificial training observations are inserted to satisfy a shape cap.
use crate::{corpus::Corpus, *};
use joint::{
    constraints::{Domain as ExactDomain, Projection},
    projection as project,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Anchors {
    All,
    Groups(Vec<u32>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadSpec {
    pub coordinates: Vec<f64>,
    pub budgets: Vec<f64>,
    pub radius: f64,
    pub minimum_mass: f64,
    pub anchors: Anchors,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedDomain {
    pub original_groups: usize,
    pub selected_groups: Vec<u32>,
    pub constraints: ExactDomain,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Domain {
    Exact(ExactDomain),
    Selected(SelectedDomain),
}

impl Domain {
    pub fn new(corpus: &Corpus, spec: &HeadSpec) -> Result<Self> {
        corpus.validate()?;
        joint::data::coordinates(&spec.coordinates)?;
        if spec.coordinates.len() != corpus.limits().vocabulary
            || spec.budgets.len() != corpus.limits().programs
        {
            return Err("domain specification differs from complete corpus dimensions".into());
        }
        match &spec.anchors {
            Anchors::All => Ok(Self::Exact(ExactDomain::new(
                corpus.groups(),
                &spec.coordinates,
                spec.budgets.clone(),
                spec.radius,
                spec.minimum_mass,
            )?)),
            Anchors::Groups(groups) => {
                Ok(Self::Selected(SelectedDomain::new(corpus, spec, groups)?))
            }
        }
    }
    pub fn constraints(&self) -> &ExactDomain {
        match self {
            Self::Exact(d) => d,
            Self::Selected(d) => &d.constraints,
        }
    }
    pub fn violation(&self, theta: &[Parameters]) -> f64 {
        self.constraints().violation(theta)
    }
    pub fn project(
        &self,
        input: &[Parameters],
        cycles: usize,
        tolerance: f64,
    ) -> Result<Projection> {
        match self {
            Self::Exact(d) => d.project(input, cycles, tolerance),
            Self::Selected(d) => d.project(input, cycles, tolerance),
        }
    }
}

impl SelectedDomain {
    fn new(corpus: &Corpus, spec: &HeadSpec, groups: &[u32]) -> Result<Self> {
        if groups.len() > corpus.groups().len()
            || groups.windows(2).any(|w| w[0] >= w[1])
            || groups.iter().any(|&g| g as usize >= corpus.groups().len())
        {
            return Err("selected anchor groups must be ordered distinct original groups".into());
        }
        let mut anchors = vec![vec![[0., 0., 0., 0., 1., -1.]]; spec.budgets.len()];
        for &g in groups {
            let r = &corpus.groups()[g as usize];
            for (p, h) in r.features.iter().enumerate() {
                anchors[p].push([h[0], h[1], h[2], h[3], h[4], -spec.coordinates[r.label]]);
            }
        }
        let basis = anchors
            .iter()
            .map(|r| project::basis(r))
            .collect::<Result<Vec<_>>>()?;
        let feature_basis = anchors
            .iter()
            .map(|r| {
                let rows: Vec<_> = r
                    .iter()
                    .map(|r| {
                        let mut h = *r;
                        h[5] = 0.;
                        h
                    })
                    .collect();
                project::basis(&rows)
            })
            .collect::<Result<Vec<_>>>()?;
        let out = Self {
            original_groups: corpus.groups().len(),
            selected_groups: groups.to_vec(),
            constraints: ExactDomain {
                radius: spec.radius,
                minimum_mass: spec.minimum_mass,
                budgets: spec.budgets.clone(),
                anchors,
                basis,
                feature_basis,
            },
        };
        out.validate()?;
        if out.constraints.basis.iter().all(|b| b.len() == 6) {
            return Err("selected anchors force every program mass to zero".into());
        }
        Ok(out)
    }
    pub fn validate(&self) -> Result<()> {
        let d = &self.constraints;
        let p = d.budgets.len();
        if !(1..=joint::MAX_RECORDS).contains(&self.original_groups)
            || self.selected_groups.len() > self.original_groups
            || self.selected_groups.windows(2).any(|w| w[0] >= w[1])
            || self
                .selected_groups
                .iter()
                .any(|&g| g as usize >= self.original_groups)
            || !(1..=joint::MAX_PROGRAMS).contains(&p)
            || d.anchors.len() != p
            || d.basis.len() != p
            || d.feature_basis.len() != p
            || !d.radius.is_finite()
            || !(1e-6..=1e12).contains(&d.radius)
            || !d.minimum_mass.is_finite()
            || d.minimum_mass <= 0.
            || 6. * d.minimum_mass > d.radius
            || d.budgets
                .iter()
                .any(|b| !b.is_finite() || !(6. ..=1e6).contains(b))
        {
            return Err("invalid saved selected-anchor domain".into());
        }
        joint::data::forecast(self.original_groups.max(2), p, 2)?;
        for (i, rows) in d.anchors.iter().enumerate() {
            if rows.len() != self.selected_groups.len() + 1
                || rows[0] != [0., 0., 0., 0., 1., -1.]
                || rows
                    .iter()
                    .flatten()
                    .any(|x| !x.is_finite() || x.abs() > 4096.)
                || d.basis[i] != project::basis(rows)?
            {
                return Err("selected anchors or constraint basis changed".into());
            }
            let features: Vec<_> = rows
                .iter()
                .map(|r| {
                    let mut h = *r;
                    h[5] = 0.;
                    h
                })
                .collect();
            if d.feature_basis[i] != project::basis(&features)? {
                return Err("selected observable feature basis changed".into());
            }
        }
        Ok(())
    }
    /// Same bounded Dykstra updates and full residual checks as CM57. The
    /// selected equations are fixed; reaching the cycle cap remains a refusal.
    pub fn project(
        &self,
        input: &[Parameters],
        max_cycles: usize,
        tolerance: f64,
    ) -> Result<Projection> {
        self.validate()?;
        let d = &self.constraints;
        if input.len() != d.budgets.len()
            || input
                .iter()
                .flatten()
                .any(|x| !x.is_finite() || x.abs() > 1e15)
            || !(1..=20000).contains(&max_cycles)
            || !tolerance.is_finite()
            || !(1e-12..=1e-7).contains(&tolerance)
        {
            return Err("invalid bounded selected-anchor projection request".into());
        }
        let mut x = input.to_vec();
        let mut corrections = vec![vec![[0.; 6]; x.len()]; 4];
        for cycle in 1..=max_cycles {
            let mut movement: f64 = 0.;
            for (kind, correction) in corrections.iter_mut().enumerate() {
                let old = x.clone();
                for (t, c) in x.iter_mut().zip(correction.iter()) {
                    for j in 0..6 {
                        t[j] += c[j];
                    }
                }
                let before = x.clone();
                match kind {
                    0 => {
                        for (t, b) in x.iter_mut().zip(&d.basis) {
                            project::linear(t, b);
                        }
                    }
                    1 => {
                        for (t, b) in x.iter_mut().zip(&d.budgets) {
                            *t = project::perspective(*t, *b)?;
                        }
                    }
                    2 => {
                        let shift = ((d.minimum_mass - x.iter().map(|t| t[5]).sum::<f64>())
                            / x.len() as f64)
                            .max(0.);
                        for t in &mut x {
                            t[5] += shift;
                        }
                    }
                    _ => {
                        let values: Vec<_> = x.iter().flatten().copied().collect();
                        let weights: Vec<_> = x.iter().flat_map(|_| joint::COSTS).collect();
                        let flat = project::weighted_l1(&values, &weights, d.radius)?;
                        for (t, src) in x.iter_mut().zip(flat.as_chunks::<6>().0) {
                            t.copy_from_slice(src);
                        }
                    }
                }
                for p in 0..x.len() {
                    for j in 0..6 {
                        correction[p][j] = before[p][j] - x[p][j];
                        movement = movement.max((old[p][j] - x[p][j]).abs());
                    }
                }
            }
            let scale = x.iter().flatten().map(|x| x.abs()).fold(1., f64::max);
            let violation = d.violation(&x);
            if violation <= tolerance && movement / scale <= tolerance {
                let squared_distance = input
                    .iter()
                    .flatten()
                    .zip(x.iter().flatten())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                return Ok(Projection {
                    theta: x,
                    cycles: cycle,
                    normalized_violation: violation,
                    normalized_step: movement / scale,
                    squared_distance,
                });
            }
        }
        Err("selected-anchor projection did not converge within its declared cycle budget".into())
    }
}
