use super::*;
impl Request {
    pub fn validate(&self) -> Result<u64> {
        self.system.validate()?;
        let f = &self.factor;
        let p = &self.policy;
        if !(2..=257).contains(&self.n)
            || !(2..=257).contains(&self.classes)
            || self.system.variables != 2 * self.n + self.classes
            || self.system.rows.len() > 8191
            || self.system.right.iter().any(|&b| b != 0)
            || !finite_system::prime(f.prime)
            || !(1..=8).contains(&f.exponent)
            || (f.prime as u64).checked_pow(f.exponent as u32) != Some(f.modulus as u64)
            || f.modulus > 257
            || !self.classes.is_multiple_of(f.modulus as usize)
            || p.stages > 32
            || p.states > 4096
            || p.total_scalar_updates > 4_000_000_000
            || p.enumeration.points > 4096
        {
            return Err("component homogeneous shape, factor or work limits exceeded".into());
        }
        self.base_request(0).validate()?;
        let rows = (self.system.rows.len() + 1) as u64;
        let variables = self.system.variables as u64;
        let stages = if f.exponent == 1 {
            p.stages.min(1)
        } else {
            p.stages
        };
        // Includes retained stage matrices/instructions, branch points, JSON
        // and independent checker/replay copies. No space-sized allocation.
        let bound = 2 * 1024 * 1024 * 1024
            + stages as u64
                * (64 * rows * variables
                    + 256 * p.elimination.row_operations as u64
                    + 64 * p.enumeration.points as u64 * (rows + variables))
            + 128 * (1 + stages) as u64 * p.enumeration.points as u64 * variables;
        if bound > 32 * 1024 * 1024 * 1024 {
            return Err("component retained-tree forecast exceeds32GiB".into());
        }
        Ok(bound)
    }
    pub(super) fn elimination(&self, used: u64) -> finite_system::Policy {
        finite_system::Policy {
            scalar_updates: self
                .policy
                .elimination
                .scalar_updates
                .min(self.policy.total_scalar_updates - used),
            ..self.policy.elimination.clone()
        }
    }
    pub(super) fn base_request(&self, used: u64) -> affine::Request {
        let p = self.factor.prime as i64;
        affine::Request {
            problem: finite_system::Problem {
                system: finite_system::System {
                    prime: self.factor.prime,
                    variables: self.system.variables,
                    rows: self
                        .system
                        .rows
                        .iter()
                        .map(|r| r.iter().map(|a| a.rem_euclid(p) as u16).collect())
                        .collect(),
                    right: vec![0; self.system.rows.len()],
                },
                policy: self.elimination(used),
            },
            policy: self.policy.enumeration.clone(),
        }
    }
    pub(super) fn lift_request(&self, state: &State, used: u64) -> Result<lifts::Request> {
        let anchor = state
            .anchor
            .ok_or("retained component state lacks class anchor")?;
        if anchor >= self.classes {
            return Err("component class anchor is outside the original class block".into());
        }
        self.system.validate_point(state.modulus, &state.point)?;
        let mut system = self.system.clone();
        let mut row = vec![0; system.variables];
        row[2 * self.n + anchor] = 1;
        system.rows.push(row);
        system.right.push(1);
        Ok(lifts::Request {
            system,
            modulus: state.modulus,
            prime: self.factor.prime,
            base: state.point.clone(),
            elimination: self.elimination(used),
            enumeration: self.policy.enumeration.clone(),
        })
    }
}
pub(super) fn original(r: &Request, m: u16, x: &[u16]) -> Result<()> {
    r.system.validate_point(m, x)?;
    for row in &r.system.rows {
        let value: i128 = row
            .iter()
            .zip(x)
            .map(|(&a, &z)| a as i128 * z as i128)
            .sum();
        if value.rem_euclid(m as i128) != 0 {
            return Err("normalized component point fails original equations".into());
        }
    }
    Ok(())
}
