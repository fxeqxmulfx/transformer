use super::*;

pub(super) fn factors(q: usize) -> Vec<Factor> {
    let mut remaining = q;
    let mut result = vec![];
    for prime in 2..=q {
        let (mut exponent, mut modulus) = (0, 1);
        while remaining.is_multiple_of(prime) {
            remaining /= prime;
            exponent += 1;
            modulus *= prime;
        }
        if exponent != 0 {
            result.push(Factor {
                prime: prime as u16,
                exponent,
                modulus: modulus as u16,
            });
        }
    }
    result
}
impl Request {
    pub(super) fn component_request(&self, factor: Factor) -> component::Request {
        component::Request {
            n: self.n,
            classes: self.classes,
            factor,
            system: self.system.clone(),
            policy: self.component_policy.clone(),
        }
    }
    pub fn validate(&self) -> Result<u64> {
        if !(2..=257).contains(&self.classes)
            || self.policy.points > 512
            || self.policy.original_products > 2_000_000_000
        {
            return Err("composite domain or tuple/product limits exceeded".into());
        }
        let factors = factors(self.classes);
        let mut bound = 2 * 1024 * 1024 * 1024;
        for factor in &factors {
            bound += self.component_request(factor.clone()).validate()?;
        }
        let rows = self.system.rows.len() as u64;
        let variables = self.system.variables as u64;
        // Retained CRT steps contain their actual matrices and witnesses.
        // Include serialization and independent replay copies before running.
        bound += 64
            * self.policy.points as u64
            * ((factors.len() - 1) as u64 * (rows * variables + rows + variables) + 2 * variables);
        if bound > 48 * 1024 * 1024 * 1024 {
            return Err("composite retained-bank forecast exceeds48GiB".into());
        }
        Ok(bound)
    }
    pub(super) fn products_per_branch(&self, factors: usize) -> u64 {
        (3 * (factors - 1) + 1) as u64
            * self.system.rows.len() as u64
            * self.system.variables as u64
    }
}
