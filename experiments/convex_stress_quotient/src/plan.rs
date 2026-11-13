use super::*;
fn name(names: &mut Vec<usize>, value: usize) -> usize {
    names.iter().position(|&x| x == value).unwrap_or_else(|| {
        names.push(value);
        names.len() - 1
    })
}
impl Request {
    pub(crate) fn problem(&self) -> Result<(Names, finite_system::Problem, u64)> {
        let input = &self.input;
        if !(2..=257).contains(&input.n)
            || !(2..=1024).contains(&input.vocabulary)
            || input.observations.is_empty()
            || input.observations.len() > input.n * input.n
            || !(1..=8190).contains(&input.policy.equations)
            || !catalogue().contains(&self.prime)
            || input.policy.elimination.nullity != 1
        {
            return Err("invalid bounded categorical alias discovery input".into());
        }
        let mut names = Names {
            rows: vec![],
            columns: vec![],
            classes: vec![],
        };
        let mut cells = BTreeSet::new();
        for &[cell, label] in &input.observations {
            if cell >= input.n * input.n || label >= input.vocabulary || !cells.insert(cell) {
                return Err("invalid or repeated original categorical cell/label".into());
            }
            name(&mut names.rows, cell % input.n);
            name(&mut names.columns, cell / input.n);
            name(&mut names.classes, label);
        }
        let variables = names.rows.len() + names.columns.len() + names.classes.len();
        if variables > 1024 || names.classes.len() < 2 {
            return Err("observed coordinate dimension outside bounds".into());
        }
        let mut rows = vec![];
        for &[cell, label] in input.observations.iter().take(input.policy.equations) {
            let mut row = vec![0; variables];
            row[names
                .rows
                .iter()
                .position(|&r| r == cell % input.n)
                .unwrap()] = 1;
            row[names.rows.len()
                + names
                    .columns
                    .iter()
                    .position(|&c| c == cell / input.n)
                    .unwrap()] = 1;
            row[names.rows.len()
                + names.columns.len()
                + names.classes.iter().position(|&c| c == label).unwrap()] = self.prime - 1;
            rows.push(row);
        }
        // Row-first and class-first shift gauges only. No missing-axis pins.
        for variable in [0, names.rows.len() + names.columns.len()] {
            let mut row = vec![0; variables];
            row[variable] = 1;
            rows.push(row);
        }
        let problem = finite_system::Problem {
            system: finite_system::System {
                prime: self.prime,
                variables,
                right: vec![0; rows.len()],
                rows,
            },
            policy: input.policy.elimination.clone(),
        };
        let bytes = 4 * problem.validate()? + 512 * 1024 * 1024;
        if bytes > 8 * 1024 * 1024 * 1024 {
            return Err("single-stage replay forecast exceeds8GiB".into());
        }
        Ok((names, problem, bytes))
    }
}
pub(crate) fn fibers(code: &Code) -> Vec<Vec<usize>> {
    (0..code.prime)
        .map(|z| {
            code.names
                .classes
                .iter()
                .zip(&code.classes)
                .filter_map(|(&label, &value)| (value == z).then_some(label))
                .collect()
        })
        .collect()
}
