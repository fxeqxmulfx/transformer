use bilinear::guard;
use learned_retrieval::evaluation;
use learned_retrieval_transfer::{Result, case};
use std::path::Path;

const USAGE: &str = concat!(
    "usage: learned-retrieval-transfer CASE fit WORKSPACE STORAGE RUN MODE SEED",
    " | CASE seal RUN | CASE score|replay WORKSPACE STORAGE RUN"
);

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [name, op, root] if op == "seal" => case(name)?.study.seal(Path::new(root)),
        [name, op, workspace, storage, root, rest @ ..] => {
            let workspace = Path::new(workspace);
            let storage = Path::new(storage);
            let root = Path::new(root);
            let admit = || guard::admit(1 << 30, workspace, storage).map(|_| ());
            admit()?;
            let c = case(name)?;
            match (op.as_str(), rest) {
                ("fit", [mode, seed]) => c.study.fit_cell(
                    root,
                    c.task.vocab,
                    &c.task.splits["train"],
                    mode,
                    seed.parse()?,
                    admit,
                ),
                ("score" | "replay", []) => evaluation::score(
                    &c.study,
                    &c.task,
                    &c.criteria,
                    root,
                    op == "replay",
                    admit,
                    evaluation::prefix_split,
                ),
                _ => Err(USAGE.into()),
            }
        }
        _ => Err(USAGE.into()),
    }
}
