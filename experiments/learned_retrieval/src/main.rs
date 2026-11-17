use learned_retrieval::{Result, cohort, evaluation};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [op,workspace,storage,root,mode,seed] if op=="fit" =>
            cohort::fit(Path::new(workspace),Path::new(storage),Path::new(root),mode,seed.parse()?),
        [op,root] if op=="seal" => cohort::seal(Path::new(root)),
        [op,workspace,storage,root] if op=="score" || op=="replay" =>
            evaluation::run(Path::new(workspace),Path::new(storage),Path::new(root),op=="replay"),
        _=> Err("usage: learned-retrieval fit WORKSPACE STORAGE RUN MODE SEED | seal RUN | score|replay WORKSPACE STORAGE RUN".into()),
    }
}
