use convex_stress_paths_compare::{Request, Result, fingerprint, prepare, replay};
use std::{fs::File, io::Write, path::Path};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let value = match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["fingerprint"] => { println!("{}",fingerprint()); return Ok(()); },
        ["prepare", request, destination] => {
            let request: Request = serde_json::from_reader(File::open(request)?)?;
            prepare(Path::new(destination),&request)?;
            return Ok(());
        },
        ["replay",root] => replay(Path::new(root))?,
        _ => return Err("usage: convex-stress-paths-compare fingerprint | prepare REQUEST FRESH_ROOT | replay ROOT".into()),
    };
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    serde_json::to_writer(&mut out, &value)?;
    writeln!(out)?;
    Ok(())
}
