//! Bounded evidence traversal, ported refined_quotient/cohort/source at4f210282.
use super::*;
/// Reserve checked before each bounded8GiB stage write, as well as per case.
pub(crate) fn reserve(path: &Path) -> Result<()> {
    let output = std::process::Command::new("df")
        .args(["-Pk", "--"])
        .arg(path)
        .output()?;
    if !output.status.success() {
        return Err("cannot inspect cohort disk reserve".into());
    }
    let text = std::str::from_utf8(&output.stdout)?;
    let available: u64 = text
        .lines()
        .last()
        .and_then(|line| line.split_whitespace().nth(3))
        .ok_or("missing available disk blocks")?
        .parse()?;
    if available < 16 * 1024 * 1024 {
        return Err("less than16GiB stage disk reserve".into());
    }
    Ok(())
}
pub(crate) fn evidence(path: &Path) -> Result<Value> {
    Ok(json!({"path":path,"sha256":digest(path)?,"bytes":fs::metadata(path)?.len()}))
}
pub(crate) fn tree(path: &Path, files: &mut Vec<Value>) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let mut entries: Vec<_> = fs::read_dir(path)?.collect::<std::result::Result<_, _>>()?;
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        if entry.file_type()?.is_dir() {
            tree(&entry.path(), files)?;
        } else {
            files.push(evidence(&entry.path())?);
        }
    }
    Ok(())
}
pub(crate) fn exit(root: &Path, filename: &str) -> Result<i32> {
    Ok(fs::read_to_string(root.join(filename))?.trim().parse()?)
}
