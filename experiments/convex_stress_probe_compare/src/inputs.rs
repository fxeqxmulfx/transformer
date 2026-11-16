//! Frozen complete task361 data, including the routing teacher refusal and
//! unchanged original transformer history. Only training() crosses into model.
use crate::{resources::Guard, schema::Header, *};
use probes::{
    dataset::Dataset,
    inventory::{Entry, Inventory},
};
use std::{fs, mem::size_of, path::Path};
pub(crate) const OTHER_BYTES: u128 = 8 * io::JSON_MAX as u128 + 1024 * 1024;
pub struct Inputs {
    pub header: Header,
    pub inventory: Inventory,
}
pub fn prepare(workspace: &Path, root: &Path, data: &Path, producer: &str) -> Result<()> {
    if producer.len() != 40 || !producer.bytes().all(|x| x.is_ascii_hexdigit()) {
        return Err("expected frozen producer commit".into());
    }
    let guard = Guard::new(workspace, root.parent().ok_or("missing run parent")?)?;
    let inventory = probe(probes::inventory::read(data))?;
    let artifact = io::artifact(&data.join("inventory.json"))?;
    if artifact.sha256 != DATA_HASH {
        return Err("complete prepared source inventory differs from verified task361".into());
    }
    let per_run_forecast = crate::forecast::all(data, &inventory)?;
    fs::create_dir(root)?;
    fs::create_dir(root.join("data"))?;
    io::copy(
        &data.join("inventory.json"),
        &root.join("data/inventory.json"),
        &guard,
    )?;
    for entry in &inventory.cases {
        let name = id(entry)?;
        if entry.artifact.is_some() {
            io::copy(
                &data.join(format!("{name}.json")),
                &root.join(format!("data/{name}.json")),
                &guard,
            )?;
        }
    }
    let header = Header {
        schema: 1,
        producer: producer.into(),
        data_inventory: artifact,
        required_modes: MODES,
        process_bytes: resources::PROCESS,
        total_ram_bytes: resources::RAM,
        file_max: resources::FILE_MAX,
        run_max: resources::RUN_MAX,
        per_run_forecast,
    };
    io::json(&root.join("run.json"), &header, &guard)?;
    load(root)?;
    Ok(())
}
pub fn load(root: &Path) -> Result<Inputs> {
    let header: Header = io::read(&root.join("run.json"))?;
    if header.schema != 1
        || header.producer.len() != 40
        || !header.producer.bytes().all(|x| x.is_ascii_hexdigit())
        || header.required_modes != MODES
        || header.process_bytes != resources::PROCESS
        || header.total_ram_bytes != resources::RAM
        || header.file_max != resources::FILE_MAX
        || header.run_max != resources::RUN_MAX
        || header.data_inventory.sha256 != DATA_HASH
        || io::artifact(&root.join("data/inventory.json"))? != header.data_inventory
    {
        return Err("frozen probe comparison header changed".into());
    }
    let inventory = probe(probes::inventory::read(&root.join("data")))?;
    if header.per_run_forecast != crate::forecast::all(&root.join("data"), &inventory)? {
        return Err("complete original output forecast changed".into());
    }
    Ok(Inputs { header, inventory })
}
pub(crate) fn id(entry: &Entry) -> Result<&str> {
    let name = entry.source["case"]["name"]
        .as_str()
        .ok_or("missing original case name")?;
    if name.is_empty() || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-') {
        return Err("invalid case path component".into());
    }
    Ok(name)
}
pub(crate) fn entry<'a>(inputs: &'a Inputs, name: &str) -> Result<&'a Entry> {
    for entry in &inputs.inventory.cases {
        if id(entry)? == name {
            return Ok(entry);
        }
    }
    Err("case outside complete source inventory".into())
}
pub(crate) fn dataset(root: &Path, entry: &Entry) -> Result<Option<Dataset>> {
    if entry.refusal.is_some() {
        return Ok(None);
    }
    let path = root.join(format!("data/{}.json", id(entry)?));
    let expected = entry.artifact.as_ref().ok_or("missing accepted artifact")?;
    let actual = io::artifact(&path)?;
    if actual.sha256 != expected.sha256 || actual.bytes != expected.bytes as u64 {
        return Err("source dataset bytes changed".into());
    }
    let data: Dataset = io::read(&path)?;
    probe(data.validate(&entry.source))?;
    Ok(Some(data))
}
fn vector<T>(v: &Vec<T>) -> u128 {
    v.capacity() as u128 * size_of::<T>() as u128
}
pub(crate) fn owned(data: &Dataset) -> u128 {
    let mut bytes = size_of::<Dataset>() as u128
        + data.name.capacity() as u128
        + data.selection.capacity() as u128;
    for (name, split) in &data.splits {
        // BTreeMap node allowance includes keys, values, links and spare slots.
        bytes += 512 + name.capacity() as u128 + vector(&split.records);
        for row in &split.records {
            bytes += row.task.capacity() as u128
                + vector(&row.tokens)
                + vector(&row.targets)
                + vector(&row.changes)
                + vector(&row.prompt)
                + vector(&row.answer);
        }
        if let Some(words) = &split.teacher_words {
            bytes += vector(words);
            for row in words {
                bytes += vector(row);
            }
        }
    }
    bytes
}
pub(crate) fn training<'a>(
    data: &'a Dataset,
    work: &mut Work,
) -> Result<model::input::Prepared<'a>> {
    model::input::Prepared::new(
        probe(data.training())?,
        owned(data) + OTHER_BYTES,
        resources::PROCESS,
        work,
    )
}
