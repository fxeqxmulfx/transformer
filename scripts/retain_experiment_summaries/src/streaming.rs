//! Stream past large tensors instead of allocating them merely to remove
//! them. Retained arrays/maps have bounded sizes; omissions are explicit.
use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value, json};
use std::fmt;
struct Seed {
    payload: bool,
}
impl<'de> DeserializeSeed<'de> for Seed {
    type Value = Value;
    fn deserialize<D: de::Deserializer<'de>>(self, d: D) -> Result<Value, D::Error> {
        d.deserialize_any(self)
    }
}
impl<'de> Visitor<'de> for Seed {
    type Value = Value;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a JSON experiment summary")
    }
    fn visit_bool<E: de::Error>(self, x: bool) -> Result<Value, E> {
        Ok(Value::Bool(x))
    }
    fn visit_i64<E: de::Error>(self, x: i64) -> Result<Value, E> {
        Ok(Value::Number(x.into()))
    }
    fn visit_u64<E: de::Error>(self, x: u64) -> Result<Value, E> {
        Ok(Value::Number(x.into()))
    }
    fn visit_f64<E: de::Error>(self, x: f64) -> Result<Value, E> {
        Number::from_f64(x)
            .map(Value::Number)
            .ok_or_else(|| E::custom("nonfinite JSON metric"))
    }
    fn visit_str<E: de::Error>(self, x: &str) -> Result<Value, E> {
        Ok(Value::String(x.into()))
    }
    fn visit_string<E: de::Error>(self, x: String) -> Result<Value, E> {
        Ok(Value::String(x))
    }
    fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_none<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        if self.payload {
            let mut count = 0usize;
            while seq.next_element::<IgnoredAny>()?.is_some() {
                count += 1;
            }
            return Ok(json!({"omitted_payload_items":count}));
        }
        let mut items = vec![];
        let mut count = 0usize;
        let mut first = None;
        let mut last = None;
        let mut scalar = true;
        while let Some(value) = seq.next_element_seed(Seed { payload: false })? {
            scalar &= !value.is_array() && !value.is_object();
            if count == 0 {
                first = Some(value.clone());
            }
            last = Some(value.clone());
            if count < 512 {
                items.push(value);
            }
            count += 1;
        }
        if count > 512 || (count > 64 && scalar) {
            Ok(json!({"omitted_array_items":count,"first":first,"last":last}))
        } else {
            Ok(Value::Array(items))
        }
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        if self.payload {
            let mut count = 0usize;
            while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {
                count += 1;
            }
            return Ok(json!({"omitted_payload_fields":count}));
        }
        let mut result = Map::new();
        let mut count = 0usize;
        while let Some(key) = map.next_key::<String>()? {
            count += 1;
            if count <= 2048 {
                let value = map.next_value_seed(Seed {
                    payload: crate::compact::payload(&key),
                })?;
                result.insert(key, value);
            } else {
                map.next_value::<IgnoredAny>()?;
            }
        }
        if count > 2048 {
            Ok(json!({"omitted_object_fields":count}))
        } else {
            Ok(Value::Object(result))
        }
    }
}
pub fn read(reader: impl std::io::Read) -> serde_json::Result<Value> {
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    let result = Seed { payload: false }.deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(result)
}
pub fn finding(reader: impl std::io::Read) -> serde_json::Result<Value> {
    match read(reader) {
        Ok(summary) => Ok(json!({"summary":summary,"incomplete_report":false,"parse_error":null})),
        Err(error) if !error.is_io() => {
            Ok(json!({"summary":null,"incomplete_report":true,"parse_error":error.to_string()}))
        }
        Err(error) => Err(error),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incomplete_reports_are_explicitly_unscored_and_io_errors_still_stop() {
        for bytes in [&b""[..], &b"{\"accuracy\":1."[..]] {
            let kept = finding(bytes).unwrap();
            assert_eq!(kept["incomplete_report"], true);
            assert!(kept["summary"].is_null());
            assert!(kept["parse_error"].is_string());
        }
        struct Broken;
        impl std::io::Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("read failure"))
            }
        }
        assert!(finding(Broken).is_err());
    }
    #[test]
    fn streaming_matches_independent_tree_compaction_with_large_nested_payloads() {
        let source = json!({"records":(0..600).map(|i|json!({"case":i,"targets":2000,"metrics":{"correct":1999,"loss":0.003},
            "model":{"weights":vec![1.234;300]},"training":{"rows":[[1,2,3]]}})).collect::<Vec<_>>(),
            "counts":(0..100).collect::<Vec<_>>(),"modes":["Forward","Full","Mini"],"source_commit":"abcdef"});
        let bytes = serde_json::to_vec(&source).unwrap();
        assert_eq!(
            read(bytes.as_slice()).unwrap(),
            crate::compact::value(source, "")
        );
        assert!(read(&b"{\"complete\":true} trailing"[..]).is_err());
    }
}
