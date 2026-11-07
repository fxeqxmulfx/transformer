//! Native port of scripts/index.py at 22e6bbe2. Preserve its index conventions.
use regex::Regex;
use std::collections::BTreeMap;

pub type Incomplete = BTreeMap<(String, usize), String>;

#[derive(Debug)]
pub struct Declaration {
    pub name: String,
    pub kind: String,
    pub status: &'static str,
    pub line: usize,
}

pub fn code(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = bytes.to_vec();
    let mut i = 0;
    while i < bytes.len() {
        let (a, b, end) = if bytes[i..].starts_with(b"--") {
            let end = bytes[i..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(bytes.len(), |n| i + n);
            (i, end, end)
        } else if bytes[i..].starts_with(b"/-") {
            let a = i + 2;
            let mut end = a;
            let mut depth = 1;
            while end < bytes.len() && depth > 0 {
                if bytes[end..].starts_with(b"/-") {
                    depth += 1;
                    end += 2;
                } else if bytes[end..].starts_with(b"-/") {
                    depth -= 1;
                    end += 2;
                } else {
                    end += 1;
                }
            }
            (a, end, end)
        } else if bytes[i] == b'"' {
            let a = i + 1;
            let mut end = a;
            while end < bytes.len() && bytes[end] != b'"' {
                end += if bytes[end] == b'\\' && end + 1 < bytes.len() {
                    2
                } else {
                    1
                };
            }
            (a, end, (end + 1).min(bytes.len()))
        } else {
            i += 1;
            continue;
        };
        for byte in &mut out[a..b] {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
        i = end;
    }
    String::from_utf8(out).expect("comment blanking preserves UTF-8")
}

pub struct Scanner {
    declaration: Regex,
    stop: Regex,
    trivial: Regex,
    arxiv: Regex,
}

impl Scanner {
    pub fn new() -> Self {
        Self {
            declaration: Regex::new(r"(?m)^(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+)*(theorem|lemma|def|abbrev|structure|class|instance)\s+([^\s({\[:]+)").unwrap(),
            stop: Regex::new(r"(?m)^(?:end|namespace|section|open|variable|universe|attribute|set_option|/-|#)\b").unwrap(),
            trivial: Regex::new(r"^(?:by\s+)?(?:exact\s+)?(?:(?:∀|∃)[^,]*,\s*)*(?:True|0|∅|⊥)$").unwrap(),
            arxiv: Regex::new(r"arXiv:(\d{4}\.\d{4,5})").unwrap(),
        }
    }

    pub fn scan(
        &self,
        path: &str,
        source: &str,
        incomplete: &mut Incomplete,
    ) -> (Vec<Declaration>, Option<String>) {
        let text = code(source);
        let marks: Vec<_> = self.declaration.captures_iter(&text).collect();
        let preamble_end = marks
            .first()
            .map_or(source.len(), |m| m.get(0).unwrap().start());
        let paper = self
            .arxiv
            .captures(&source[..preamble_end])
            .map(|m| m[1].to_owned());
        let mut out = Vec::new();
        for (i, mark) in marks.iter().enumerate() {
            let span = mark.get(0).unwrap();
            let end = marks
                .get(i + 1)
                .map_or(text.len(), |m| m.get(0).unwrap().start());
            let end = self
                .stop
                .find(&text[span.end()..end])
                .map_or(end, |s| span.end() + s.start());
            let whole = text[span.start()..end]
                .replace("/-", " ")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            let (signature, body) = whole.rsplit_once(":=").unwrap_or(("", &whole));
            let line_at = |offset| text[..offset].bytes().filter(|&b| b == b'\n').count() + 1;
            let kind = &mark[1];
            let status = if incomplete
                .remove(&(path.to_owned(), line_at(mark.get(2).unwrap().start())))
                .is_some()
            {
                "sorry"
            } else if matches!(kind, "theorem" | "lemma") {
                if signature.trim_end().ends_with("True") {
                    "vacuous"
                } else {
                    "proved"
                }
            } else if matches!(kind, "def" | "abbrev") && self.trivial.is_match(body.trim()) {
                "placeholder"
            } else {
                ""
            };
            out.push(Declaration {
                name: mark[2].to_owned(),
                kind: kind.to_owned(),
                status,
                line: line_at(span.start()),
            });
        }
        (out, paper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_comments_strings_and_unicode_do_not_create_declarations() {
        let text = "/- α /- nested -/\ntheorem fake : True := by trivial\n-/\ndef s := \"-- /- \\\"\"\n-- theorem fake2\n@[simp]\ntheorem actual : 1 = 1 := by rfl\n";
        let (ds, _) = Scanner::new().scan("fixture", text, &mut Incomplete::new());
        assert_eq!(
            ds.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(),
            ["s", "actual"]
        );
        assert_eq!(ds[1].line, 6);
        assert_eq!(ds[1].status, "proved");
        assert_eq!(code(text).len(), text.len());
    }

    #[test]
    fn build_positions_override_syntax_and_keep_attribute_lines() {
        let text = "@[simp]\ntheorem pending : True := by sorry\ndef empty : Nat := 0\ntheorem vacuous : ∃ x : Nat, True := by trivial\n";
        let mut incomplete = Incomplete::from([(("fixture".into(), 2), "pending".into())]);
        let (ds, _) = Scanner::new().scan("fixture", text, &mut incomplete);
        assert!(incomplete.is_empty());
        assert_eq!(
            ds.iter().map(|d| d.status).collect::<Vec<_>>(),
            ["sorry", "placeholder", "vacuous"]
        );
        assert_eq!(ds[0].line, 1);
    }
}
