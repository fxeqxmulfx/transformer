//! Port of the released Google DeepMind MRCR V2 grader. Matching operates on
//! Unicode characters. The second sequence supplies the popularity histogram.
use std::{collections::HashMap, ops::Range};

fn trim(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

pub fn grade(prediction: &str, target: &str) -> f64 {
    let target = trim(target);
    if prediction.is_empty() || target.chars().take(12).count() < 12 {
        return 0.0;
    }
    let end = target
        .char_indices()
        .nth(12)
        .map_or(target.len(), |(i, _)| i);
    let certificate = &target[..end];
    let prediction = trim(prediction);
    let Some(last) = prediction.rfind(certificate) else {
        return 0.0;
    };
    ratio(trim(&target[end..]), trim(&prediction[last + end..]))
}

// Longest contiguous matching block in each subproblem, breaking ties by
// earliest reference start and then earliest prediction start. As in
// SequenceMatcher, popular characters can extend a block from either end.
struct Matcher {
    reference: Vec<char>,
    prediction: Vec<char>,
    positions: HashMap<char, Vec<usize>>,
}

impl Matcher {
    fn new(a: &str, b: &str) -> Self {
        let prediction: Vec<_> = b.chars().collect();
        let mut positions: HashMap<_, Vec<_>> = HashMap::new();
        for (i, &c) in prediction.iter().enumerate() {
            positions.entry(c).or_default().push(i);
        }
        if prediction.len() >= 200 {
            let cutoff = prediction.len() / 100 + 1;
            positions.retain(|_, indices| indices.len() <= cutoff);
        }
        Self {
            reference: a.chars().collect(),
            prediction,
            positions,
        }
    }

    fn longest(&self, a: Range<usize>, b: Range<usize>) -> (usize, usize, usize) {
        let (mut left, mut right, mut length) = (a.start, b.start, 0);
        let mut previous = HashMap::<usize, usize>::new();
        let mut current = HashMap::new();
        for i in a.clone() {
            current.clear();
            if let Some(js) = self.positions.get(&self.reference[i]) {
                let first = js.partition_point(|&j| j < b.start);
                for &j in js[first..].iter().take_while(|&&j| j < b.end) {
                    let k = 1 + previous.get(&j).copied().unwrap_or(0);
                    current.insert(j + 1, k);
                    if k > length {
                        (left, right, length) = (i + 1 - k, j + 1 - k, k);
                    }
                }
            }
            std::mem::swap(&mut previous, &mut current);
        }
        while left > a.start
            && right > b.start
            && self.reference[left - 1] == self.prediction[right - 1]
        {
            left -= 1;
            right -= 1;
            length += 1;
        }
        while left + length < a.end
            && right + length < b.end
            && self.reference[left + length] == self.prediction[right + length]
        {
            length += 1;
        }
        (left, right, length)
    }

    fn matched(&self) -> usize {
        let mut blocks = vec![(0..self.reference.len(), 0..self.prediction.len())];
        let mut total = 0;
        while let Some((a, b)) = blocks.pop() {
            let (i, j, length) = self.longest(a.clone(), b.clone());
            if length == 0 {
                continue;
            }
            total += length;
            if a.start < i && b.start < j {
                blocks.push((a.start..i, b.start..j));
            }
            if i + length < a.end && j + length < b.end {
                blocks.push((i + length..a.end, j + length..b.end));
            }
        }
        total
    }
}

fn ratio(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let matcher = Matcher::new(a, b);
    2.0 * matcher.matched() as f64 / (matcher.reference.len() + matcher.prediction.len()) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certificate_and_whitespace_follow_the_executable_grader() {
        let answer = "AbCd01234567λ🦉";
        assert_eq!(grade(" λ🦉", answer), 0.);
        assert_eq!(
            grade("anything AbCd01234567wrong AbCd01234567 λ🦉\u{1e}", answer),
            1.
        );
        assert_eq!(grade("AbCd01234567λ🦉 AbCd01234567", answer), 0.);
        assert_eq!(grade("", answer), 0.);
        assert_eq!(grade("short", "short"), 0.);
        assert_eq!(grade("AbCd01234567", "AbCd01234567"), 1.);
        assert_eq!(grade("🦉bCd01234567λ鹿", "🦉bCd01234567λ🦉"), 0.5);
    }

    #[test]
    fn sequence_matcher_ties_and_default_autojunk() {
        assert_eq!(ratio("tide", "diet"), 0.25);
        assert_eq!(ratio("diet", "tide"), 0.5);
        let common = "a".repeat(198);
        assert_eq!(
            ratio(&format!("x{common}"), &format!("y{common}")),
            198.0 / 199.0
        );
        let common = "a".repeat(199);
        assert_eq!(ratio(&format!("x{common}"), &format!("y{common}")), 0.);
        assert_eq!(
            ratio(&format!("x{common}z"), &format!("y{common}z")),
            200.0 / 201.0
        );
    }
}
