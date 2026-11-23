//! Adapter to the pinned GPT scorer. Only the outer CE sees target IDs.
use crate::{metrics::Packed, text::Model};
pub use minibatch::Result;
use minibatch::random::Random;
use tch::{Device, Kind, Tensor, nn};

pub struct Config {
    pub vocab: usize,
    pub max_batch: usize,
}
pub struct Predictions(Tensor);
impl Predictions {
    pub fn read(&self) -> Result<Vec<u32>> {
        Ok(Vec::<i64>::try_from(&self.0.view([-1]))?
            .into_iter()
            .map(|v| v as u32)
            .collect())
    }
}
pub struct Logits(Tensor);
impl Logits {
    pub fn read(&self) -> Result<Vec<f32>> {
        Ok(Vec::<f32>::try_from(&self.0.reshape([-1]))?)
    }
}
pub struct Engine {
    pub config: Config,
    pub variables: nn::VarStore,
    pub model: Model,
    pub predictions: Predictions,
    pub logits: Logits,
    loss: Tensor,
}
impl Engine {
    pub fn new(width: usize, heads: usize, vocab: usize, seed: u32) -> Result<Self> {
        if width < 2 || !width.is_multiple_of(2) || heads == 0 || !width.is_multiple_of(heads) {
            return Err("invalid text model dimensions".into());
        }
        tch::manual_seed(i64::from(seed));
        let variables = nn::VarStore::new(Device::Cpu);
        let mut model = Model::new(&variables.root(), vocab as i64, width as i64, heads as i64);
        let mut rng = Random::new(seed);
        let initial: Vec<_> = (0..vocab * width)
            .map(|_| rng.gauss(0., 0.02) as f32)
            .collect();
        tch::no_grad(|| {
            model
                .embedding
                .copy_(&Tensor::from_slice(&initial).view([vocab as i64, width as i64]))
        });
        Ok(Self {
            config: Config {
                vocab,
                max_batch: if crate::text::RELEASE { 32 } else { 64 },
            },
            variables,
            model,
            predictions: Predictions(Tensor::new()),
            logits: Logits(Tensor::new()),
            loss: Tensor::new(),
        })
    }
    fn inputs(&self, tokens: &[u32], b: usize, t: usize) -> Result<Tensor> {
        if b.checked_mul(t) != Some(tokens.len())
            || b == 0
            || t == 0
            || tokens.iter().any(|&x| x as usize >= self.config.vocab)
        {
            return Err("invalid token batch".into());
        }
        let ids: Vec<i64> = tokens.iter().map(|&x| i64::from(x)).collect();
        Ok(Tensor::from_slice(&ids).view([b as i64, t as i64]))
    }
    fn targets(&self, labels: &[u32]) -> Tensor {
        let ids: Vec<i64> = labels
            .iter()
            .map(|&y| {
                if (y as usize) < self.config.vocab {
                    i64::from(y)
                } else {
                    -100
                }
            })
            .collect();
        Tensor::from_slice(&ids)
    }
    pub fn train_loss(&self, p: &Packed, batch: usize) -> Result<Tensor> {
        let inputs = self.inputs(&p.tokens, batch, p.length)?;
        let logits = self.model.logits(&inputs);
        Ok(logits
            .view([-1, self.config.vocab as i64])
            .cross_entropy_for_logits(&self.targets(&p.labels)))
    }
    pub fn forward(&mut self, tokens: &[u32], labels: &[u32], b: usize, t: usize) -> Result<()> {
        let inputs = self.inputs(tokens, b, t)?;
        if labels.len() != tokens.len() {
            return Err("invalid evaluation label buffer".into());
        }
        let scores = tch::no_grad(|| self.model.logits(&inputs));
        self.predictions = Predictions(scores.argmax(-1, false));
        let target = self.targets(labels);
        self.loss = -scores
            .reshape([-1, self.config.vocab as i64])
            .log_softmax(-1, Kind::Float)
            .gather(-1, &target.clamp_min(0).unsqueeze(-1), false)
            .view([-1])
            * target.ge(0);
        self.logits = Logits(scores);
        Ok(())
    }
    pub fn losses(&self, n: usize) -> Result<Vec<f32>> {
        if n != self.loss.numel() {
            return Err("evaluation loss shape mismatch".into());
        }
        Ok(Vec::<f32>::try_from(&self.loss)?)
    }
}
