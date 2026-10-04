"""Attention observations for EXPERIMENT_PLAN.md, step 2.

Temporary pre-hooks reconstruct each head's scores and probabilities during
one uncompiled, no-gradient forward on a fixed validation batch. Support is
strictly positive weight (arXiv:1602.02068v2, section 2.2), with a sparsemax
shadow under every normalizer. These measure an eager floating-point model,
whose rounding can differ from compiled or fused attention. For fused
softmax, probabilities are the explicit softmax reference of its scores;
QKNorm's scale multiplies queries, as in `QKNormScores.attend`.

Self-only routes also report value norms below XSA's normalization epsilon:
the exact self-cancellation identity need not hold for these clipped values.
The forward draws no random numbers, builds no model and takes no optimizer
step. Hooks and every module's training flag are restored even on failure.
Only the observer's previous causal support masks change; they are kept on
CPU and saved with the checkpoint to continue turnover across sessions.
"""

import torch

from ..nn.attention import Attention, QKNormScores, softmax
from ..nn.sparsemax import causal_sparsemax
from .attention_batch import FixedBatch, answer_routes
from .attention_stats import statistics


class AttentionObserver:
    def __init__(self, spec, task):
        self.fixed = FixedBatch(task, spec.examples)
        self.previous, self.step = {}, None

    def state_dict(self):
        return {"rows_sha256": self.fixed.fingerprint, "step": self.step, "supports": self.previous}

    def load_state_dict(self, state, step):
        if state["rows_sha256"] != self.fixed.fingerprint or state["step"] != step:
            raise ValueError("Checkpointed attention supports do not describe this observation's fixed rows and step")
        self.step = state["step"]
        self.previous = {name: {kind: support.cpu() for kind, support in maps.items()}
                         for name, maps in state["supports"].items()}

    @torch.no_grad()
    def observe(self, model, task, step):
        """Measure the current model once; atomically replace the observer's previous supports."""
        layers, supports, hooks = {}, {}, []
        fixed = self.fixed

        def capture(name, module, arguments):
            x, rotary = arguments
            grouped_scores, grouped_weights, grouped_shadow, grouped_norms = [], [], [], []
            for q, k, value, index in module.projections(x):
                if module.fused and isinstance(module.scores, QKNormScores):
                    q, k, alpha = module.scores.unit(q, k, rotary, index)
                    scores = (q * alpha) @ k.transpose(-2, -1)
                else:
                    scores = module.scores(q, k, rotary, index)
                weights = softmax(scores) if module.fused else module.weights(scores)
                shadow = weights if module.weights is causal_sparsemax else causal_sparsemax(scores)
                norm = value.norm(dim=-1)
                if scores.ndim == 3:
                    scores, weights, shadow = (tensor.unsqueeze(1) for tensor in (scores, weights, shadow))
                    norm = norm.unsqueeze(1)
                grouped_scores.append(scores)
                grouped_weights.append(weights)
                grouped_shadow.append(shadow)
                grouped_norms.append(norm)
            scores, weights, shadow = (torch.cat(group, dim=1) for group in
                                       (grouped_scores, grouped_weights, grouped_shadow))
            old = self.previous.get(name, {})
            actual, actual_support = statistics(scores, weights, fixed.lengths, fixed.supervised, old.get("actual"))
            projected, projected_support = statistics(scores, shadow, fixed.lengths, fixed.supervised,
                                                      old.get("shadow_sparsemax"))
            layers[name] = {"actual": actual, "shadow_sparsemax": projected,
                            "answer_routes": answer_routes(weights, fixed.queries),
                            "shadow_answer_routes": answer_routes(shadow, fixed.queries),
                            "fused_softmax_reference": module.fused}
            if module.exclusive is not None:
                positions = torch.arange(weights.shape[-1], device=weights.device)
                valid = positions[None, :] < fixed.lengths[:, None]
                self_only = ((weights > 0).sum(-1) == 1) & (weights.diagonal(dim1=-2, dim2=-1) > 0)
                clipped = self_only & valid[:, None] & (torch.cat(grouped_norms, dim=1) < module.exclusive)
                layers[name]["xsa_self_only_below_epsilon_rows"] = {
                    "all_rows": clipped.sum((0, 2)).cpu().tolist(),
                    "nontrivial_rows": (clipped & (positions > 0)).sum((0, 2)).cpu().tolist()}
            supports[name] = {"actual": actual_support, "shadow_sparsemax": projected_support}

        modes = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            for name, module in model.named_modules():
                if isinstance(module, Attention):
                    hooks.append(module.register_forward_pre_hook(
                        lambda layer, arguments, name=name: capture(name, layer, arguments)))
            output, targets = task.forward(model, fixed.batch)
            if not torch.equal(targets, fixed.targets):
                raise ValueError("The attention forward must use the fixed supervised targets")
            predictions = fixed.predictions(output)
        finally:
            for hook in hooks:
                hook.remove()
            for module, training in modes:
                module.training = training
        self.previous, self.step = supports, step
        return {"step": step, "split": fixed.split, "examples": len(fixed.batch),
                "rows_sha256": fixed.fingerprint, "forward": "uncompiled_teacher_forced",
                **predictions, "layers": layers}
