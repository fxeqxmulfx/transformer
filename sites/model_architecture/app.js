"use strict";

// Recorded evidence is generated from the repository reports by build.py.
const state = { mode: "scaled", prime: 97, part: "attention", evidence: null, evidenceError: false };
const byId = id => document.getElementById(id);
const put = (id, value) => { byId(id).textContent = value; };
const percent = value => `${(100 * value).toFixed(3)}%`;

const components = {
  attention: {
    tag: "Attention", title: "Normalize, rotate, retrieve.",
    copy: "Normalize each query and key, then apply geometric RoPE. Causal softmax combines values from the current and earlier positions. Centered XSA applies a clipped removal of the component along the current value.",
    formula: "z = normalized branch input\nq = RoPE(RMS(Wq z))\nk = RoPE(RMS(Wk z))\np = causal_softmax(qkᵀ / √32)\nu = v / max(‖v‖, 10⁻¹²)\nXSA = (pv) − ⟨pv, u⟩u",
    note: "Four heads of width 32. RoPE base 10,000, in every block. The source zeros the first attention output and retains its numerical repeated-prefix guard. This cohort did not ablate Q/K normalization.",
  },
  norm: {
    tag: "Input normalization", title: "Control scale, without learned gain.",
    copy: "Parameter-free RMSNorm prepares the input of each branch. The embedding and final residual stream use the same normalization convention.",
    formula: "N(x) = x / √(mean(x²) + ε)\nε = 2⁻²³ in measured FP32\nlearned gain = none\nbias = none",
    note: "Q/K normalization acts separately within each attention head. Torch 2.15 uses FP32 epsilon for FP16/BF16/FP32 and FP64 epsilon for FP64. The current backward separates radial/tangent terms in FP64; mixed-precision grokking is unmeasured.",
  },
  outputnorm: {
    tag: "Output normalization", title: "Normalize first. Apply strength second.",
    copy: "Both the attention and feed-forward output pass through RMSNorm before the branch coefficient is applied. This preserves a direct route for the coefficient to change the branch amplitude.",
    formula: "branch = N(Branch(N(h)))\nh_next = h + α · branch",
    note: "A gain before RMS can largely disappear through normalization. The finite-epsilon derivatives and synthetic recovery benchmark explain why placement matters.",
  },
  ffn: {
    tag: "Feed-forward", title: "A gated feature transformation.",
    copy: "SwiGLU combines a SiLU gate and a value projection, then maps back to width 128. Its input is the updated residual stream after attention.",
    formula: "g, v = W_in · N(h)\nf = W_out · (SiLU(g) ⊙ v)\n128 → 2 × 341 → 128",
    note: "The fused input matrix has 682 outputs. All linear layers have no bias. SwiGLU’s feature gate is distinct from the optional scalar residual gate.",
  },
  skip: {
    tag: "Residual stream", title: "Keep the identity path.",
    copy: "The input bypasses each normalized branch and is added with coefficient one. The feed-forward branch receives the result of the attention addition.",
    formula: "h_next = h + α · F(h)\nJacobian = I + α · J_F(h)",
    note: "An identity term alone does not guarantee gradient transport: the branch Jacobian can cancel it. The Lean witnesses describe this possibility; they do not prove convergence of a full transformer.",
  },
};

function gainNotes() {
  if (state.mode === "scaled") return {
    tag: "Fixed branch strength", title: "A smaller step after each branch.",
    copy: "Multiply each normalized branch by 1/√(2L). At four blocks, both attention and feed-forward use the fixed coefficient 0.353553.",
    formula: "α = 1 / √(2 × 4)\nα ≈ 0.353553\nh_next = h + α · N(Branch(N(h)))",
    note: state.evidence?.continuation?.length ? "Both P97 continuations reached 99% validation at 24k updates, versus 32k/37k for ordinary addition. The final test scores were 99.227%/98.711%. The 99.9% stability target is still unconfirmed." : "This moved P97 generalization earlier in both measured seeds. The runs ended at 23k updates before the final stability target. Their eventual accuracy is unknown.",
  };
  if (state.mode === "gated") return {
    tag: "Learned branch strength", title: "Learn amplitude after normalization.",
    copy: "One scalar logit per attention and feed-forward branch controls a bounded nonnegative coefficient. Eight new parameters at depth four, all initialized to zero.",
    formula: "α = 2 · sigmoid(s)\ns₀ = 0  →  α₀ = 1\n0 ≤ α ≤ 2 in floating point",
    note: "The current implementation does not also apply 1/√(2L). Synthetic recovery and actual GPT integration pass; a full-model grokking advantage has not been measured.",
  };
  return {
    tag: "Ordinary branch strength", title: "The measured accuracy baseline.",
    copy: "Add the normalized branch with coefficient one. No extra trainable parameters or depth-aggregation mechanism are introduced.",
    formula: "α = 1\nh_next = h + N(Branch(N(h)))",
    note: "Both P97 and P193 controls reached the prespecified stable-validation target for seeds 11 and 17. P97 needed longer training than P193 in these runs.",
  };
}

function inspect(part) {
  state.part = part;
  const details = part === "gain" ? gainNotes() : components[part];
  for (const [field, value] of Object.entries(details)) put(`component-${field}`, value);
  document.querySelectorAll(".node").forEach(node => {
    const selected = node.dataset.part === part;
    node.classList.toggle("selected", selected);
    node.setAttribute("aria-pressed", String(selected));
  });
}

function renderProfile() {
  const { mode, prime, evidence } = state;
  const measured = evidence?.cases.filter(row => row.mode === mode && row.prime === prime) ?? [];
  const badge = byId("profile-badge");
  badge.classList.toggle("experimental", mode !== "add");
  if (mode === "add") {
    put("profile-badge", "Default · measured");
    put("profile-heading", "The accuracy baseline.");
    put("profile-description", "All four ordinary-addition controls reached stable validation. Keep this as the reference when changing the residual path.");
    put("result-label", `Final test · both P${prime} seeds`);
    put("profile-limit", prime === 97 ? "Stable validation confirmed at 40k and 45k updates. The reported test score is on unseen operand pairs at the same prime." : "Stable validation confirmed at 19k and 11k updates. The larger prime was a separately trained task, not a transfer test.");
  } else if (mode === "scaled") {
    const continued = evidence?.continuation?.length;
    put("profile-badge", prime === 97 ? "Earlier transition · measured" : "Unmeasured at P193");
    put("profile-heading", continued ? "Earlier at the 99% validation target." : "An earlier transition, with an open endpoint.");
    put("profile-description", continued ? "Both P97 seeds crossed 99% validation at 24k updates. Ordinary addition first crossed it at 32k/37k: scaling needed 25%/35% fewer updates." : "At update 23k on P97, scaled validation accuracy was 97.595–98.368%; ordinary addition was 27.234–34.966% at that same update.");
    put("result-label", prime === 97 ? `Final test · P97 at update ${continued ? "24k" : "23k"}` : "No scaled P193 run in this cohort");
    put("profile-limit", continued ? "The continuations stopped at the requested first 99% validation crossing. Seed 17’s test score remains below 99%. The original test split was reused; 99.9% stability is still unconfirmed." : "The scaled P97 runs stopped when their time allowance ended. Neither confirmed the 99.9% stability target; final convergence and speed to that target remain unknown.");
  } else {
    put("profile-badge", "Experimental · synthetic evidence");
    put("profile-heading", "A gradient-path candidate.");
    put("profile-description", "The after-RMS gain recovered every tested synthetic amplitude. The optional GPT implementation passes integration tests, but has no full-transformer grokking comparison yet.");
    put("result-label", "Full-model generalization result");
    put("profile-limit", "Synthetic known-solution recovery tests gate placement. It does not establish better held-out accuracy, faster grokking or superiority to ordinary addition.");
  }
  const values = [...new Set(measured.map(row => row.test_accuracy))].sort((a, b) => a - b);
  put("result-value", state.evidenceError ? "Unavailable" : values.length ? (values.length === 1 ? percent(values[0]) : `${(100 * values[0]).toFixed(3)}–${percent(values.at(-1))}`) : evidence ? "Unmeasured" : "Loading…");
  byId("result-value").classList.toggle("long", byId("result-value").textContent.length > 9);
}

function svgElement(name, attributes, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderChart() {
  if (!state.evidence) return;
  const chart = byId("learning-chart");
  chart.replaceChildren();
  const cases = state.evidence.cases.filter(row => row.prime === state.prime);
  const maxStep = state.prime === 97 ? 45000 : 20000;
  const x = step => 44 + step / maxStep * 638;
  const y = value => 231 - value * 210;
  const add = (name, attributes, text) => chart.append(svgElement(name, attributes, text));
  add("title", {}, `Recorded P${state.prime} validation accuracy; no extrapolated endpoints`);
  for (const accuracy of [0, 0.25, 0.5, 0.75, 1]) {
    add("line", { x1: 44, x2: 682, y1: y(accuracy), y2: y(accuracy), stroke: "#e6ece8", "stroke-width": 1 });
    add("text", { x: 32, y: y(accuracy) + 4, fill: "#728583", "font-size": 10, "text-anchor": "end" }, `${accuracy * 100}%`);
  }
  const ticks = state.prime === 97 ? [0, 10000, 20000, 30000, 40000, 45000] : [0, 5000, 10000, 15000, 20000];
  for (const step of ticks) add("text", { x: x(step), y: 251, fill: "#728583", "font-size": 10, "text-anchor": "middle" }, step ? `${step / 1000}k` : "0");
  add("text", { x: 360, y: 273, fill: "#728583", "font-size": 10, "text-anchor": "middle" }, "Optimizer updates");
  for (const row of cases) {
    const color = row.mode === "add" ? "#172e35" : "#0a7d72";
    const line = svgElement("polyline", {
      points: row.curve.map(point => `${x(point.step)},${y(point.accuracy)}`).join(" "),
      fill: "none", stroke: color, "stroke-width": 2.2,
      "stroke-dasharray": row.seed === 17 ? "5 4" : "none",
      "stroke-linejoin": "round", "data-label": row.label,
    });
    line.append(svgElement("title", {}, `${row.mode}, seed ${row.seed}: ${percent(row.validation_accuracy)} validation; ${row.stop.step.toLocaleString()} updates`));
    chart.append(line);
    const end = row.curve.at(-1);
    add("circle", { cx: x(end.step), cy: y(end.accuracy), r: 3, fill: color, stroke: "white", "stroke-width": 1 });
  }
  put("chart-caption", `Validation accuracy · P${state.prime} · two initialization seeds`);
  byId("learning-chart").setAttribute("aria-label", `Measured P${state.prime} validation curves for ordinary addition${state.prime === 97 ? " and fixed scaling" : " only"}. Solid is seed 11, dashed is seed 17.`);
  document.querySelector(".legend-scaled").parentElement.hidden = state.prime !== 97;
  const note = state.prime === 97 ? state.evidence.continuation?.length ? "Scaled curves include the full-state continuations to 24k, stopping at the first 99% validation crossing. The original 23k endpoints remain in the archived report." : "Scaled curves end at 23k updates because their time allowance ended. Lines connect recorded observations; they do not extrapolate." : "Only ordinary addition was measured at P193 in this comparison. Both seeds reached stable validation.";
  put("chart-note", `${note}${state.mode === "gated" ? " No learned-gate curve is available; these are reference curves." : ""}`);
}

function render() {
  const { mode, prime } = state;
  document.querySelectorAll(".mode").forEach(button => {
    button.classList.toggle("active", button.dataset.mode === mode);
    button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  });
  const parameters = prime * 128 + 4 * (4 * 128 ** 2 + 3 * 128 * 341) + (mode === "gated" ? 8 : 0);
  put("parameter-count", parameters.toLocaleString("en-US"));
  put("readout-shape", `[B, ${prime}] final logits`);
  put("initial-gain", mode === "scaled" ? "0.353553" : "1");
  document.querySelectorAll(".gain-value").forEach(node => {
    node.textContent = mode === "scaled" ? "× 0.3536" : mode === "gated" ? "× 2σ(s)" : "× 1";
  });
  put("branch-formula", mode === "add" ? "h ← h + N(Branch(N(h)))" : `h ← h + ${mode === "scaled" ? "1/√8" : "2σ(s)"} · N(Branch(N(h)))`);
  put("model-code", `from diff_gpt.domain.model.gpt import GPT\n\nmodel = GPT(\n    vocab_size=${prime}, n_embd=128,\n    n_head=4, n_layer=4, block_size=33,\n    rope_base=10_000.0,\n    pad_vocab_size_to=1, use_checkpoint=False,\n).float()`);
  put("code-note", `Current GPT always uses fixed scaling and the accurate RMS backward. Supervise final-position logits only.${mode === "scaled" ? "" : " The selected diagram is a historical comparison; replay it at its original Git revision."}`);
  put("copy-config", "Copy config");
  const recovery = state.evidence?.rms_backward;
  put("rms-recovery", recovery ? `${recovery.accurate}/4 recovered · native ${recovery.native}/4` : "Loading recorded recovery…");
  byId("continuation-download").hidden = !state.evidence?.continuation?.length;
  inspect(state.part);
  renderProfile();
  renderChart();
}

document.querySelectorAll(".mode").forEach(button => button.addEventListener("click", () => {
  state.mode = button.dataset.mode;
  render();
}));
byId("prime").addEventListener("change", event => {
  state.prime = Number(event.target.value);
  render();
});
document.querySelectorAll(".node").forEach(node => {
  node.addEventListener("click", () => inspect(node.dataset.part));
  node.addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      inspect(node.dataset.part);
    }
  });
});
byId("copy-config").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(byId("model-code").textContent);
    put("copy-config", "Copied");
  } catch {
    const range = document.createRange();
    range.selectNodeContents(byId("model-code"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    put("copy-config", "Code selected · copy manually");
  }
});
render();
fetch("evidence.json").then(response => {
  if (!response.ok) throw new Error(`Evidence request failed: ${response.status}`);
  return response.json();
}).then(evidence => {
  state.evidence = evidence;
  for (const placement of ["before", "after"]) {
    const result = evidence.synthetic_gate_recovery[placement];
    put(`synthetic-${placement}`, `${result.recovered}/${result.total}`);
  }
  render();
  document.body.dataset.evidenceReady = "true";
}).catch(error => {
  state.evidenceError = true;
  put("result-value", "Unavailable");
  put("chart-note", "Recorded evidence could not be loaded. Restart with ./run.py site to rebuild the local assets.");
  document.body.dataset.evidenceReady = "error";
  console.error(error);
});
