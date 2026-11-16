// Native ATen port of DiffGPT5a37c515 research/current_grokking/probes.py.
// Data construction only: no model, optimizer or Python interpreter is linked.
// Keeping the original CPU tensor operations preserves RNG and teacher rounding.
#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIndexing.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using at::indexing::Slice;

static void word(uint64_t value) {
  for (unsigned i = 0; i < 8; ++i) std::cout.put(char(value >> (8 * i)));
}

template <typename T> static void tensor(const at::Tensor &value) {
  auto contiguous = value.contiguous();
  const auto *data = contiguous.const_data_ptr<T>();
  for (int64_t i = 0; i < contiguous.numel(); ++i) {
    uint64_t bits;
    static_assert(sizeof(T) == sizeof(bits));
    std::memcpy(&bits, data + i, sizeof(bits));
    word(bits);
  }
}

static void output(const at::Tensor &inputs, const at::Tensor &labels,
                   const at::Tensor &teacher, uint64_t vocabulary) {
  if (inputs.dim() != 2 || labels.dim() != 2 || inputs.size(0) != labels.size(0))
    throw std::runtime_error("invalid complete probe tensor shape");
  std::cout.write("GPTPRB01", 8);
  word(inputs.size(0)); word(inputs.size(1)); word(labels.size(1));
  word(teacher.numel()); word(vocabulary);
  tensor<int64_t>(inputs); tensor<int64_t>(labels); tensor<double>(teacher);
  std::cout.flush();
  if (!std::cout) throw std::runtime_error("native probe output failed");
}

int main(int argc, char **argv) {
  try {
    if (argc != 3) throw std::runtime_error("expected exact probe and split names");
    const std::string name = argv[1], split = argv[2];
    // run.py and audit.py pin two intra-op threads before building each task.
    at::set_num_threads(2);
    at::set_num_interop_threads(1);
    const auto integers = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
    const auto doubles = at::TensorOptions().dtype(at::kDouble).device(at::kCPU);
    if (name == "order-discrimination") {
      if (split != "train" && split != "validation")
        throw std::runtime_error("order probe has only the original two splits");
      std::vector<int64_t> pairs;
      for (int64_t a = 0; a < 32; ++a)
        for (int64_t b = 0; b < 32; ++b)
          if (a != b) { pairs.push_back(a); pairs.push_back(b); }
      auto rng = at::detail::createCPUGenerator(1);
      auto x = at::from_blob(pairs.data(), {992, 2}, integers).clone();
      x = x.index_select(0, at::randperm(992, rng, integers));
      auto labels = (x.select(1, 0) < x.select(1, 1)).to(at::kLong) + 68;
      const int64_t start = split == "train" ? 0 : 496;
      output((x + 36).narrow(0, start, 496),
             labels.narrow(0, start, 496).unsqueeze(1), at::empty({0}, doubles), 70);
      return 0;
    }
    if (name != "routing-mixture" && name != "branch-amplitudes" &&
        name != "conditional-amplitudes" && name != "linear-transport")
      throw std::runtime_error("unknown numeric probe");
    if (split != "train" && split != "validation" && split != "test")
      throw std::runtime_error("unknown original split");
    const int64_t count = split == "train" ? 1024 : 256;
    const uint64_t seed = split == "train" ? 1001 : split == "validation" ? 2001 : 3001;
    auto rng = at::detail::createCPUGenerator(seed);
    at::Tensor inputs, answer;
    double step = 0.25;
    int64_t offset = 16, maximum = 32;
    if (name == "routing-mixture") {
      step = 1.0 / 32; offset = 128; maximum = 256;
      auto sources = (at::randn({count, 8, 16}, rng, doubles) / 2 * 8)
                         .round().clamp(-16, 16) / 8;
      auto selected = at::randint(8, {count}, rng, integers);
      sources.index_put_({Slice(), Slice(), 0}, -4);
      sources.index_put_({at::arange(count, integers), selected, 0}, 4);
      auto scores = sources.select(2, 0) /
                    ((sources.square().mean(-1) + 1e-6) * (1 + 1e-6)).sqrt();
      answer = (scores.softmax(-1).unsqueeze(-1) * sources).sum(1);
      inputs = sources.flatten(1);
    } else {
      const int64_t branches = name == "conditional-amplitudes" ? 4 : 1;
      auto signs = (2 * at::randint(2, {count, branches + 1, 32}, rng, integers) - 1)
                       .to(at::kDouble);
      auto identity = signs.select(1, 0);
      auto residuals = signs.narrow(1, 1, branches);
      auto coefficients = at::tensor({0.25, 0.5, 0.75, 0.25}, doubles)
                              .narrow(0, 0, branches).expand({count, branches}).clone();
      if (name == "conditional-amplitudes")
        coefficients.index_put_({identity.select(1, 0) > 0},
                               at::tensor({0.75, 0.25, 0.5, 0.75}, doubles));
      if (name == "linear-transport") {
        answer = identity / 4;
        inputs = identity;
      } else {
        answer = identity + (coefficients.unsqueeze(-1) * residuals /
                             std::sqrt(1 + std::ldexp(1.0, -23))).sum(1);
        inputs = signs.flatten(1);
      }
    }
    auto tokens = (inputs / step).round().to(at::kLong) + offset + 36;
    auto labels = (answer / step).round().to(at::kLong) + offset + 36;
    if (tokens.min().item<int64_t>() < 36 || tokens.max().item<int64_t>() > 36 + maximum ||
        labels.min().item<int64_t>() < 36 || labels.max().item<int64_t>() > 36 + maximum)
      throw std::runtime_error("source numeric grid exceeded");
    output(tokens, labels, answer, 37 + maximum);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
