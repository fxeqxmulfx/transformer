// Independent ATen reconstruction of DiffGPT 5a37c515 probes.text and TensorRows.
#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <bit>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3 || std::endian::native != std::endian::little) return 2;
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) return 2;
    const std::string text{std::istreambuf_iterator<char>(input), {}};
    std::string alphabet = text;
    std::sort(alphabet.begin(), alphabet.end());
    alphabet.erase(std::unique(alphabet.begin(), alphabet.end()), alphabet.end());
    std::vector<int64_t> ids;
    ids.reserve(text.size());
    for (char c : text) ids.push_back(alphabet.find(c));

    at::set_num_threads(2);
    const auto options = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
    const auto all = at::tensor(ids, options);
    const auto first = static_cast<int64_t>(all.numel() * 0.8);
    const auto second = static_cast<int64_t>(all.numel() * 0.9);
    struct Segment { int64_t begin, end, count, seed; };
    const std::map<std::string, Segment> segments{
        {"train", {0, first, 50000, 1}},
        {"validation", {first, second, 1024, 1001}},
        {"test", {second, all.numel(), 1024, 2001}},
    };
    const auto found = segments.find(argv[2]);
    if (found == segments.end()) return 2;
    const auto [begin, end, count, seed] = found->second;
    const auto segment = all.slice(0, begin, end);
    auto rng = at::make_generator<at::CPUGeneratorImpl>(seed);
    const auto starts = at::randperm(segment.numel() - 128, rng, options).narrow(0, 0, count);
    const auto windows = segment.unfold(0, 129, 1).index_select(0, starts);
    const auto x = windows.narrow(1, 0, 128);
    const auto y = windows.narrow(1, 1, 128);
    const auto rows = at::stack({x, y, at::zeros_like(x)}, 1).contiguous();
    std::cout << "torch.int64:(" << count << ", 3, 128):";
    std::cout.write(static_cast<const char*>(rows.const_data_ptr()), rows.nbytes());
    return std::cout ? 0 : 1;
}
