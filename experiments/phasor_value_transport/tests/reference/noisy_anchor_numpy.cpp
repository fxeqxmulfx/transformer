// Reproduce the six DiffGPT series_data.corrupt_answers draws with NumPy 2.5.4.
// The selected indices and replacement residues are emitted in little-endian order.
#include "associative_numpy.hpp"
#include <bit>
#include <set>

extern "C" uint64_t random_bounded_uint64(bitgen_t *, uint64_t, uint64_t, uint64_t, bool);

int main(int argc, char **argv) {
    static_assert(std::endian::native == std::endian::little);
    if (argc != 3) return 1;
    const int population = std::atoi(argv[1]);
    const int size = std::atoi(argv[2]);
    if (population <= 0 || population > 10000 || size < 0 || size > population) return 1;
    Numpy source(1 ^ 0xc077, 0);
    auto draw = [&](int maximum) {
        return static_cast<int>(random_bounded_uint64(&source.rng, 0, maximum, 0, false));
    };
    // Generator.choice with replace=False, shuffle=True and population <= 10,000.
    std::set<int> seen;
    std::vector<int> indices;
    for (int j = population - size; j < population; ++j) {
        int value = draw(j);
        if (!seen.insert(value).second) {
            value = j;
            seen.insert(value);
        }
        indices.push_back(value);
    }
    for (int i = size - 1; i > 0; --i) std::swap(indices[i], indices[draw(i)]);
    std::sort(indices.begin(), indices.end());
    auto replacements = source.integers(97, size);
    for (int i = 0; i < size; ++i) {
        std::array<uint64_t, 2> pair{static_cast<uint64_t>(indices[i]), replacements[i]};
        if (std::fwrite(pair.data(), sizeof(uint64_t), pair.size(), stdout) != pair.size()) return 1;
    }
}
