// Independent native reconstruction of DiffGPT 5a37c515 mqar_div97.queries.
// Use NumPy's original PCG64 and its distribution routines for every draw.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

extern "C" {
#include "pcg64.h"
}
#include "numpy/random/bitgen.h"
extern "C" uint64_t random_interval(bitgen_t *, uint64_t);
extern "C" double random_standard_uniform(bitgen_t *);
extern "C" void random_bounded_uint64_fill(bitgen_t *, uint64_t, uint64_t, intptr_t, bool, uint64_t *);

using U32 = uint32_t;
using U64 = uint64_t;
constexpr int modulus = 97;
constexpr int count = 4656;

std::array<U64,4> seed_words(std::array<U32,4> input) {
    U32 hash = 0x43b0d7e5U;
    auto seed_hash = [&](U32 v) {
        v ^= hash;
        hash *= 0x931e8875U;
        v *= hash;
        return v ^ (v >> 16);
    };
    std::array<U32,4> pool{};
    std::transform(input.begin(), input.end(), pool.begin(), seed_hash);
    for (int src=0; src<4; ++src) for (int dst=0; dst<4; ++dst) {
        if (src == dst) continue;
        U32 v = 0xca01f9ddU * pool[dst] - 0x4973f715U * seed_hash(pool[src]);
        pool[dst] = v ^ (v >> 16);
    }
    hash = 0x8b51f9ddU;
    std::array<U64,4> state{};
    for (int i=0; i<8; ++i) {
        U32 v = pool[i % 4] ^ hash;
        hash *= 0x58f38dedU;
        v *= hash;
        state[i/2] |= U64(v ^ (v >> 16)) << (32*(i%2));
    }
    return state;
}

struct Numpy {
    pcg64_random_t core{};
    pcg64_state wrapper{&core,0,0};
    bitgen_t rng{};
    Numpy(U32 low, U32 high) {
        auto seed = seed_words({low,high,0,0});
        pcg64_set_seed(&wrapper, seed.data(), seed.data()+2);
        rng.state = &wrapper;
        rng.next_uint64 = [](void* s) { return pcg64_next64(static_cast<pcg64_state*>(s)); };
        rng.next_uint32 = [](void* s) { return pcg64_next32(static_cast<pcg64_state*>(s)); };
        rng.next_double = [](void* s) {
            return (pcg64_next64(static_cast<pcg64_state*>(s)) >> 11) * 0x1p-53;
        };
        rng.next_raw = rng.next_uint64;
    }
    std::vector<U64> integers(int high, size_t size) {
        std::vector<U64> values(size);
        random_bounded_uint64_fill(&rng, 0, high-1, static_cast<intptr_t>(size), false, values.data());
        return values;
    }
    void permute(std::vector<int>& row) {
        for (size_t i=row.size(); i>1; --i)
            std::swap(row[i-1], row[random_interval(&rng, i-1)]);
    }
};

std::vector<std::array<int,3>> equations(bool random_labels) {
    std::vector<std::array<int,3>> table;
    for (int a=0; a<modulus; ++a) for (int b=1; b<modulus; ++b) {
        int q=0;
        while ((q*b)%modulus != a) ++q;
        table.push_back({a,b,q});
    }
    U64 state = 1 ^ 0x73706c6974730002ULL;
    for (U64 bound=table.size(); bound>1; --bound) {
        U64 bits;
        do {
            state += 0x9e3779b97f4a7c15ULL;
            bits = state;
            bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
            bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebULL;
            bits ^= bits >> 31;
        } while (bits < -bound % bound);
        std::swap(table[bound-1],table[bits % bound]);
    }
    if (random_labels) {
        U64 entropy = 1 ^ 0x72616e646f6dULL;
        Numpy rng(static_cast<U32>(entropy), static_cast<U32>(entropy>>32));
        auto y = rng.integers(modulus,table.size());
        for (size_t i=0; i<table.size(); ++i) table[i][2] = y[i];
    }
    return table;
}

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    int loads = std::atoi(argv[1]), extra = std::atoi(argv[2]);
    bool is_test = std::atoi(argv[3]), random = std::atoi(argv[4]);
    const int writes = loads+extra;
    const auto table = equations(random);
    Numpy source(1,is_test?2:0);
    using Row = std::vector<int>;
    std::vector<Row> keys(count,Row(32)), order(count,Row(writes)), last(count,Row(loads));
    for (auto& row : keys) {
        std::iota(row.begin(),row.end(),1);
        source.permute(row);
        row.resize(loads);
    }
    auto again = source.integers(loads,count*extra);
    for (int i=0; i<count; ++i) {
        auto &row = order[i];
        std::iota(row.begin(),row.begin()+loads,0);
        std::copy_n(again.begin()+i*extra,extra,row.begin()+loads);
        source.permute(row);
        for (int t=0; t<writes; ++t) last[i][row[t]]=t;
    }
    std::vector<std::array<int,2>> operands(count);
    for (auto& pair : operands) {
        std::vector<double> score(loads);
        for (auto& v : score) v=random_standard_uniform(&source.rng);
        Row slot(loads);
        std::iota(slot.begin(),slot.end(),0);
        std::stable_sort(slot.begin(),slot.end(),[&](int a,int b) { return score[a]>score[b]; });
        pair = {slot[0],slot[1]};
    }
    auto finals = source.integers(modulus,count*loads);
    for (int i=0; i<count; ++i) for (int j=0; j<2; ++j)
        finals[i*loads+operands[i][j]]=table[is_test*count+i][j];
    std::vector<Row> values(count,Row(writes));
    for (int t=writes-1; t>=0; --t) {
        auto draws = source.integers(modulus-1,count);
        for (int i=0; i<count; ++i) {
            const int k = order[i][t];
            auto& following = finals[i*loads+k];
            if (last[i][k] != t) following = draws[i]+(draws[i]>=following);
            values[i][t]=following;
        }
    }
    // Write levels to an independent encoder. The stored label occupies the
    // last field, which the benchmark leaves out of its predictor's input.
    for (int i=0; i<count; ++i) {
        std::vector<int16_t> row;
        for (int t=0; t<writes; ++t) {
            row.push_back(-keys[i][order[i][t]]);
            row.push_back(-96);
            row.push_back(values[i][t]);
        }
        for (int k: operands[i]) row.push_back(-keys[i][k]);
        row.push_back(table[is_test*count+i][2]);
        if (fwrite(row.data(),sizeof(int16_t),row.size(),stdout)!=row.size()) return 3;
    }
}
