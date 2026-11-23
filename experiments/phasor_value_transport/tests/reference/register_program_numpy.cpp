// Independent replay of DiffGPT 5a37c515 mqar_div97._program_attempt and programs.
#include "associative_numpy.hpp"
#include <set>

using Grid = std::array<std::array<int, modulus>, modulus>;
constexpr int loads = 8;
struct State {
    std::vector<int> keys, order;
    std::array<int,loads> held{};
    std::set<std::pair<int,int>> used;
    std::vector<int16_t> raw;
    bool valid = true;
};

std::vector<State> attempt(int size, int steps, const Grid& allowed,
                          const Grid& operation, Numpy& rng) {
    std::vector<State> rows(size);
    for (auto& row : rows) {
        row.keys.resize(32);
        std::iota(row.keys.begin(),row.keys.end(),1);
        rng.permute(row.keys);
        row.keys.resize(loads);
    }
    for (auto& row : rows) {
        row.order.resize(loads);
        std::iota(row.order.begin(),row.order.end(),0);
        rng.permute(row.order);
    }
    for (int t=0; t<loads; ++t) {
        const auto fresh = rng.integers(modulus-1, size);
        const auto changed = rng.integers(modulus-2, size);
        for (int i=0; i<size; ++i) {
            auto& row = rows[i];
            const int slot = row.order[t];
            const int old = row.held[slot];
            const int increment = 1 + changed[i];
            row.held[slot] = old == 0 ? int(1+fresh[i]) : increment+(increment>=old);
            row.raw.push_back(-row.keys[slot]);
            row.raw.push_back(-96);
            row.raw.push_back(row.held[slot]);
        }
    }
    for (int t=0; t<steps; ++t) {
        for (auto& row : rows) {
            int left=0,right=0;
            double best=-1;
            for (int i=0; i<loads; ++i) for (int j=0; j<loads; ++j) {
                const double draw=random_standard_uniform(&rng.rng);
                const int a=row.held[i], b=row.held[j];
                if (i!=j && allowed[a][b] && !row.used.contains({a,b}) && draw>best) {
                    left=i; right=j; best=draw;
                }
            }
            row.valid &= best>=0;
            const int a=row.held[left], b=row.held[right], y=operation[a][b];
            row.raw.push_back(-row.keys[left]);
            row.raw.push_back(-row.keys[right]);
            row.raw.push_back(y);
            row.used.insert({a,b});
            row.held[left]=y;
        }
    }
    return rows;
}

int main(int argc, char** argv) {
    if (argc!=3) return 2;
    const int steps=std::atoi(argv[1]), split=std::atoi(argv[2]);
    if ((steps!=2 && steps!=4) || (split!=0 && split!=1)) return 2;
    const auto shuffled=equations(false);
    Grid allowed{}, operation{};
    int pairs=0;
    for (int i=0;i<int(shuffled.size());++i) {
        const auto [a,b,y]=shuffled[i];
        operation[a][b]=y;
        if (i/count==split && a>0) {allowed[a][b]=1; ++pairs;}
    }
    int remaining=(pairs+steps-1)/steps;
    Numpy rng(1,split==0?0:2);
    while (remaining) {
        auto rows=attempt(remaining,steps,allowed,operation,rng);
        for (const auto& row: rows) {
            if (!row.valid) continue;
            if (fwrite(row.raw.data(),sizeof(int16_t),row.raw.size(),stdout)!=row.raw.size()) return 3;
            --remaining;
        }
    }
}
