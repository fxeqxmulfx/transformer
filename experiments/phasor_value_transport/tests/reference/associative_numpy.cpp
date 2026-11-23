// DiffGPT 5a37c515 single-compute equation frames.
#include "associative_numpy.hpp"

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
