// Multithreaded Monte Carlo kernel for the same contracts as the Python engine.
// Prints one JSON object so the Python harness can record price and throughput.
//
//   mc_pricer --kind european_call --spot 100 --rate 0.05 --vol 0.2 --strike 100 \
//             --maturity 1 --steps 1 --paths 2000000 --seed 1 [--barrier 130] [--threads N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct Params {
    std::string kind = "european_call";
    double spot = 100, rate = 0.05, vol = 0.2, strike = 100, maturity = 1, barrier = 0;
    int steps = 1;
    std::uint64_t paths = 1'000'000, seed = 1;
    unsigned threads = std::max(1u, std::thread::hardware_concurrency());
};

struct Sums {
    double sum = 0, sum_sq = 0;
};

static double payoff(const Params& p, double terminal, double running_sum, double running_max) {
    if (p.kind == "european_call") return std::max(terminal - p.strike, 0.0);
    if (p.kind == "european_put") return std::max(p.strike - terminal, 0.0);
    if (p.kind == "asian_call") return std::max(running_sum / p.steps - p.strike, 0.0);
    return running_max < p.barrier ? std::max(terminal - p.strike, 0.0) : 0.0;  // up-and-out call
}

static Sums simulate(const Params& p, std::uint64_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal;
    const double dt = p.maturity / p.steps;
    const double drift = (p.rate - 0.5 * p.vol * p.vol) * dt;
    const double diffusion = p.vol * std::sqrt(dt);
    const double discount = std::exp(-p.rate * p.maturity);
    Sums out;
    for (std::uint64_t i = 0; i < n; ++i) {
        double log_s = std::log(p.spot), running_sum = 0, running_max = 0, s = p.spot;
        for (int j = 0; j < p.steps; ++j) {
            log_s += drift + diffusion * normal(rng);
            s = std::exp(log_s);
            running_sum += s;
            running_max = std::max(running_max, s);
        }
        const double value = discount * payoff(p, s, running_sum, running_max);
        out.sum += value;
        out.sum_sq += value * value;
    }
    return out;
}

static Params parse(int argc, char** argv) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
        const std::string value = argv[++i];
        if (flag == "--kind") p.kind = value;
        else if (flag == "--spot") p.spot = std::stod(value);
        else if (flag == "--rate") p.rate = std::stod(value);
        else if (flag == "--vol") p.vol = std::stod(value);
        else if (flag == "--strike") p.strike = std::stod(value);
        else if (flag == "--maturity") p.maturity = std::stod(value);
        else if (flag == "--barrier") p.barrier = std::stod(value);
        else if (flag == "--steps") p.steps = std::stoi(value);
        else if (flag == "--paths") p.paths = std::stoull(value);
        else if (flag == "--seed") p.seed = std::stoull(value);
        else if (flag == "--threads") p.threads = static_cast<unsigned>(std::stoul(value));
        else throw std::invalid_argument("unknown flag " + flag);
    }
    const bool known = p.kind == "european_call" || p.kind == "european_put" || p.kind == "asian_call" ||
                       p.kind == "barrier_up_out_call";
    if (!known) throw std::invalid_argument("unknown kind " + p.kind);
    if (p.kind == "barrier_up_out_call" && p.barrier <= p.strike)
        throw std::invalid_argument("barrier_up_out_call needs --barrier above the strike");
    if (p.spot <= 0 || p.vol <= 0 || p.strike <= 0 || p.maturity <= 0 || p.steps < 1 || p.paths < 2 || p.threads < 1)
        throw std::invalid_argument("spot, vol, strike, maturity, steps, paths and threads must be positive");
    return p;
}

int main(int argc, char** argv) {
    Params p;
    try {
        p = parse(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
    const unsigned workers = static_cast<unsigned>(std::min<std::uint64_t>(p.threads, p.paths));
    std::vector<Sums> partial(workers);
    std::vector<std::thread> pool;
    std::seed_seq seq{p.seed};
    std::vector<std::uint32_t> seeds(2 * workers);
    seq.generate(seeds.begin(), seeds.end());

    const auto start = std::chrono::steady_clock::now();
    for (unsigned w = 0; w < workers; ++w) {
        const std::uint64_t n = p.paths / workers + (w < p.paths % workers ? 1 : 0);
        const std::uint64_t seed = (static_cast<std::uint64_t>(seeds[2 * w]) << 32) | seeds[2 * w + 1];
        pool.emplace_back([&, w, n, seed] { partial[w] = simulate(p, n, seed); });
    }
    for (auto& t : pool) t.join();
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    double sum = 0, sum_sq = 0;
    for (const auto& s : partial) {
        sum += s.sum;
        sum_sq += s.sum_sq;
    }
    const double n = static_cast<double>(p.paths);
    const double mean = sum / n;
    const double variance = std::max((sum_sq - n * mean * mean) / (n - 1), 0.0);
    std::cout.precision(12);
    std::cout << "{\"price\": " << mean << ", \"stderr\": " << std::sqrt(variance / n) << ", \"paths\": " << p.paths
              << ", \"threads\": " << workers << ", \"wall_time_s\": " << elapsed << ", \"paths_per_s\": "
              << n / elapsed << "}\n";
    return 0;
}
