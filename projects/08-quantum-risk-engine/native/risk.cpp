#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// C ABI: inputs borrowed; all results returned through two scalar outputs.
// Catch exceptions at the ABI boundary; no C++ exception crosses into Python.
extern "C" int empirical_tail_risk(const double* losses, std::size_t n,
                                   double alpha, double* var, double* es) noexcept {
    if (!losses || !var || !es || n == 0 || !std::isfinite(alpha) || alpha <= 0 || alpha >= 1) return 1;
    try {
        std::vector<double> sorted(losses, losses+n);
        for (double loss : sorted) if (!std::isfinite(loss)) return 1;
        std::sort(sorted.begin(), sorted.end());
        auto index = static_cast<std::size_t>(std::ceil(alpha * static_cast<double>(n)))-1;
        index = std::min(index, n-1);
        *var = sorted[index];
        long double excess = 0;
        for (double loss : sorted) excess += std::max(0.0, loss-*var);
        *es = *var + static_cast<double>(excess / (static_cast<long double>(n) * (1-alpha)));
        return 0;
    } catch (...) {
        return 2;
    }
}
