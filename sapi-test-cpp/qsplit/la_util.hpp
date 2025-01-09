#ifndef LA_UTIL
#define LA_UTIL

#define XTENSOR_USE_XSIMD
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

namespace la {

xt::xarray<double> lu(const xt::xarray<double>& mat) {
    int n = mat.shape()[0];
    xt::xarray<double> up = mat;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double factor = up(j, i) / up(i, i);
            auto up_row_j = xt::view(up, j, xt::all());
            auto up_row_i = xt::view(up, i, xt::all());
            up_row_j -= factor * up_row_i; 
        }
    }

    return up;
}

bool is_square(const xt::xarray<double>& mat) {
    return mat.shape()[0] == mat.shape()[1];
}

bool is_upper(const xt::xarray<double>& mat) {
    for (size_t i = 1; i < mat.shape()[0]; ++i)
        for (size_t j = 0; j < i; ++j)
            if (mat(i, j) != 0) return false;
    return true;
}

}

#endif // LA_UTIL