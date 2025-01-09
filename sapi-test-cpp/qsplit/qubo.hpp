#ifndef QUBO
#define QUBO

#define XTENSOR_USE_XSIMD
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xframe/xvariable.hpp>
#include <cstdint>
#include "la_util.hpp"

namespace qubo {

    enum class QubitState : uint8_t {
        False = 0,
        True = 1,
        Error = 2
    };

    using sol_df_t = xf::xvariable<std::vector<QubitState>, double>;

class QUBOProblem {
private:
    xt::xarray<double> _matrix;
    double _offset;
    std::vector<int> _cols_idx;
    std::vector<int> _rows_idx;
    size_t _problem_size;
    sol_df_t _solution_df;
public:
    QUBOProblem(const xt::xarray<double>& matrix, double offset, const std::vector<int> cols_idx, 
                const std::vector<int> rows_idx, bool to_transform) {
        if (!la::is_square(matrix)) throw std::invalid_argument("Error: input matrix must be square");
        if (cols_idx.size() != rows_idx.size()) 
            throw std::invalid_argument("Error: cols_idx and rows_idx have different sizes");
        if (cols_idx.size() != matrix.shape()[0]) 
            throw std::invalid_argument("Error: matrix size must be equal to indexes size");

        xt::xarray<double> new_mat;
        if (matrix.shape()[0] % 2 != 0) {
            new_mat(matrix.shape()[0] + 1, matrix.shape()[1] + 1);
            new_mat.fill(0);
            xt::view(new_mat, xt::range(0, matrix.shape()[0]), xt::range(0, matrix.shape()[1])) = matrix;
        } else {
            new_mat = matrix;
        }

        if (!to_transform || la::is_upper(matrix))
            _matrix = matrix;
        else
            _matrix = la::lu(matrix);

        _offset = offset;
        _cols_idx = cols_idx;
        if (_cols_idx.size() % 2 != 0)
            _cols_idx.push_back(_cols_idx.size());
        _rows_idx = rows_idx;
        if (_rows_idx.size() % 2 != 0)
            _rows_idx.push_back(_rows_idx.size());
        _problem_size = _cols_idx.size();
    }

    /**************************************************
    * GETTER                                          *
    **************************************************/

    const xt::xarray<double>& get_matrix() const {
        return _matrix;
    }

    const double get_offset() const {
        return _offset;
    }

    const std::vector<int>& get_cols_idx() const {
        return _cols_idx;
    }

    const std::vector<int>& get_rows_idx() const {
        return _rows_idx;
    }

    const size_t get_problem_size() const {
        return _problem_size;
    }

    const sol_df_t& get_solution_df() const {
        return _solution_df;
    }

    /**************************************************
    * SETTER                                          *
    **************************************************/

    void set_solution_df(const sol_df_t& solution_df) {
        _solution_df = solution_df;
    }
};

}

#endif // QUBO