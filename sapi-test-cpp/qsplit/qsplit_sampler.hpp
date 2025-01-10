#ifndef QSPLIT
#define QSPLIT

#include <array>
#include <vector>
#include <cmath>
#include <xtensor/xview.hpp>
#include <xtensor/xindex_view.hpp> 
#include "qubo.hpp"

namespace qsplit {

enum SamplerKind {
    SimulatedAnnealing,
    QPU
};

class QSplitSampler {
public:
    QSplitSampler(SamplerKind sampler, int cut_dim) : _sampler(sampler), _cut_dim(cut_dim) {}

    void sample_qubo(qubo::QUBOProblem& qubo, double& energy) {
        if (qubo.get_problem_size() < _cut_dim || xt::count_nonzero(qubo.get_matrix())() <= _cut_dim*(_cut_dim+1)/2) {
            throw std::logic_error("Not implemented yet");
        }

        std::array<qubo::QUBOProblem, 3> subqubos = split_problem(qubo);
        double q_time = 0;
        for (auto sq : subqubos) {
            double tmp_q_time;
            sample_qubo(sq, tmp_q_time);
            q_time += tmp_q_time;
        }

        aggregate(qubo, subqubos, q_time);
    }
private:
    SamplerKind _sampler;
    int _cut_dim;

    /**
     * Returns 3 unsolved sub-problems in qubo form.
     * The 3 sub-problems correspond to the matrices obtained by dividing the qubo matrix of the original problem 
     * in half both horizontally and vertically.
     * The sub-problem for the sub-matrix in the bottom left corner is not given as this is always empty.
     * The order of the results is:
     *      - Upper left sub-matrix,
     *      - Upper right sub-matrix,
     *      - Lower right sub-matrix.
     * All sub-problems are converted to obtain an upper triangular matrix aka a new qubo problem.
     */
    std::array<qubo::QUBOProblem, 3> split_problem(const qubo::QUBOProblem& qubo) {
        size_t qubo_size = qubo.get_problem_size();
        size_t half = qubo_size / 2;

        std::vector<int> cols = qubo.get_cols_idx();
        std::vector<int> cols_first(cols.begin(), cols.begin() + half);
        std::vector<int> cols_last(cols.begin() + half, cols.end());

        std::vector<int> rows = qubo.get_rows_idx();
        std::vector<int> rows_first(rows.begin(), rows.begin() + half);
        std::vector<int> rows_last(rows.begin() + half, rows.end());

        return {
            qubo::QUBOProblem(
                xt::view(qubo.get_matrix(), xt::range(0, half), xt::range(0, half)),
                qubo.get_offset(), cols_first, rows_first, false
            ),
            qubo::QUBOProblem(
                xt::view(qubo.get_matrix(), xt::range(0, half), xt::range(half, qubo_size)),
                qubo.get_offset(), cols_last, rows_first, true
            ),
            qubo::QUBOProblem(
                xt::view(qubo.get_matrix(), xt::range(half, qubo_size), xt::range(half, qubo_size)),
                qubo.get_offset(), cols_last, rows_last, false
            )
        };
    }

    qubo::sol_df_t fill_with_error(const std::vector<int>& columns, const qubo::sol_df_t& ur_sol) {
        qubo::sol_df_t res;
        for (qubo::solved_assignment s : ur_sol) {
            qubo::solved_assignment tmp;
            tmp.second = s.second;
            for (size_t i : columns) {
                if (s.first.count(i) == 1) {
                    tmp.first[i] = s.first.at(i);
                } else {
                    tmp.first[i] = qubo::QubitState::Error;
                }
            }
            res.push_back(tmp);
        }
        return res;
    } 

    qubo::sol_df_t combine_ul_lr(const qubo::QUBOProblem& ul, const qubo::QUBOProblem& lr) {
        std::vector<int> all_index(ul.get_rows_idx().begin(), ul.get_rows_idx().end());
        all_index.insert(all_index.end(), lr.get_cols_idx().begin(), lr.get_cols_idx().end());
        
        qubo::sol_df_t res;
        for (size_t i = 0; i < ul.get_solution_df().size(); ++i) {
            qubo::solved_assignment tmp;

            for (auto const& [key, val] : ul.get_solution_df()[i].first) { tmp.first[key] = val; }
            for (auto const& [key, val] : lr.get_solution_df()[i].first) { tmp.first[key] = val; }
            tmp.second = ul.get_solution_df()[i].second + lr.get_solution_df()[i].second;
            res.push_back(tmp);
        }

        return fill_with_error(all_index, res);
    }

    qubo::sol_df_t get_closest_assignments(const qubo::sol_df_t& starting_sols, const qubo::sol_df_t& or_sol_filled) {
        throw std::logic_error("Not implemented yet");
    }

    qubo::sol_df_t combine_dataframe(const qubo::sol_df_t& starting_sols, const qubo::sol_df_t& ur_sol_filled) {
        qubo::sol_df_t res;
        const qubo::QubitState err = qubo::QubitState::Error;
        for (size_t i = 0; i < starting_sols.size(); ++i) {
            qubo::solved_assignment row;

            const qubo::solved_assignment start_assign = starting_sols[i];
            const qubo::solved_assignment ur_assign = ur_sol_filled[i];

            bool contain_error = false;
            for (auto const& [key, s_val] : start_assign.first) {
                qubo::QubitState ur_val = ur_assign.first.count(key) == 1 ? ur_assign.first.at(key) : err;
                if (ur_val == err && s_val != err) {
                    row.first[key] = s_val;
                } else if (s_val == err && ur_val != err) {
                    row.first[key] = ur_val;
                } else if (s_val == ur_val) {
                    row.first[key] = s_val;
                    if (s_val == err) contain_error = true;
                } else {
                    row.first[key] = err;
                    contain_error = true;
                }
            }

            if (contain_error || (isnan(start_assign.second) && isnan(ur_assign.second))) {
                row.second = nan("Invalid energy");
            } else if (isnan(start_assign.second)) {
                row.second = ur_assign.second;
            } else if (isnan(ur_assign.second)) {
                row.second = start_assign.second;
            } else {
                row.second = start_assign.second + ur_assign.second;
            }


            res.push_back(row);
        }

        return res;
    }

    double local_search(const qubo::sol_df_t& combined_df, qubo::QUBOProblem& qubo) {
        throw std::logic_error("Not implemented yet");
    }

    void aggregate(qubo::QUBOProblem& qubo, std::array<qubo::QUBOProblem, 3>& solved_subqubos, double& q_time) {
        // Aggregate upper-left qubo with lower-right
        qubo::sol_df_t starting_sols = combine_ul_lr(solved_subqubos[0], solved_subqubos[2]);
        // Set missing columns in upper-right qubo to qubo::QubitState.Error
        qubo::sol_df_t ur_sol_filled = fill_with_error(qubo.get_cols_idx(), solved_subqubos[1].get_solution_df());
        // Search the closest assignments between upper-right qubo and merged solution (UL and LR qubos)
        qubo::sol_df_t closest_df = get_closest_assignments(starting_sols, ur_sol_filled);
        // Combine
        qubo::sol_df_t combined_df = combine_dataframe(starting_sols, closest_df);
        // Conflicts resolution
        q_time += local_search(combined_df, qubo);
    }
};

}

#endif // QSPLIT