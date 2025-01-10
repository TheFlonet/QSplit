#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include "dwave/dwave.hpp"
#include <me/find_embedding/find_embedding.hpp>

template <typename S>
std::ostream& operator<<(std::ostream& os, const std::vector<S>& vector) {
    os << "{ ";
    for (const auto& i : vector)
        os << i << " ";
    os << "}";
    return os;
}

dwave::ProblemRequest get_dummy_problem() {
    std::unordered_map<std::string, double> biases = {{"x", 0.25}, {"y", 0.5}, {"xy", -1.0}};

    std::vector<double> lin_vec(5614, NAN);
    lin_vec[60] = biases["x"];
    lin_vec[61] = biases["y"];
    std::vector<double> quad_vec = {biases["xy"]};
    std::string SOLVER_ID = "Advantage_system5.4";

    dwave::ProblemRequest req;
    req.solver_id = SOLVER_ID;
    req.label = "2025 jan 8 - 1";
    req.data.format = "qp";
    req.data.linear = lin_vec;
    req.data.quadratic = quad_vec;
    req.type = "qubo";
    req.params.num_reads = 10;

    return req;
}

class MyCppInteractions : public find_embedding::LocalInteraction {
  public:
    bool _canceled = false;
    void cancel() { _canceled = true; }

  private:
    void displayOutputImpl(int, const std::string& mess) const override { std::cout << mess << std::endl; }
    void displayErrorImpl(int, const std::string& mess) const override { std::cerr << mess << std::endl; }
    [[nodiscard]] bool cancelledImpl() const override { return _canceled; }
};

void load_env_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: unable to load .env file");
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        size_t delimiter_pos = line.find('=');
        if (delimiter_pos == std::string::npos) continue;

        std::string key = line.substr(0, delimiter_pos);
        std::string value = line.substr(delimiter_pos + 1);

        setenv(key.c_str(), value.c_str(), 1);
    }
}

int main() {
    load_env_file(".env");
    const std::string DWAVE_API_TOKEN = std::getenv("DWAVE_API_TOKEN");
    const std::string SOLVER_ID = std::getenv("SOLVER_ID");

    curl_global_init(CURL_GLOBAL_DEFAULT);
    CURL* curl;
    curl = curl_easy_init();

    if (!curl) {
        curl_global_cleanup();
        throw std::runtime_error("Error during curl initialization");
    }

    curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("X-Auth-Token: " + DWAVE_API_TOKEN).c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    dwave::Solver qpu_solver = dwave::get_solver(curl, SOLVER_ID);
    std::vector<int> aside;
    std::vector<int> bside;
    for (auto couple : qpu_solver.properties.couplers) {
        aside.push_back(couple[0]);
        bside.push_back(couple[1]);
    }
    graph::input_graph qpu_graph(qpu_solver.properties.num_qubits, aside, bside);
    graph::input_graph input(3, {0, 1, 2}, {1, 2, 0});
    find_embedding::optional_parameters params;
    params.localInteractionPtr = std::make_shared<MyCppInteractions>();
    std::vector<std::vector<int>> chains;
    
    if (!findEmbedding(input, qpu_graph, params, chains)) {
        std::cerr << "Error: unable to find the embedding" << std::endl;
        return -1;
    }

    std::cout << chains << std::endl;

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);
    curl_global_cleanup();

    return 0;
}

/**
 * CURL
 * 
    curl_global_init(CURL_GLOBAL_DEFAULT);
    CURL* curl;
    curl = curl_easy_init();

    if (!curl) {
        curl_global_cleanup();
        throw new std::runtime_error("Error during curl initialization");
    }

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("X-Auth-Token: " + DWAVE_API_TOKEN).c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    ... CODE ...

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);
    curl_global_cleanup();
 */

/** 
 * SAPI
 *
    dwave::Solver a = dwave::get_solver(curl, SOLVER_ID);
    std::cout << a.id << " " << a.properties.num_qubits << std::endl;
    std::cout << a.properties.couplers[0] << std::endl;
    std::cout << a.properties.qubits[0] << std::endl;

    dwave::ProblemAnswer answer = dwave::get_problem_answer(curl, "647a6803-7a86-49d1-906a-753f6ed74587");
    std::cout << answer.answer.solutions << " # " << answer.answer.energies << " # " << answer.answer.timing.qpu_access_time << std::endl;

    dwave::ProblemRequest req = get_dummy_problem();
    std::cout << req.solver_id << std::endl << req.label << std::endl;
    std::cout << req.data.format << std::endl << req.data.linear << std::endl << req.data.quadratic << std::endl;
    std::cout << req.type << std::endl << req.params.num_reads << std::endl;
    
    dwave::ProblemSubmission sub = dwave::submit_problem(curl, req);
    std::cout << sub.status << std::endl << sub.problem_id << std::endl << sub.solver_id << std::endl;
    std::cout << sub.type << std::endl << sub.submitted_on << std::endl << sub.label << std::endl;
 */

/**
 * ME
 * 
    graph::input_graph input(3, {0, 1, 2}, {1, 2, 0});
    graph::input_graph qpu(4, {0, 1, 2, 3}, {1, 2, 3, 0});
    find_embedding::optional_parameters params;
    params.localInteractionPtr.reset(new MyCppInteractions());
    std::vector<std::vector<int>> chains;
    
    if (!find_embedding::findEmbedding(input, qpu, params, chains)) {
        std::cerr << "Error: unable to find the emebdding" << std::endl;
        return -1;
    }

    // chains[i] -> nodes of the qpu to represent input.nodes[i]

    std::cout << chains << std::endl;
 */