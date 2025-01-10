#ifndef DWAVE
#define DWAVE

#include <string>
#include <vector>
#include <curl/curl.h>
#include "external/nlohmann_json.hpp"
#include "external/base64/base64.h"

namespace dwave {

    /**************************************************
    * DWAVE API CONSTANTS                             *
    **************************************************/

    const std::string SAPI_HOME_EU = "https://eu-central-1.cloud.dwavesys.com/sapi/v2/";
    const std::string SOLVER_ACCESS = "solvers/remote/";
    const std::string PROBLEM_ACCESS = "problems/";
    const std::string ANSWER_ACCESS = "answer/"; 

    /**************************************************
    * DWAVE STRUCT                                    *
    **************************************************/

    struct Solver {
        std::string id;
        std::string status;
        struct Properties {
            int num_qubits{};
            std::vector<int> qubits;
            std::vector<std::vector<int>> couplers;
            std::string category;
        } properties;
        double avg_load{};
    };

    struct ProblemRequest {
        std::string solver_id;
        std::string label;
        struct QPProblem {
            std::string format;
            std::vector<double> linear;
            std::vector<double> quadratic;
        } data;
        std::string type;
        struct SolverParams {
            int num_reads;
        } params;
    };

    struct ProblemSubmission {
        std::string status;
        std::string problem_id;
        std::string solver_id;
        std::string type;
        std::string submitted_on;
        std::string label;
    };

    struct ProblemAnswer {
        struct Answer {
            std::vector<bool> solutions; // TODO check, this type may be std::vector<std::vector<bool>>
            double energies{}; // TODO check, this type may be std::vector<double>
            struct QPUTiming {
                double qpu_access_time;
            } timing{};
            std::string format;
        } answer;
    };

    /**************************************************
    * UTILITY METHODS                                 *
    **************************************************/

    inline size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
        static_cast<std::string *>(userp)->append(static_cast<char *>(contents), size * nmemb);
        return size * nmemb;
    }

    inline std::vector<unsigned char> serialize_to_binary(const std::vector<double>& data) {
        std::vector<unsigned char> binaryData(data.size() * sizeof(double));
        std::memcpy(binaryData.data(), data.data(), binaryData.size());
        return binaryData;
    }

    inline std::string encode_base64(const std::vector<double>& input) {
        try {
            std::vector<unsigned char> binaryData = serialize_to_binary(input);
            return base64_encode(binaryData.data(), binaryData.size());
        } catch (const std::exception& e) {
            throw std::runtime_error("Unable to encode data: " + std::string(e.what()));
        }
    }

    // TODO check, this type may be std::vector<std::vector<bool>>
    inline std::vector<bool> decodeBase64Solutions(const std::string& encoded) {
        std::string decoded = base64_decode(encoded);
        std::vector<bool> solution;

        for (unsigned char byte : decoded) {
            for (int i = 0; i < 8; ++i) {
                bool bit = (byte >> i) & 1;
                solution.push_back(bit);
            }
        }
        return solution;
    }

    // TODO check, this type may be std::vector<double>
    inline double decodeBase64Energies(const std::string& encoded) {
        std::string decoded = base64_decode(encoded);
        if (decoded.size() != sizeof(double)) {
            throw std::runtime_error("Invalid decoded size for energy value");
        }

        double energy;
        std::memcpy(&energy, decoded.data(), sizeof(double));
        return energy;
    }

    inline std::string get_request(CURL* curl, const std::string& url) {
        std::string response_string;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
        CURLcode res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            throw std::runtime_error(std::string("Error in HTTP request: ") + curl_easy_strerror(res));
        }

        long http_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code != 200) {
            throw std::runtime_error("Error response code: " + std::to_string(http_code));
        }

        return response_string;
    }

    inline std::string post_request(CURL* curl, const std::string& url, const std::string& json) {
        std::string response_string;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
        CURLcode res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            throw std::runtime_error(std::string("Error in HTTP request: ") + curl_easy_strerror(res));
        }

        long http_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code != 201) {
            throw std::runtime_error("Error response code: " + std::to_string(http_code));
        }

        return response_string;
    }

    /**************************************************
    * DWAVE SAPI METHODS                              *
    **************************************************/

    inline std::vector<Solver> get_solvers(CURL* curl) {
        std::string response_string = get_request(curl, SAPI_HOME_EU + SOLVER_ACCESS);
        nlohmann::json j = nlohmann::json::parse(response_string);

        std::vector<Solver> solvers;
        for (auto item : j) {
            Solver solver;
            solver.id = item["id"];
            solver.status = item["status"];
            solver.properties.num_qubits = item["properties"]["num_qubits"];
            solver.properties.qubits = item["properties"]["qubits"].get<std::vector<int>>();
            solver.properties.couplers = item["properties"]["couplers"];
            solver.properties.category = item["properties"]["category"];
            solver.avg_load = item["avg_load"];
            solvers.push_back(solver);
        }
        
        return solvers;
    }

    inline Solver get_solver(CURL* curl, const std::string& solver_id) {
        std::vector<Solver> solvers = get_solvers(curl);
        for (auto solver : solvers) {
            if (solver.id == solver_id) return solver;
        }
        throw std::runtime_error("Error: no solver with id " + solver_id);
    }

    inline ProblemSubmission submit_problem(CURL* curl, const ProblemRequest& problem) {
        nlohmann::json problemData;
        problemData["solver"] = problem.solver_id;
        problemData["label"] = problem.label;
        problemData["data"]["format"] = problem.data.format;
        problemData["data"]["lin"] = encode_base64(problem.data.linear);
        problemData["data"]["quad"] = encode_base64(problem.data.quadratic);
        problemData["type"] = problem.type;
        problemData["params"]["num_reads"] = problem.params.num_reads;

        std::string response_string = post_request(curl, SAPI_HOME_EU + PROBLEM_ACCESS, problemData.dump());
        nlohmann::json j = nlohmann::json::parse(response_string);
        ProblemSubmission response;
        response.status = j["status"];
        response.problem_id = j["id"];
        response.solver_id = j["solver"];
        response.type = j["type"];
        response.submitted_on = j["submitted_on"];
        response.label = j["label"];

        return response;
    }

    inline ProblemAnswer get_problem_answer(CURL* curl, const std::string& problemID) {
        std::string response_string = get_request(curl, SAPI_HOME_EU + PROBLEM_ACCESS + problemID + "/" + ANSWER_ACCESS);

        nlohmann::json j = nlohmann::json::parse(response_string);
        ProblemAnswer problem_answer;
        problem_answer.answer.solutions = decodeBase64Solutions(j["answer"]["solutions"]);
        problem_answer.answer.energies = decodeBase64Energies(j["answer"]["energies"]);
        problem_answer.answer.timing.qpu_access_time = j["answer"]["timing"]["qpu_access_time"];
        problem_answer.answer.format = j["answer"]["format"];

        return problem_answer;
    }

}

#endif // DWAVE