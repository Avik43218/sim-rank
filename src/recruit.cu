#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cctype>
#include <algorithm>
#include <unordered_set>
#include <filesystem>

using namespace std;
namespace fs = filesystem;

struct DataCorpus {
    uint32_t *d_all_hashes = nullptr;
    int *d_offsets = nullptr;
    int n_resumes = 0;
    vector<string> applicant_ids;
};

__device__ __host__ inline uint32_t splitmix32(uint64_t x) {

    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;

    return static_cast<uint32_t>((x ^ (x >> 31)) & 0xFFFFFFFF);
}

static vector<string> tokenize(const string &text) {

    vector<string> tokens;
    string word;

    for (char c : text) {
        if (isalpha(static_cast<unsigned char>(c))) {
            word += static_cast<unsigned char>(c);
        }
        else if (!word.empty()) {
            tokens.push_back(word);
            word.clear();
        }
    }
    if (!word.empty()) tokens.push_back(word);

    return tokens;
}

static string read_file(const fs::path &path) {

    ifstream file(path);
    if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + path.string());
    }

    ostringstream buf;
    buf << file.rdbuf();

    return buf.str();
}

__global__ void compare_resumes(const uint32_t *all_hashes, const int *offsets,
                                int n_resumes, const uint32_t *target,
                                int target_len, double *scores)
    {
        extern __shared__ uint32_t s_target[];

        for (int i = threadIdx.x; i < target_len; i += blockDim.x) {
            s_target[i] = target[i];
        }
        __syncthreads();

        int r_id = blockIdx.x;
        if (r_id > n_resumes) return;

        int start = offsets[r_id];
        int end = offsets[r_id + 1];
        int len = end - start;

        __shared__ int match_count;
        if (threadIdx.x == 0) match_count = 0;
        __syncthreads();

        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            uint32_t token = all_hashes[start + i];
            for (int j = 0; j < target_len; j++) {
                if (token == s_target[j]) {
                    atomicAdd(&match_count, 1);
                    break;
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            double denom = sqrt((double)len * (double)target_len);
            scores[r_id] = (denom > 0.0) ? ((double)match_count / denom) : 0.0;
        }
    }

DataCorpus preprocess_corpus(const string &folder_path) {

    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
        throw runtime_error("Invalid folder path!");
    }

    vector<string> file_paths;
    for (const auto &entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            file_paths.push_back(entry.path().string());
        }
    }

    if (file_paths.empty()) {
        throw runtime_error("No .txt files found in the folder!");
    }

    vector<uint32_t> all_hashes;
    vector<int> offsets = {0};
    vector<string> applicant_ids;

    for (const auto &file : file_paths) {
        string text = read_file(file);
        auto tokens = tokenize(text);

        vector<uint32_t> hashes;

        hashes.reserve(tokens.size());
        for (auto &t : tokens) {
            uint64_t seed = 0;
            unit8_t index = 0;
            for (char c : t) {
                seed = (uint8_t)(seed + ((uint8_t)c * index));
                index++;
            }
            hashes.push_back(splitmix32(seed));
        }

        all_hashes.insert(all_hashes.end(), hashes.begin(), hashes.end());
        offsets.push_back(all_hashes.size());
        applicant_ids.push_back(fs::path(file).stem().string().substr(8));
    }

    DataCorpus corpus;
    corpus.n_resumes = file_paths.size();
    corpus.applicant_ids = applicant_ids;

    cudaMalloc(&corpus.d_all_hashes, all_hashes.size() * sizeof(uint32_t));
    cudaMalloc(&corpus.d_offsets, offsets.size() * sizeof(int));
    cudaMemcpy(corpus.d_all_hashes, all_hashes.data(), all_hashes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(corpus.d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    cout << "Loaded " << corpus.n_resumes << " into GPU memory!\n";
    return corpus;
}

void compare_all_resumes(const DataCorpus &corpus, const string &target_path) {

    ifstream target_file(target_path, ios::binary);
    if (!target_file.is_open()) {
        throw runtime_error("Failed to open the target file!");
    }

    vector<uint32_t> h_target;
    target_file.seekg(0, ios::end);
    size_t size = target_file.tellg();
    target_file.seekg(0, ios::beg);
    h_target.resize(size / sizeof(uint32_t));
    target_file.read(reinterpret_cast<char*>(h_target.data()), size);
    target_file.close();

    uint32_t *d_target;
    double *d_scores;
    cudaMalloc(&d_target, h_target.size() * sizeof(uint32_t));
    cudaMalloc(&d_scores, corpus.n_resumes * sizeof(double));
    cudaMemcpy(d_target, h_target.data(), h_target.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int threads = 512;
    int shared_mem = h_target.size() * sizeof(uint32_t);
    compare_resumes<<<corpus.n_resumes, threads, shared_mem>>>(
        corpus.d_all_hashes, corpus.d_offsets, corpus.n_resumes,
        d_target, h_target.size(), d_scores);
    cudaDeviceSynchronize();

    vector<double> h_scores(corpus.n_resumes);
    cudaMemcpy(h_scores.data(), d_scores, corpus.n_resumes * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_target);
    cudaFree(d_scores);

    vector<pair<string, double>> results;
    for (int i = 0; i < corpus.n_resumes; ++i) {
        results.emplace_back(corpus.applicant_ids[i], h_scores[i]);
    }

    sort(results.begin(), results.end(), [](auto &a, auto &b) { return a.second > b.second; });

    ofstream out("../scores.csv");
    out << "applicant,similarity\n";
    for (auto &r : results)
        out << r.first << "," << r.second * 10 << "\n";
    out.close();

    cout << "Scores written to scores.csv\n";
}

int main(int argc, char **argv) {
    if (argc < 3) {
        cerr << "Usage: ./recruit <resumes_folder> <target_resume.bin>\n";
        return 1;
    }

    try {
        cout << "Step 1: Preprocessing resumes...\n";
        DataCorpus corpus = preprocess_corpus(argv[1]);

        cout << "Step 2: Comparing with target resume...\n";
        compare_all_resumes(corpus, argv[2]);
    } 
    catch (const exception &e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
