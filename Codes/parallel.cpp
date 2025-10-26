// Author: Vinayak Sharma
// Roll No: 2023BCS0002
// Description: Parallelized Monte Carlo simulation using per-thread RNGs and local visit counts

// #include <bits/stdc++.h>
//Use the above only if you have gcc compiler and comment the rest libraries
#include <iostream>
#include <vector>
#include <utility>
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <omp.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << "Vinayak Sharma, 2023BCS0002" << endl;

    int S = 3;
    vector<vector<double>> P = {
        {0.1, 0.6, 0.3},
        {0.4, 0.4, 0.2},
        {0.5, 0.3, 0.2}
    };

    // Precompute cumulative probabilities
    vector<vector<double>> cumP(S, vector<double>(S, 0.0));
    for (int i = 0; i < S; ++i) {
        cumP[i][0] = P[i][0];
        for (int j = 1; j < S; ++j)
            cumP[i][j] = cumP[i][j - 1] + P[i][j];
    }

    vector<pair<int,int>> test_cases = { {1000,100}, {5000,500}, {10000,1000} };

    cout << "Parallel Monte Carlo Simulation Results:\n";

    for (int idx = 0; idx < test_cases.size(); ++idx) {
        int N = test_cases[idx].first;
        int T = test_cases[idx].second;
        vector<double> global_visit(S, 0.0);

        double start_time = omp_get_wtime();

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            mt19937_64 rng(chrono::high_resolution_clock::now().time_since_epoch().count() + tid * 1000);
            uniform_real_distribution<double> uni(0.0, 1.0);
            vector<double> local_visit(S, 0.0);

            #pragma omp for
            for (int i = 0; i < N; ++i) {
                int state = rng() % S;
                for (int t = 0; t < T; ++t) {
                    double r = uni(rng);
                    int next = 0;
                    while (next < S - 1 && cumP[state][next] < r) next++;
                    state = next;
                    local_visit[state]++;
                }
            }

            #pragma omp critical
            {
                for (int i = 0; i < S; ++i)
                    global_visit[i] += local_visit[i];
            }
        }

        double end_time = omp_get_wtime();
        double total_time = end_time - start_time;
        double total_visits = accumulate(global_visit.begin(), global_visit.end(), 0.0);

        cout << "\nTest Case " << idx+1 << " (N=" << N << ", T=" << T << "):\n";
        for (int i = 0; i < S; ++i)
            cout << "pi[" << i << "] = " << global_visit[i] / total_visits << "\n";
        cout << "Execution Time: " << total_time << " seconds\n";
        cout << "Threads Used: " << omp_get_max_threads() << "\n";
    }

    cout << "Executed by: Vinayak Sharma (2023BCS0002)\n";
    return 0;
}