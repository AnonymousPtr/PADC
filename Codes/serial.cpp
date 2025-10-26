// Author: Vinayak Sharma
// Roll No: 2023BCS0002
// Description: Estimates stationary distribution using Serial Monte Carlo simulation

// #include <bits/stdc++.h>
// Use the above only if you have gcc compiler and comment the rest libraries
#include <iostream>
#include <vector>
#include <utility>
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
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

    vector<pair<int,int>> test_cases = { {1000, 100}, {5000, 500}, {10000, 1000} };

    mt19937_64 rng(42);  //I kept the seed fixed good for reproductibility
    uniform_real_distribution<double> uni(0.0, 1.0);

    // precomputing cumulative probabilities
    vector<vector<double>> cumP(S, vector<double>(S, 0.0));
    for(int i = 0; i < S; ++i){
        cumP[i][0] = P[i][0];
        for(int j = 1; j < S; ++j)
            cumP[i][j] = cumP[i][j-1] + P[i][j];
    }

    cout << "Serial Monte Carlo Simulation Results:\n";

    for(int idx = 0; idx < test_cases.size(); ++idx){
        int N = test_cases[idx].first;
        int T = test_cases[idx].second;
        vector<double> visit_count(S, 0.0);

        auto start = chrono::high_resolution_clock::now();

        for(int i = 0; i < N; ++i){
            int state = rng() % S;
            for(int t = 0; t < T; ++t){
                double r = uni(rng);
                int next = 0;
                while(next < S-1 && cumP[state][next] < r) next++;
                state = next;
                visit_count[state]++;
            }
        }

        auto end = chrono::high_resolution_clock::now();
        double duration = chrono::duration<double>(end-start).count();
        double total_visits = accumulate(visit_count.begin(), visit_count.end(), 0.0);

        cout << "\nTest Case " << idx+1 << " (N=" << N << ", T=" << T << "):\n";
        for(int i = 0; i < S; ++i)
            cout << "pi[" << i << "] = " << visit_count[i]/total_visits << "\n";
        cout << "Execution Time: " << duration << " seconds\n";
    }

    return 0;
}