# Parallelization of Markov Chain Simulation using OpenMP

**Author:** Vinayak Sharma  
**Roll No:** 2023BCS0002  
**Course:** CSS 311-Parallel and Distributed Computing  
**Instructor:** Dr.John Paul Martin  
**Institute:** Indian Institute of Information Technology, Kottayam  
**Date:** 26th October 2025

---

## About the Project
This project explores how a basic Markov Chain Monte Carlo (MCMC) simulation can be parallelized using OpenMP.  
The main goal was to analyze how execution time changes with increasing thread count.
Secondly we implemented to understand the practical performance limits of CPU-based parallelism.

The project includes three implementations:
1. **Serial Version** – baseline, single-threaded approach  
2. **Parallel Version** – straightforward OpenMP parallelization  
3. **Improved Parallel Version** – attempted optimization (loop restructuring and load balancing tweaks)

All codes are written in C++ and tested on a local system with an Intel i5 processor.

---

## ⚙️ How to Reproduce the Work

### **1. Prerequisites**
Make sure you have:
- A C++ compiler with OpenMP support (e.g., `g++ >= 9.0`)
- A Linux or Windows environment with terminal/command prompt
- No external dependencies are required

---

### **2. Compilation**

Use the following commands to compile each version:

```bash
# Serial version
g++ serial.cpp -o serial

# Basic parallel version
g++ parallel.cpp -fopenmp -o parallel

# Improved parallel version
g++ improved_parallel.cpp -fopenmp -o improved_parallel
