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
The goal was to observe how parallelism impacts performance and accuracy, and to compare execution time across different thread counts.

---

The project includes three implementations:
1. **Serial Version**: baseline, single-threaded approach  
2. **Parallel Version**: straightforward OpenMP parallelization  
3. **Improved Parallel Version**: attempted optimization (loop restructuring and load balancing tweaks)

All codes are written in C++ and tested on a local system with an AMD Ryzen 7 Processor.

---

## How to Reproduce the Work

### **1. Prerequisites**
To run the code, ensure that the following are installed:

- **GCC compiler** with OpenMP support (version 9.0 or above recommended)  
- **Linux / Windows / macOS** terminal  
- No external dependencies are required

---

### **2. Compilation**

- Copy the codes given in the Codes Folder. Then,  
use the following commands to compile each version:

```bash
# Serial version
g++ serial.cpp -o serial

# Basic parallel version
g++ parallel.cpp -fopenmp -o parallel

# Improved parallel version
g++ improved_parallel.cpp -fopenmp -o improved_parallel
```
### **3. Run The Simulation**

```bash
./improved_parallel # Similarly you can run other executable files after compiling
```

### **4. (optional) Modify the number of states or iterations in the code to test scalability.**

### **5. Compare The Resuts**  
- The terminal will display execution time for different configurations.
- You can record these.
- (optional) visualize them in Excel, Python or any plotting tool.

