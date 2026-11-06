# Secure-Adaptive-Reduction-Simulation

This repository contains the simulation framework for **Higher-Order Correlation Power Analysis (HOCPA)** on Masked Kyber, implementing a **Bounded Secure Adaptive Reduction** side-channel countermeasure. 

This project is containerized using Docker to ensure identical dependencies, performance metrics, and environment configurations across all machines.

---

## 🛠️ Prerequisites

Before running the simulation, make sure you have installed:
* **Git**: [Download Git](https://git-scm.com)
* **Docker Desktop**: [Download Docker Desktop](https://docker.com)

> 💡 **Windows Users**: Ensure Docker Desktop is open and the engine status bar in the bottom-left corner is **Green (Engine Running)** before running commands.

---

## 🚀 How to Run the Simulation

### 1. Clone the Repository
Open your terminal (Command Prompt or PowerShell on Windows) and run:
```bash
git clone https://github.com
cd Secure-Adaptive-Reduction-Simulation
```

### 2. Build the Docker Image
Build the isolated execution environment. This layer automatically installs all necessary dependencies like `numpy` and `matplotlib`:
```bash
docker build -t hocpa-kyber .
```

### 3. Run and Extract the Evaluation Graphs
Run the container using the command matching your operating system. This automatically syncs a local directory called `output` to save the results directly on your machine.

* **Windows (Command Prompt / `cmd`):**
  ```cmd
  docker run --rm -v "%cd%/output:/app/output" hocpa-kyber
  ```

* **Windows (PowerShell):**
  ```powershell
  docker run --rm -v "\${PWD}/output:/app/output" hocpa-kyber
  ```

* **Mac / Linux (Terminal):**
  ```bash
  docker run --rm -v "\$(pwd)/output:/app/output" hocpa-kyber
  ```

---

## 📊 Expected Outputs

Once the execution finishes, the container will stream 5 specific validation phases directly to your terminal:
1. **Attack Success Rate (ASR) Swings** (tracking depths from 500 to 5,000 trace samples)
2. **Peak Correlation (r_max)** measured at high-trace counts (N=20,000)
3. **Functional Correctness** percentages validating Kyber's mathematical compliance
4. **Computational Overhead** latency benchmarks
5. **Signal-to-Noise Ratio (SNR)** variance evaluations

### Checking Your Results Chart
After the console displays `EXPERIMENT COMPLETE`, check your local project folder. A new folder named **`output/`** will be generated containing **`results.png`**, which provides the final high-resolution comparison chart.
