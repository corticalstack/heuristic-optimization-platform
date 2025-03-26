# 🚀 Heuristic Optimization Platform (HOP)

A flexible and extensible framework for implementing, testing, and comparing various heuristic optimization algorithms on different problem domains.

## 📚 Description

The Heuristic Optimization Platform (HOP) is a Python-based framework designed to facilitate research and experimentation with different metaheuristic optimization algorithms. It provides a structured environment for:

- Implementing and testing various optimization algorithms
- Defining different problem domains
- Comparing algorithm performance across multiple metrics
- Visualizing optimization results
- Implementing hyper-heuristics that combine multiple optimization strategies

HOP supports both combinatorial optimization problems (like Flow Shop Scheduling) and continuous optimization problems (like Rastrigin function), with a modular architecture that makes it easy to extend with new algorithms and problem domains.

## ✨ Features

- **Multiple Optimization Algorithms**:
  - Simulated Annealing (SA) with various neighborhood operators
  - Genetic Algorithms (GA) with different crossover and mutation operators
  - Particle Swarm Optimization (PSO)
  - Differential Evolution (DE)
  - Evolution Strategy (ES)
  - Hyper-heuristics (HH) that can dynamically select from multiple algorithms

- **Problem Domains**:
  - Flow Shop Scheduling Problem (FSSP)
  - Rastrigin function (continuous optimization)
  - Extensible framework for adding new problems

- **Comprehensive Analysis**:
  - Statistical comparison of algorithm performance
  - Visualization of fitness trends
  - Gantt charts for scheduling problems
  - Detailed logging of optimization runs

- **Configurable Parameters**:
  - YAML-based configuration for problems and optimizers
  - Adjustable computational budget
  - Customizable algorithm parameters

## 🔧 Setup Guide

### Prerequisites

- Python 3.6+
- NumPy
- Pandas
- Matplotlib (for visualization)
- PyYAML (for configuration)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/heuristic-optimization-platform.git
   cd heuristic-optimization-platform
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib pyyaml
   ```

## 🔍 Usage

### Basic Usage

Run the platform with default settings:

```bash
python main.py
```

This will execute all enabled optimizers on all enabled problems as defined in the configuration files.

### Configuration

The platform uses YAML configuration files located in the `config/` directory:

- `general.yaml`: General settings like computational budget and runs per optimizer
- `problems.yaml`: Problem-specific settings and benchmarks
- `optimizers.yaml`: Optimizer-specific settings and parameters

Example of enabling a specific optimizer in `optimizers.yaml`:

```yaml
PSO:
  enabled: True
  description: Particle Swarm Optimization
  type: low
  optimizer: PSO
  generator_comb: continuous
  generator_cont: continuous
  initial_sample: False
  lb: 0
  ub: 4
```

Example of enabling a specific problem in `problems.yaml`:

```yaml
FSSP:
  enabled: True
  type: combinatorial
  description: Flow Shop Scheduling Problem
  benchmarks:
    taillard_20_5_i1.txt:
      enabled: True
  lb: 0
  ub: nmax
  inertia_coeff: 0.4
  local_coeff: 2.1
  global_coeff: 2.1
```

### Results

Results are stored in the `results/` directory, organized by timestamp. For each optimization run, the platform generates:

- CSV files with detailed statistics
- Fitness trend plots
- Gantt charts for scheduling problems
- Summary statistics comparing different optimizers

## 🏗️ Architecture

The platform is organized into several key components:

- **Main Controller** (`main.py`): Entry point that initializes and runs the platform
- **Heuristics Manager** (`heuristics_manager.py`): Manages the execution of optimization jobs
- **Problems** (`problems/`): Implementations of different optimization problems
- **Optimizers** (`optimizers/`): Implementations of different optimization algorithms
- **Utilities** (`utilities/`): Helper functions, logging, statistics, and visualization tools

### Adding a New Problem

To add a new problem:

1. Create a new class in the `problems/` directory that inherits from `Problem`
2. Implement the required methods: `evaluator()`, `pre_processing()`, and `post_processing()`
3. Add the problem configuration to `config/problems.yaml`

### Adding a New Optimizer

To add a new optimizer:

1. Create a new class in the `optimizers/` directory that inherits from `Optimizer`
2. Implement the required methods: `optimize()`, `pre_processing()`, and `post_processing()`
3. Add the optimizer configuration to `config/optimizers.yaml`

## 📊 Example Results

When running the platform, you'll get various outputs including:

- Fitness trend plots showing the convergence of different algorithms
- Statistical comparisons between algorithms
- For scheduling problems, Gantt charts of the best solutions found

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
