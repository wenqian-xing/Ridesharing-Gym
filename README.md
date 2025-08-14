# NYC Rideshare Treatment Effect Benchmark

A JAX-accelerated, high-fidelity rideshare simulation for evaluating treatment effect estimation methods using real Manhattan street network and taxi data.

## Overview

This benchmark provides a realistic NYC rideshare simulation environment with enhanced driver dynamics for testing causal inference algorithms. Built with JAX for scalable batch processing and vectorized driver behavior modeling.

## Quick Start

```bash
# Install dependencies
poetry install

# Run NYC rideshare simulation demo
poetry run python run_rideshare_demo.py

# Run tests
poetry run python -m pytest test_benchmark.py -v
```

## Installation

```bash
# Install Poetry if needed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies (requires Python 3.11)
poetry install
```

**Requirements**: Python 3.11, JAX, Flax, and other dependencies managed through Poetry.

## Features

### NYC Rideshare Simulation
- **High-fidelity modeling**: 300 drivers, 50K+ ride requests using real Manhattan street network
- **Enhanced driver dynamics**: 6-state driver lifecycle with realistic behavior patterns
- **Real-world data**: 2.98M historical NYC taxi trips and 4,333-node street network
- **JAX acceleration**: Vectorized batch processing for scalable experimentation

### Driver Behavior Modeling
- **Experience levels**: New (25%), experienced (45%), veteran (30%) drivers
- **Shift patterns**: Full-time, part-time day/evening, weekend workers
- **Economic decisions**: Daily hours/earnings targets, multi-platform competition
- **Spatial intelligence**: Home locations, preferred work zones, repositioning strategies

### Treatment Effect Estimation
- **Multiple estimators**: Direct Method, Truncated DQ, OPE, Stationary DQ
- **Experimental designs**: Switchback (10/20/30/60 min), randomized A/B testing
- **Dynamic pricing**: Treatment effect estimation for ride pricing policies

## Core Architecture

### Simulation Engine
- **`picard/rideshare_dispatch.py`**: JAX-based NYC rideshare environment with enhanced driver dynamics
- **`environments/rideshare.py`**: Treatment effect wrapper with vectorized batch processing
- **`treatment_effect_gym.py`**: Benchmark framework with abstract base classes

### Enhanced Driver Dynamics  
- **`enhanced_driver_dynamics.py`**: Realistic driver behavior modeling implementation
- **Driver states**: Offline, online idle, en route, on trip, repositioning, break
- **Economic behavior**: Shift patterns, earnings targets, multi-platform competition

### Estimation Methods
- **`estimators/`**: Treatment effect estimators (Direct Method, Truncated DQ, OPE, Stationary DQ)
- **`experiment_designs/`**: Experimental designs (Switchback, randomized A/B testing)

### Data & Results
- **`data/`**: Manhattan street network and 2.98M historical taxi trips
- **`rideshare_demo_results/`**: Generated benchmarks and visualizations

## Performance

- **Scale**: 500K+ ride requests × 100 parallel environments
- **Speed**: JAX vectorization provides ~10x acceleration
- **Realism**: Real Manhattan geography + data-driven driver behavior