# Emergency Routing GA

A metaheuristic approach to **Emergency Vehicle Dispatch and Routing (EVDRP)** using multiple **Genetic Algorithm (GA)** variants — Standard, Elitist, Adaptive, and Hybrid.  
This project integrates **real-world map data** using the **OpenStreetMap API** to generate city-based routing datasets, enabling realistic optimization and comparative analysis.

---

## Project Overview

Emergency Vehicle Dispatch and Routing (EVDRP) is a crucial component of smart city management.  
The goal is to identify the **fastest and most efficient routes** for emergency vehicles such as ambulances, fire trucks, and police cars, considering distance and traffic density.

This project applies **metaheuristic optimization** techniques to dynamically adapt to real-world constraints using various GA models.

---

## Objectives

- Minimize total **response time** and **travel distance**  
- Maximize **coverage** and **service efficiency**  
- Compare and evaluate four **Genetic Algorithm variants**  
- Use **real map-based datasets** for better realism  

---

## GA Variants Implemented

| Variant | Description |
|----------|-------------|
| **Standard GA** | Baseline implementation using selection, crossover, and mutation. |
| **Elitist GA** | Retains best individuals in each generation to ensure solution quality. |
| **Adaptive GA** | Adjusts mutation and crossover rates dynamically to maintain diversity. |
| **Hybrid GA** | Combines GA with a local search heuristic (2-Opt) to refine solutions. |

---

## Dataset Generation

Real-world road networks from OpenStreetMap via `osmnx`.

### Generate Datasets
```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Generate city networks
python generate_dataset.py

# Analyze results
python analyze_datasets.py
```

### Cities Included
- Koramangala, Indiranagar, Jayanagar (Bangalore)
- Bangalore Central
- Manhattan Midtown, Cambridge MA

### Files Generated
- `{city}_nodes.csv` — Node coordinates
- `{city}_edges.csv` — Roads with distance, traffic, travel time
- `{city}_graph.graphml` — Network graph
- `datasets_summary.csv` — Overview
- `results/dataset_statistics.csv` — Analysis
- `results/datasets_table.tex` — LaTeX table for paper
  
---

## Methodology Overview

1. **Data Input**: Road network extracted using Maps API  
2. **Population Initialization**: Random sets of possible vehicle routes  
3. **Fitness Evaluation**: Based on weighted sum of response time and distance  
4. **Genetic Operations**:
 - Selection → Crossover → Mutation  
 - Elitism/Adaptation/Local Search applied as per variant  
5. **Termination**: Upon convergence or reaching max generations  
6. **Output**: Optimized dispatch plan and route set  

---

## Analysis & Results

The following analyses are used to compare algorithm performance:
- **Convergence Plots** – Fitness value vs. generations  
- **Box Plots** – Distribution of final fitness values  
- **Pareto Fronts** – Trade-off between distance and response time  

### Expected Findings
- **Hybrid GA** → Best accuracy and convergence  
- **Adaptive GA** → Best population diversity  
- **Elitist GA** → Balanced performance and stability  

---

## Team Members

| Name | Role |
|------|------|
| **Sruthi Mahadevan** | Team Lead & Hybrid GA Development|
| **Vipul Bohra** | Elitist GA  |
| **Vishnu L** | Adaptive GA Implementation & Experimental Analysis |
| **Sumit S Nair** | Standard GA, Integration & Visualization |

---

## References

1. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning.* Addison-Wesley.  
2. Srinivas, M., & Patnaik, L. M. (1994). *Adaptive probabilities of crossover and mutation in genetic algorithms.* IEEE Transactions on Systems, Man, and Cybernetics, 24(4), 656–667.  
3. Vidal, T., Crainic, T. G., Gendreau, M., & Prins, C. (2013). *A hybrid genetic algorithm for multi-depot and site-dependent vehicle routing problems.* Computers & Operations Research, 40(1), 158–173.  
4. Saxena, K., & Chauhan, D. S. (2010). *Elitism-based genetic algorithm for solving vehicle routing problem.* IJCA, 8(4), 6–10.  
5. Gendreau, M., Laporte, G., & Potvin, J. Y. (2002). *Vehicle routing: Modern heuristics.* Springer.

---

## Tech Stack

- **Python 3.x**
- **osmnx** – OpenStreetMap data extraction  
- **googlemaps** – Distance and travel time validation  
- **pandas**, **numpy** – Data handling and computation  
- **matplotlib**, **networkx** – Visualization  

---

## Future Scope

- Integrate **real-time traffic data** from APIs  
- Extend to **multi-depot emergency routing**  
- Deploy as a **web-based route optimization tool**  

---


