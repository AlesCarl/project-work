import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from Problem import Problem
import solver_logic as sl




def solution(p: Problem):
    
    # Setup graph parameters
    p._nodes_list = list(p._graph.nodes)
    n_cities = len(p._nodes_list)
    
    # --- STRATEGY SELECTION ---
    
    # Strategy 1: Beta > 1 (Hub & Spoke)
    if p.beta > 1.0 and p.alpha != 0:       
        return sl.solve_hub_spoke_beta_high(p)

    # Pre-compute distance and fatigue matrices for GA and ILS
    if not hasattr(p, '_matrix_init_done'):
        sl._precompute_matrices(p)

    active_cities_count = n_cities - 1
    path = []

    # Strategy 2: Alpha = 0 (Pure TSP)
    if p.alpha == 0.0:
        if active_cities_count <= 200:
            path = sl.solve_genetic(p)
        else:
            path = sl.solve_ils(p, max_iter=1000)

    # Strategy 3: Beta <= 1 (Accumulation)
    else:
        if active_cities_count <= 200:
            path = sl.solve_genetic(p)
        else:
            path = sl.solve_ils(p, max_iter=500)


    # --- SAFETY CHECK & BASELINE COMPARISON ---
    
    # 1. Build the baseline path
    gold_vals = p._gold_cache if hasattr(p, '_gold_cache') else [p._graph.nodes[i]['gold'] for i in range(n_cities)]
    base_path = []
    for c in range(1, n_cities):
        if gold_vals[c] > 0:
            base_path.extend([(c, gold_vals[c]), (0, 0)])

    # 2. Return baseline if solver failed
    if not path:
        return base_path

    # 3. Compare costs using the hybrid optimized cost function
    cost_algo = sl.check_solution_cost(p, path)
    cost_base = sl.check_solution_cost(p, base_path)

    # 4. Return the path with the minimum cost
    if cost_algo < cost_base:
            return path
    else:
            return base_path
