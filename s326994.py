
from Problem import Problem
#import solver_logic as sl
import networkx as nx 
from src import solver_logic as sl


def make_path_admissible(p: Problem, path):
    """
    Fills the "jumps" in the path using the actual shortest path in the graph.
    """

    if not path:
        return path
    
    real_path = []
    curr = 0 
    
    for next_node, gold in path:
        
        if curr == next_node:
            continue
            
        # If a direct edge exists, we simply add the node
        if p._graph.has_edge(curr, next_node):
            real_path.append((int(next_node), float(gold)))
        else:

            try:
                route = nx.shortest_path(p._graph, curr, next_node, weight='dist')
            except nx.NetworkXNoPath:
                # Caso estremo
                return path 
            
            # Add intermediate nodes with gold = 0
            for node in route[1:-1]:
                real_path.append((node, 0.0))

            # Add the final destination with its gold value
            real_path.append((int(next_node), float(gold)))
            
        curr = next_node
        
    return real_path


def is_valid(path, p: Problem):

    if not path:
        return False

    '''
    # 2.  FINIRE a (0, 0)      ## è un extra:
    if path[-1][0] != 0:
        return False
    '''

    # 3. Controllo Connettività (Archi esistenti)
    for (c1, g1), (c2, g2) in zip(path, path[1:]):
        if not p._graph.has_edge(c1, c2):
            return False
            
    return True




def is_valid_old(path, p: Problem):
    """
    Checks strictly if the path consists of existing edges.
    """
    for (c1, g1), (c2, g2) in zip(path, path[1:]):
        if not p._graph.has_edge(c1, c2):
            return False
    return True





def solution(p: Problem):
    
    # 1. Init veloce
    p._nodes_list = list(p._graph.nodes)
    n_cities = len(p._nodes_list)
    
    # --- STRATEGY SELECTION ---
    
    final_path = []

    # Strategy 1: Beta > 1 (Hub & Spoke)
    if p.beta > 1.0 and p.alpha != 0:       
        raw_path = sl.solve_hub_spoke_beta(p)
        final_path = make_path_admissible(p, raw_path)

    else:
        # Pre-compute matrices ( per GA/ILS)
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
        
        
        final_path = make_path_admissible(p, path)


    # --- SAFETY CHECK & BASELINE COMPARISON ---
    
    # Baseline
    gold_vals = p._gold_cache if hasattr(p, '_gold_cache') else [p._graph.nodes[i]['gold'] for i in range(n_cities)]
    base_raw = []
    for c in range(1, n_cities):
        if gold_vals[c] > 0:
            base_raw.extend([(c, gold_vals[c]), (0, 0)])
    
    base_path = make_path_admissible(p, base_raw)

    if not final_path:
        return base_path


    
    cost_algo = sl.check_solution_cost(p, final_path)
    cost_base = sl.check_solution_cost(p, base_path)

    best_path = final_path if cost_algo < cost_base else base_path 


    # --- FINAL VALIDATION ---
    
    if is_valid(best_path, p):
        #print(f"[OK] Best Path valid (len: {len(best_path)}). Returning it.")    
        return best_path
        
    elif is_valid(base_path, p):
        #print(f"[WARN] Best Path INVALID. Fallback to Baseline (len: {len(base_path)}).")   
        return base_path
        
    else:
        #print("[CRITICAL] Both paths INVALID. Returning broken Algo Path.")    
        return best_path
    


