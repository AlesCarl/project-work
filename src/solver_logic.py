from __future__ import annotations # <--- AGGIUNGI QUESTA
import logging
import random
import time
import networkx as nx
import numpy as np
from numba import jit, prange # prange serve se usi i loop paralleli in Floyd-Warshall

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Problem import Problem


### ------------------------------------------------------------------------------
### 1. Initialization & Pre-computation
### ------------------------------------------------------------------------------

def _precompute_matrices(p: Problem):
    """
    Sostituisce la vecchia versione lenta.
    Genera sia liste (compatibilità) che numpy array (velocità).
    """
    n = len(list(p._graph.nodes))
    p._nodes_list = list(p._graph.nodes) # Sicurezza
    
    # 1. Costruisci matrice adiacenza densa per Numba
    adj_matrix = np.zeros((n, n), dtype=np.float64)
    for u, v, data in p._graph.edges(data=True):
        adj_matrix[u, v] = data['dist']
        adj_matrix[v, u] = data['dist'] # Grafo non orientato
        
    # 2. Floyd-Warshall Numba (Fulmineo)
    dist_mat, beta_mat = floyd_warshall_numba(adj_matrix, p.beta, n)
    
    # 3. Salvataggio Ibrido (Liste + Numpy)
    # Liste: per compatibilità con Genetic, _calculate_exact_cost, ecc.
    p._mat_dist = dist_mat.tolist()
    p._mat_beta = beta_mat.tolist()
    
    # Numpy: per ILS veloce
    p._mat_dist_np = dist_mat
    p._mat_beta_np = beta_mat
    
    # Cache Oro
    p._gold_cache = np.array([p._graph.nodes[i]['gold'] for i in range(n)], dtype=np.float64)
    
    # 4. Calcolo Vicini (Neighbor List) - Fondamentale per ILS
    p._neighbors = []
    for i in range(n):
        # Prendiamo i 40 vicini più stretti (escludendo se stessi)
        row = dist_mat[i].copy()
        row[i] = np.inf 
        sorted_idx = np.argsort(row)[:40] 
        p._neighbors.append(sorted_idx.tolist())
        
    p._matrix_init_done = True


@jit(nopython=True, parallel=True, fastmath=True)
def floyd_warshall_numba(adj_matrix, beta, n_cities):
    """
    Calcola All-Pairs Shortest Path per Distanza e Fisica (d^beta).
    O(N^3) ma fulmineo grazie a C+Parallel.
    """
    # Init matrici con infinito
    dist_mat = np.full((n_cities, n_cities), np.inf, dtype=np.float64)
    beta_mat = np.full((n_cities, n_cities), np.inf, dtype=np.float64)
    
    # Riempimento iniziale
    for i in range(n_cities):
        dist_mat[i, i] = 0.0
        beta_mat[i, i] = 0.0
        for j in range(n_cities):
            d = adj_matrix[i, j]
            if d > 0: # Esiste arco
                dist_mat[i, j] = d
                beta_mat[i, j] = d ** beta

    # Core Floyd-Warshall
    for k in range(n_cities):
        for i in prange(n_cities): # Parallel loop
            for j in range(n_cities):
                # Rilassamento
                d_new = dist_mat[i, k] + dist_mat[k, j]
                if d_new < dist_mat[i, j]:
                    dist_mat[i, j] = d_new
                    # La fisica è additiva sul percorso: sum(d^beta)
                    beta_mat[i, j] = beta_mat[i, k] + beta_mat[k, j]
                    
    return dist_mat, beta_mat


def get_safe_return_paths(p):  ## OCCHIO QUI
    n = len(p._nodes_list)
    beta = p.beta
    
    # 1. Pre-calcoliamo i pesi per velocizzare Dijkstra
    # (NetworkX è più veloce se legge un attributo che se chiama una funzione lambda)
    for u, v, d in p._graph.edges(data=True):
        d['phys_weight'] = d['dist'] ** beta

    # 2. Dijkstra Pesato (Fisico)
    # paths: dizionario {nodo: [0, n1, n2, ... nodo]}
    phys_costs, paths = nx.single_source_dijkstra(p._graph, 0, weight='phys_weight')
    
    # 3. Costruzione Array Predecessori (Compatibile Numba)
    # preds[i] contiene il nodo successivo per andare da i verso 0
    preds = np.zeros(n, dtype=np.int32)
    dist_from_0 = np.zeros(n, dtype=np.float64)
    beta_from_0 = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        beta_from_0[i] = phys_costs.get(i, float('inf'))
        
        path = paths.get(i, [])
        if len(path) > 1:
            # Il path di NetworkX è [0, A, B, i]. 
            # Per tornare da i a 0, il predecessore di i è B (path[-2])
            preds[i] = path[-2]
            
            # Calcolo distanza geometrica reale lungo il path sicuro
            d_real = 0.0
            for k in range(len(path) - 1):
                u_node, v_node = path[k], path[k+1]
                d_real += p._graph[u_node][v_node]['dist']
            dist_from_0[i] = d_real
        else:
            preds[i] = 0 # Base o irraggiungibile
            if i != 0: dist_from_0[i] = float('inf')
            
    return dist_from_0, beta_from_0, preds


# * `_precompute_matrices(p)`
# * `floyd_warshall_numba(adj_matrix, beta, n_cities)`
# * `get_safe_return_paths(p)`


### ------------------------------------------------------------------------------
### 2. Strategy: Hub & Spoke (Beta > 1)
### ------------------------------------------------------------------------------


def solve_hub_spoke_beta_high(p):#OK
    n_cities = len(p._nodes_list)
    
    # 1. Calcola Distanze e Predecessori (Veloce)
    safe_dist, safe_beta, preds = get_safe_return_paths(p)

    # 2. Recupero Oro
    if hasattr(p, '_gold_cache'):
        gold_arr = np.array(p._gold_cache, dtype=np.float64)
    else:
        gold_arr = np.array([p._graph.nodes[i]['gold'] for i in range(n_cities)], dtype=np.float64)
    
    # 3. Kernel Numba (Veloce)
    # Passiamo 'preds' invece che calcolare dopo
    raw_path = hub_spoke_numba_kernel_final(safe_dist, safe_beta, gold_arr, p.alpha, p.beta, n_cities, preds)
    
    # 4. Conversione Finale (Veloce, lista comprehension)
    # Non serve più logica complessa, il kernel ha già fatto tutto
    final_path = [(int(raw_path[i, 0]), raw_path[i, 1]) for i in range(len(raw_path))]
        
    if not final_path: return [(0,0)]
    return final_path


@jit(nopython=True)
def hub_spoke_numba_kernel_final(dist_from_0, beta_from_0, gold_arr, alpha, beta, n_cities, preds): #ok
    # Aumentato il buffer per contenere anche i waypoint intermedi
    # Stima conservativa: max_k * (2 + media_nodi_intermedi)
    max_k_allowed = 100000  #TODO - posso abbassarlo? si riducono i tempi??  -- 2000
    max_size = n_cities * max_k_allowed * 10 
    
    # Se la memoria è un problema, riduci 10 a 5, ma per N=1000 16GB reggono bene.
    path_data = np.zeros((max_size, 2), dtype=np.float64)
    idx = 0
    inv_beta = 1.0 / beta
    
    if alpha > 0: alpha_pow = alpha ** beta
    else: alpha_pow = 0.0

    for c in range(1, n_cities):
        gold = gold_arr[c]
        if gold <= 0: continue
            
        dist = dist_from_0[c]
        d_beta = beta_from_0[c]
        
        # --- Calcolo K Ottimo (Invariato) ---
        if alpha <= 0 or dist == 0: k = 1
        else:
            fatigue_term = alpha_pow * (gold ** beta) * d_beta
            numerator = fatigue_term * (beta - 1.0)
            denominator = 2.0 * dist
            
            if denominator < 1e-12: k = 1
            else:
                k_float = (numerator / denominator) ** inv_beta
                k_floor = int(k_float)
                if k_floor < 1: k_floor = 1
                k_ceil = k_floor + 1
                
                g1 = gold / k_floor
                c1 = k_floor * (2 * dist + (alpha * g1)**beta * d_beta)
                g2 = gold / k_ceil
                c2 = k_ceil * (2 * dist + (alpha * g2)**beta * d_beta)
                
                k = k_floor if c1 < c2 else k_ceil

        if k > max_k_allowed: k = max_k_allowed
        
        # Check Buffer (Safety Reale)
        # Stima approssimativa: ogni viaggio aggiunge almeno 2 nodi + waypoints
        # Se siamo vicini al limite, fermiamoci
        if idx + (k * 10) >= max_size: 
            break 
            
        gold_per_trip = gold / k
        
        for _ in range(k):
            # 1. Andata verso la città (Diretta, scarichi)
            path_data[idx, 0] = c
            path_data[idx, 1] = gold_per_trip
            idx += 1
            
            # 2. Ritorno verso la base (Sicuro, con Waypoints)
            curr = c
            while curr != 0:
                # Muoviti verso il predecessore (verso la base)
                curr = preds[curr]
                # Scrivi il nodo (con 0 oro)
                path_data[idx, 0] = curr
                path_data[idx, 1] = 0.0
                idx += 1
            
            # Nota: il while sopra scriverà (0, 0) come ultimo step, che è corretto.
            
    return path_data[:idx]


#* `solve_hub_spoke_beta_high(p)`
#* `hub_spoke_numba_kernel_final(dist_from_0, beta_from_0, gold_arr, alpha, beta, n_cities, preds)`


### ------------------------------------------------------------------------------
### 3. Strategy: Genetic Algorithm (N <= 200)
### ------------------------------------------------------------------------------


def solve_genetic(p: Problem):
    """
    GA Memetico POTENZIATO per istanze trattabili.
    - Look-ahead multi-step
    - Popolazione iniziale diversificata
    - Rotazioni circolari post-ottimizzazione
    """
    cities = [n for n in range(len(p._nodes_list)) if n != 0]
    num_cities = len(cities)
    
    # --- Tuning Parametri MIGLIORATI ---
    if num_cities < 50:
        population_size = 250       # +50
        generations = 400           # +200
        elite_size = 25             # +5
        mutation_rate = 0.35

    elif num_cities < 150:
        population_size = 180       # +30
        generations = 400           # +150
        elite_size = 20             # +5
        mutation_rate = 0.45

    elif num_cities <= 300:
        population_size = 150       # +30
        generations = 500           # +200
        elite_size = 15             # +3
        mutation_rate = 0.60
    else:
        # Fallback
        population_size = 60
        generations = 800
        elite_size = 5
        mutation_rate = 0.70

    population = []
    
    # 1. Inizializzazione DIVERSIFICATA
    # a) Greedy Nearest Neighbor (30%)
    num_greedy = int(population_size * 0.30)
    for _ in range(num_greedy):
        start_node = random.choice(cities)
        population.append(_nearest_neighbor_from(p, cities, start_node))
    
    # b) Savings Heuristic (20%) - Ottima per beta basso
    num_savings = int(population_size * 0.20)
    for _ in range(num_savings):
        tour = _savings_simple(p, cities)
        # Variante randomizzata per diversità
        if random.random() < 0.5:
            random.shuffle(tour)
        population.append(tour)
    
    # c) Farthest Insertion (10%) - Diversità geometrica
    num_farthest = int(population_size * 0.10)
    for _ in range(num_farthest):
        population.append(_farthest_insertion_simple(p, cities))
    
    # d) Random puro (40% rimanente)
    while len(population) < population_size:
        ind = list(cities)
        random.shuffle(ind)
        population.append(ind)
        
    # Valutazione con funzione POTENZIATA
    fitnesses = [_eval_chrom_enhanced(p, ind) for ind in population]
    
    # Best tracking
    best_idx = int(np.argmin(fitnesses))
    best_solution = population[best_idx]
    best_cost = fitnesses[best_idx]

    # Main Loop
    for gen in range(generations):
        new_population = []
        
        # Elitismo
        sorted_indices = np.argsort(fitnesses)
        for i in range(elite_size):
            new_population.append(population[sorted_indices[i]])
        
        # Generazione figli
        while len(new_population) < population_size:
            p1 = _tournament_selection(population, fitnesses)
            p2 = _tournament_selection(population, fitnesses)
            
            child = _order_crossover(p1, p2)
            
            if random.random() < mutation_rate:
                child = _mutation_hybrid(child)
            
            new_population.append(child)
        
        # Local Search (sporadica)
        if random.random() < 0.8:  
            idx_rnd = random.randint(elite_size, len(new_population)-1)
            new_population[idx_rnd] = _local_search_refine(p, new_population[idx_rnd], quick=True)
        
        # Elite Refine (periodica)
        if gen % 5 == 0: 
             new_population[0] = _local_search_refine(p, new_population[0], quick=False)

        population = new_population
        fitnesses = [_eval_chrom_enhanced(p, ind) for ind in population]
        
        curr_best_idx = int(np.argmin(fitnesses))
        if fitnesses[curr_best_idx] < best_cost:
            best_cost = fitnesses[curr_best_idx]
            best_solution = population[curr_best_idx]
    
    # Raffinamento finale
    best_solution = _refine_solution_final(p, best_solution)
    
    # NUOVO: Prova rotazioni circolari
    best_solution = _find_best_rotation(p, best_solution)
            
    return _build_path_ga_enhanced(p, best_solution)


def _eval_chrom_enhanced(p, chrom):
    """
    NUOVO: Valutazione POTENZIATA con look-ahead multi-step.
    Con beta basso, guarda 2-3 città avanti per decisioni migliori.
    FIX ALPHA=0: Se alpha è 0, cortocircuita su TSP Puro (velocissimo e immune a Beta).
    """
    # --- FAST PATH: ALPHA = 0 (Puro TSP) ---
    if p.alpha == 0.0:
        # Calcola semplicemente il perimetro del tour: Base -> c1 -> ... -> cn -> Base
        # Nessun calcolo di potenze, split o peso. Velocità max.
        d = p._mat_dist[0][chrom[0]]
        for i in range(len(chrom) - 1):
            d += p._mat_dist[chrom[i]][chrom[i+1]]
        d += p._mat_dist[chrom[-1]][0]
        return d
    

    # --- LOGICA NORMALE (Alpha > 0) ---
    mat_dist = p._mat_dist
    mat_beta = p._mat_beta
    golds = p._gold_cache
    beta = p.beta
    alpha_pow = p.alpha ** beta 
    
    total_cost = 0.0
    curr = 0
    w = 0.0
    n_genes = len(chrom)
    
    # Determina profondità look-ahead in base a beta
    if beta < 0.5:
        look_ahead = 3
    elif beta < 0.8:
        look_ahead = 2
    else:
        look_ahead = 1
    
    for i in range(n_genes):
        nxt = chrom[i]
        gold_nxt = golds[nxt]
        
        # === OPZIONE A: Scarico ORA e riparti ===
        cost_split = mat_dist[curr][0]
        if w > 0: 
            cost_split += (w ** beta) * alpha_pow * mat_beta[curr][0]
        cost_split += mat_dist[0][nxt]
        
        # Simula il futuro dopo lo scarico
        future_w_split = gold_nxt
        future_curr_split = nxt
        future_cost_split = 0.0
        
        for k in range(1, min(look_ahead + 1, n_genes - i)):
            fut_city = chrom[i + k]
            future_cost_split += mat_dist[future_curr_split][fut_city]
            if future_w_split > 0:
                future_cost_split += (future_w_split ** beta) * alpha_pow * mat_beta[future_curr_split][fut_city]
            future_w_split += golds[fut_city]
            future_curr_split = fut_city
        
        score_split = cost_split + future_cost_split
        
        # === OPZIONE B: Continua ad ACCUMULARE ===
        cost_direct = mat_dist[curr][nxt]
        if w > 0:
            cost_direct += (w ** beta) * alpha_pow * mat_beta[curr][nxt]
        
        future_w_direct = w + gold_nxt
        future_curr_direct = nxt
        future_cost_direct = 0.0
        
        for k in range(1, min(look_ahead + 1, n_genes - i)):
            fut_city = chrom[i + k]
            future_cost_direct += mat_dist[future_curr_direct][fut_city]
            if future_w_direct > 0:
                future_cost_direct += (future_w_direct ** beta) * alpha_pow * mat_beta[future_curr_direct][fut_city]
            future_w_direct += golds[fut_city]
            future_curr_direct = fut_city
        
        score_direct = cost_direct + future_cost_direct
        
        # Scelta ottima locale
        if score_split < score_direct:
            total_cost += cost_split
            w = gold_nxt
        else:
            total_cost += cost_direct
            w = w + gold_nxt
        
        curr = nxt
    
    # Chiusura finale
    total_cost += mat_dist[curr][0]
    if w > 0:
        total_cost += (w ** beta) * alpha_pow * mat_beta[curr][0]
    
    return total_cost


def _tournament_selection(population, fitnesses, k=3):
    idxs = random.sample(range(len(population)), k)
    best_i = min(idxs, key=lambda i: fitnesses[i])
    return population[best_i]


def _order_crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b+1] = p1[a:b+1]
    pos = 0
    for item in p2:
        if item not in child:
            while child[pos] != -1: pos += 1
            child[pos] = item
    return child


def _mutation_hybrid(sol):
    r = random.random()
    if r < 0.33: return _mutation_swap(sol)
    elif r < 0.66: return _mutation_insert(sol)
    else: return _mutation_inversion(sol)


def _mutation_swap(sol):
    s = sol[:]
    i, j = random.sample(range(len(s)), 2)
    s[i], s[j] = s[j], s[i]
    return s


def _mutation_insert(sol):
    s = sol[:]
    i, j = random.sample(range(len(s)), 2)
    c = s.pop(i)
    s.insert(j, c)
    return s


def _mutation_inversion(sol):
    s = sol[:]
    i, j = sorted(random.sample(range(len(s)), 2))
    s[i:j+1] = s[i:j+1][::-1]
    return s


def _local_search_refine(p, solution, quick=False):
    best_sol = solution[:]
    best_cost = _eval_chrom_enhanced(p, best_sol)
    n = len(best_sol)
    
    if quick:
        num_iter_2opt = 20
        # FIX SICUREZZA: Mai chiedere più di n
        num_iter_insert = min(n, 15)
        max_loops = 1 
        neighbors_to_check = 6
    else:
        num_iter_2opt = max(50, min(int(n * 0.5), 300))
        
        # FIX SICUREZZA: Calcola target e limita a n
        target_insert = max(50, min(int(n * 0.4), 300))
        num_iter_insert = min(n, target_insert)
        
        max_loops = 3 if n > 500 else 4
        neighbors_to_check = 20

    improved = True
    loop_count = 0
    
    while improved and loop_count < max_loops:
        improved = False
        loop_count += 1
        
        # 2-Opt
        for _ in range(num_iter_2opt): 
            if n < 3: break # Sicurezza N piccolissimi
            i, j = sorted(random.sample(range(n), 2))
            if j - i < 2: continue
            if quick and (j - i) > (n / 4): continue
            new_sol = best_sol[:i] + best_sol[i:j+1][::-1] + best_sol[j+1:]
            new_cost = _eval_chrom_enhanced(p, new_sol)
            if new_cost < best_cost:
                best_sol = new_sol
                best_cost = new_cost
                improved = True
                if quick: break
        
        if improved and quick: break
        if improved: continue

        # Guided Insert
        if num_iter_insert > 0:
            target_indices = random.sample(range(n), num_iter_insert)
            for idx in target_indices:
                city = best_sol[idx]
                temp_sol = best_sol[:idx] + best_sol[idx+1:]
                
                neighbors = p._neighbors[city]
                candidate_positions = set()
                for neighbor in neighbors[:neighbors_to_check]:
                    try:
                        pos_neighbor = temp_sol.index(neighbor)
                        candidate_positions.add(pos_neighbor) 
                        candidate_positions.add(pos_neighbor + 1)
                    except ValueError: continue

                # FIX SICUREZZA: random.sample non deve superare la lunghezza
                n_temp = len(temp_sol) + 1
                k_rnd = 1 if quick else 2
                k_rnd = min(k_rnd, n_temp)
                
                if k_rnd > 0:
                    candidate_positions.update(random.sample(range(n_temp), k_rnd))
                
                found_better = False
                for pos in candidate_positions:
                    if pos > len(temp_sol): pos = len(temp_sol)
                    cand = temp_sol[:pos] + [city] + temp_sol[pos:]
                    c = _eval_chrom_enhanced(p, cand)
                    if c < best_cost:
                        best_sol = cand
                        best_cost = c
                        improved = True
                        found_better = True
                        break 
                
                if found_better:
                    if quick: break
                    else: break
            
    return best_sol


def _refine_solution_final(p, sol):
    best_s = sol[:]
    best_c = _eval_chrom_enhanced(p, best_s)  # UPDATED
    n = len(best_s)
    improved = True
    window = 50 if n > 200 else n 
    max_passes = 150
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(0, n - 2):
            limit_j = min(n, i + window)
            for j in range(i + 2, limit_j):
                new_s = best_s[:i] + best_s[i:j+1][::-1] + best_s[j+1:]
                c = _eval_chrom_enhanced(p, new_s)  # UPDATED 
                if c < best_c:
                    best_c = c
                    best_s = new_s 
                    improved = True 
                    break 
            if improved: break
    return best_s


def _build_path_ga_enhanced(p, chrom):
    """
    NUOVO: Costruisce il path usando la stessa logica di _eval_chrom_enhanced.
    FIX ALPHA=0: Costruisce un tour semplice senza rientri alla base.
    """

    # --- FAST PATH: ALPHA = 0 (Puro TSP) ---
    if p.alpha == 0.0:
        path = []
        for city in chrom:
            path.append((city, p._gold_cache[city]))
        path.append((0, 0))
        return path


    # --- LOGICA NORMALE (Alpha > 0) ---
    mat_dist = p._mat_dist
    mat_beta = p._mat_beta
    golds = p._gold_cache
    beta = p.beta
    alpha_pow = p.alpha ** beta 

    path = []
    curr = 0
    w = 0.0
    n_genes = len(chrom)
    
    # Stesso look-ahead della eval
    if beta < 0.5:
        look_ahead = 3
    elif beta < 0.8:
        look_ahead = 2
    else:
        look_ahead = 1

    for i in range(n_genes):
        nxt = chrom[i]
        gold_nxt = golds[nxt]
        
        # Opzione A: Split
        c_split = mat_dist[curr][0] + mat_dist[0][nxt]
        if w > 0: c_split += (w ** beta) * alpha_pow * mat_beta[curr][0]
        
        future_w_s = gold_nxt
        future_curr_s = nxt
        future_cost_s = 0.0
        for k in range(1, min(look_ahead + 1, n_genes - i)):
            fut_city = chrom[i + k]
            future_cost_s += mat_dist[future_curr_s][fut_city]
            if future_w_s > 0:
                future_cost_s += (future_w_s ** beta) * alpha_pow * mat_beta[future_curr_s][fut_city]
            future_w_s += golds[fut_city]
            future_curr_s = fut_city
        score_split = c_split + future_cost_s
        
        # Opzione B: Direct
        c_direct = mat_dist[curr][nxt]
        if w > 0: c_direct += (w ** beta) * alpha_pow * mat_beta[curr][nxt]
        
        future_w_d = w + gold_nxt
        future_curr_d = nxt
        future_cost_d = 0.0
        for k in range(1, min(look_ahead + 1, n_genes - i)):
            fut_city = chrom[i + k]
            future_cost_d += mat_dist[future_curr_d][fut_city]
            if future_w_d > 0:
                future_cost_d += (future_w_d ** beta) * alpha_pow * mat_beta[future_curr_d][fut_city]
            future_w_d += golds[fut_city]
            future_curr_d = fut_city
        score_direct = c_direct + future_cost_d

        if score_split < score_direct:
            if curr != 0: path.append((0, 0))
            path.append((nxt, gold_nxt))
            w = gold_nxt
        else:
            path.append((nxt, gold_nxt))
            w = w + gold_nxt
        curr = nxt
        
    path.append((0, 0))
    return path


def _find_best_rotation(p, tour):
    """
    NUOVO: Prova tutte le rotazioni circolari del tour e restituisce la migliore.
    Esempio: [A,B,C,D] -> prova [B,C,D,A], [C,D,A,B], [D,A,B,C]
    """
    n = len(tour)
    best_tour = tour[:]
    best_cost = _eval_chrom_enhanced(p, best_tour)
    
    # Prova max 50 rotazioni (se N è grande, campiona)
    step = max(1, n // 50)
    
    for start in range(0, n, step):
        rotated = tour[start:] + tour[:start]
        cost = _eval_chrom_enhanced(p, rotated)
        
        if cost < best_cost:
            best_cost = cost
            best_tour = rotated
    
    return best_tour



# * `solve_genetic(p)`
# * `_eval_chrom_enhanced(p, chrom)`
# * `_tournament_selection(population, fitnesses, k=3)`
# * `_order_crossover(p1, p2)`
# * `_mutation_hybrid(sol)`
# * `_mutation_swap(sol)`
# * `_mutation_insert(sol)`
# * `_mutation_inversion(sol)`
# * `_local_search_refine(p, solution, quick=False)`
# * `_refine_solution_final(p, sol)`
# * `_find_best_rotation(p, tour)`
# * `_build_path_ga_enhanced(p, chrom)`


### ------------------------------------------------------------------------------
### 4. Strategy: Iterated Local Search (N > 200)
### ------------------------------------------------------------------------------


def solve_ils(p: Problem, max_iter=60): # 60 iter bastano con neighbor list
    
    # 1. Init (Chiama la NUOVA _precompute_matrices se serve)
    if not hasattr(p, '_matrix_init_done'):
        _precompute_matrices(p) 

    # 2. Recupero dati ULTRA-VELOCE (diretto da Numpy)
    # Non serve più convertire le liste in array, li abbiamo già!
    dist_mat = p._mat_dist_np 
    neighbors = np.array(p._neighbors, dtype=np.int32)
    
    # B. Greedy NN Iniziale (Velocissimo)
    n_cities = len(p._nodes_list)
    unvisited = np.ones(n_cities, dtype=bool)
    unvisited[0] = False
    
    tour = [0]
    curr = 0
    count = 1
    
    # Greedy NN con Neighbor List Lookup
    while count < n_cities:
        found = False
        # Prima cerca nei vicini
        for nxt in p._neighbors[curr]:
            if unvisited[nxt]:
                unvisited[nxt] = False
                tour.append(nxt)
                curr = nxt
                found = True
                break
        if not found:
            # Fallback lento (raro)
            dists = dist_mat[curr]
            # Maschera visitati mettendo inf
            temp_dists = np.where(unvisited, dists, np.inf)
            nxt = np.argmin(temp_dists)
            unvisited[nxt] = False
            tour.append(nxt)
            curr = nxt
        count += 1
    tour.append(0)
    
    # C. ILS Loop
    best_tour = np.array(tour, dtype=np.int32)
    # Valuta geometricamente (distanza pura)
    best_len = 0.0
    for i in range(len(best_tour)-1):
        best_len += dist_mat[best_tour[i], best_tour[i+1]]
        
    curr_tour = best_tour.copy()
    
    no_improv = 0
    
    for _ in range(max_iter):
        
        # 1. Local Search (Neighbor 2-Opt)
        curr_tour = ils_2opt_neighbor_fast(dist_mat, curr_tour, neighbors, k_check=30)
        
        # 2. Calc Length
        curr_len = 0.0
        for i in range(len(curr_tour)-1):
            curr_len += dist_mat[curr_tour[i], curr_tour[i+1]]
            
        # 3. Acceptance
        if curr_len < best_len - 1e-4:
            best_len = curr_len
            best_tour = curr_tour.copy()
            no_improv = 0
        else:
            no_improv += 1
            if no_improv > 10: # Restart
                curr_tour = best_tour.copy()
            
        # 4. Perturbation (Double Bridge - semplice ed efficace)
        if len(curr_tour) > 8:
            n = len(curr_tour) - 1 # Escludi ultimo 0
            idx = np.sort(np.random.choice(np.arange(1, n), 3, replace=False))
            i, j, k = idx
            # A-B-C-D -> A-D-C-B
            # Double Bridge standard:
            # tour = A + D + C + B + [0]
            # Segmenti: [0...i], [i...j], [j...k], [k...n]
            # Nuovo: [0...i] + [k...n] + [j...k] + [i...j]
            # Nota: deve finire con 0
            
            # Slice numpy
            seg_A = curr_tour[:i]
            seg_B = curr_tour[i:j]
            seg_C = curr_tour[j:k]
            seg_D = curr_tour[k:-1] # Escludi 0 finale
            
            # Ricostruisci
            # Pattern Double Bridge: A D C B
            new_t = np.concatenate((seg_A, seg_D, seg_C, seg_B))
            curr_tour = np.append(new_t, 0) # Riaggiungi 0

    # D. Split Finale (DP)
    # Converti in lista python per compatibilità DP
    clean_tour_list = [int(x) for x in best_tour if x != 0]
    
    # Usa la DP che è sicura e precisa
    path_fw = _split_tour_optimal_dp(p, clean_tour_list)
    path_bw = _split_tour_optimal_dp(p, clean_tour_list[::-1])
    
    cost_fw = _calculate_exact_cost(p, path_fw)
    cost_bw = _calculate_exact_cost(p, path_bw)
    
    return path_fw if cost_fw < cost_bw else path_bw


@jit(nopython=True)
def ils_2opt_neighbor_fast(dist_mat, tour, neighbors, k_check=25):
    """
    Esegue 2-Opt usando la lista dei vicini.
    Complessità O(N * k_check) invece di O(N^2).
    """
    n = len(tour)
    improved = True
    
    # Mappa posizione nodo -> indice nel tour (per lookup O(1))
    pos = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        pos[tour[i]] = i
        
    while improved:
        improved = False
        for i in range(n - 1):
            u = tour[i]
            v = tour[i+1] # Arco u -> v
            
            # Controlla solo i vicini geografici di u
            for k in range(k_check):
                neighbor = neighbors[u, k]
                if neighbor == -1: break # Fine lista
                if neighbor == 0: continue # Ignora base in check intermedi
                
                # Vogliamo provare a collegare u -> neighbor
                # Dobbiamo trovare dov'è 'neighbor' nel tour
                j = pos[neighbor]
                
                # Gestione indici circolari e adiacenze
                if j == i + 1: continue # È già il successivo
                
                # Per 2-opt standard, assumiamo scambio arco (u,v) e (x,y)
                # Se j > i: u->v ... neighbor(j)->next(j+1)
                # Swap diventa: u->neighbor ... v->next(j+1)
                
                # Semplificazione robusta per Numba:
                # Se neighbor è 'prima' di u, ignoriamo (evita doppi check)
                if j <= i + 1: continue
                
                # Arco 1: u(i) -> v(i+1)
                # Arco 2: neighbor(j) -> next_neighbor(j+1)
                if j == n - 1: continue # Evita overflow boundary
                
                next_neighbor = tour[j+1]
                
                curr_dist = dist_mat[u, v] + dist_mat[neighbor, next_neighbor]
                new_dist = dist_mat[u, neighbor] + dist_mat[v, next_neighbor]
                
                if new_dist < curr_dist - 1e-6:
                    # Esegui swap: inverte segmento tour[i+1 : j+1]
                    # In Numba bisogna farlo manualmente o con slice intelligente
                    p1 = i + 1
                    p2 = j
                    
                    # Reverse in-place
                    while p1 < p2:
                        temp = tour[p1]
                        tour[p1] = tour[p2]
                        tour[p2] = temp
                        
                        # Aggiorna posizioni
                        pos[tour[p1]] = p1
                        pos[tour[p2]] = p2
                        
                        p1 += 1
                        p2 -= 1
                    
                    # Aggiorna posizioni centrali (se dispari)
                    if p1 == p2:
                         pos[tour[p1]] = p1
                         
                    improved = True
                    break # Restart greedy (Strategy First Improvement)
            if improved: break
            
    return tour


def _split_tour_optimal_dp(p, tour):
    """
    NUOVO: Trova gli split ottimali con Programmazione Dinamica O(N²).
    Per ogni posizione i, prova tutti i possibili viaggi A/R che terminano lì.
    """

    # --- SHORTCUT ISTANTANEO PER ALPHA=0 ---
    if p.alpha == 0.0:
        # Se la fatica è 0, tornare alla base è sempre uno spreco di strada.
        # Restituisci il tour intero come un unico viaggio. Costo minimo, Tempo zero.
        path = []
        for city in tour:
            # Recupero oro sicuro
            g = p._gold_cache[city] if hasattr(p, '_gold_cache') else p._graph.nodes[city]['gold']
            path.append((city, g))
        path.append((0, 0))
        return path
    # ---------------------------------------


    n = len(tour)
    INF = float('inf')
    
    # dp[i] = costo minimo per servire tour[0:i] con viaggi A/R dalla base
    dp = [INF] * (n + 1)
    split_at = [-1] * (n + 1)
    dp[0] = 0.0
    
    # Limite massimo di lunghezza del viaggio basato su Beta
    if p.beta < 0.4:
        max_trip_len = n
    elif p.beta < 0.7:
        max_trip_len = min(50, n)
    else:
        max_trip_len = min(20, n)
    
    
    for i in range(1, n + 1):
        # Prova tutti i viaggi possibili che terminano in tour[i-1]
        for start in range(max(0, i - max_trip_len), i):
            # Viaggio: Base -> tour[start] -> ... -> tour[i-1] -> Base
            segment = tour[start:i]
            trip_cost = _calc_trip_cost_accurate(p, segment)
            
            candidate_cost = dp[start] + trip_cost
            if candidate_cost < dp[i]:
                dp[i] = candidate_cost
                split_at[i] = start
    
    # Ricostruzione del path ottimale
    path = []
    pos = n
    
    while pos > 0:
        start = split_at[pos]
        # Aggiungi il segmento tour[start:pos]
        for city in tour[start:pos]:
            path.append((city, p._gold_cache[city]))
        path.append((0, 0))  # Ritorno alla base
        pos = start
    
    return path

# * `solve_ils(p, max_iter)`
# * `ils_2opt_neighbor_fast(dist_mat, tour, neighbors, k_check=25)`
# * `_split_tour_optimal_dp(p, tour)`


### ------------------------------------------------------------------------------
### 5. Common Heuristics & Helpers
### ------------------------------------------------------------------------------


def _nearest_neighbor_from(p, cities, start):
    """Greedy Nearest Neighbor Unificato"""
    unvisited = set(cities)
    unvisited.remove(start)
    tour = [start]
    curr = start

    while unvisited:
        nxt = min(unvisited, key=lambda x: p._mat_dist[curr][x])
        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    return tour


def _savings_simple(p, cities):
    if len(cities) <= 1: return cities[:]
    sorted_cities = sorted(cities, key=lambda c: p._mat_dist[0][c])
    tour = [sorted_cities[0]]
    remaining = set(sorted_cities[1:])

    while remaining:
        best_city = None
        best_saving = -float('inf')
        best_pos = 0

        for city in remaining:
            for pos in [0, len(tour)]:
                if pos == 0:
                    prev, nxt = 0, tour[0]
                else:
                    prev, nxt = tour[-1], 0

                saving = (p._mat_dist[prev][nxt] - p._mat_dist[prev][city] - p._mat_dist[city][nxt])
                if saving > best_saving:
                    best_saving = saving
                    best_city = city
                    best_pos = pos

        if best_city:
            if best_pos == 0: tour.insert(0, best_city)
            else: tour.append(best_city)
            remaining.remove(best_city)
        else:
            tour.append(remaining.pop())
    return tour


def _farthest_insertion_simple(p, cities):
    if len(cities) <= 1: return cities[:]
    tour = [max(cities, key=lambda c: p._mat_dist[0][c])]
    remaining = set(cities) - set(tour)

    while remaining:
        farthest = max(remaining, key=lambda c: min(p._mat_dist[c][t] for t in tour))
        best_pos = len(tour)
        tour.insert(best_pos, farthest)
        remaining.remove(farthest)
    return tour


def _calc_trip_cost_accurate(p, segment):
    """
    Calcola il costo ESATTO di un viaggio A/R che visita 'segment'.
    Base -> segment[0] -> segment[1] -> ... -> segment[-1] -> Base
    """
    if not segment:
        return 0.0
    
    cost = 0.0
    curr = 0  # Partiamo dalla base
    w = 0.0   # Peso iniziale zero
    
    # Percorso: Base -> città in sequenza
    for city in segment:
        cost += _get_cost_matrix(p, curr, city, w)
        w += p._gold_cache[city]
        curr = city
    
    # Ritorno: ultima città -> Base
    cost += _get_cost_matrix(p, curr, 0, w)
    
    return cost

# * `_nearest_neighbor_from(p, cities, start)`
# * `_savings_simple(p, cities)`
# * `_farthest_insertion_simple(p, cities)`
# * `_calc_trip_cost_accurate(p, segment)`


### ------------------------------------------------------------------------------
### 6. Cost Calculation Functions
### ------------------------------------------------------------------------------


def check_solution_cost(p: Problem, solution_path):
    """
    Versione HYBRID FINAL:
    - Andata (0->Città): Usa Dijkstra One-To-All precalcolato (Veloce per N partenze)
    - Ritorno (Città->Predecessori): Usa check adiacenza O(1) (Veloce per path lunghi)
    """
    # 1. Se abbiamo matrici precalcolate (Beta <= 1), usiamole
    if hasattr(p, '_matrix_init_done') and p._matrix_init_done:
        return _calculate_exact_cost(p, solution_path)

    # 2. PRE-CALCOLO DISTANZE DA 0 (Andata)
    # Calcoliamo UNA volta sola tutte le distanze dalla base.
    # W=0 in andata, quindi non serve calcolare la fisica, basta la distanza.
    try:
        # single_source_dijkstra_path_length è più veloce e leggero di _path
        dists_from_0 = nx.single_source_dijkstra_path_length(p._graph, 0, weight='dist')
    except:
        return float('inf')

    total_cost = 0.0
    curr_node = 0
    curr_w = 0.0
    
    alpha = p.alpha
    beta = p.beta
    graph = p._graph

    for next_node, collected_gold in solution_path:
        cost_leg = 0.0
        
        # --- CASO 1: Movimento dalla Base (Andata) ---
        if curr_node == 0:
            # Siamo scarichi (W=0). Costo = Distanza Pura.
            # Lookup O(1) nel dizionario precalcolato
            cost_leg = dists_from_0.get(next_node, float('inf'))
            if cost_leg == float('inf'): return float('inf')
        
        # --- CASO 2: Movimento verso la Base o tra Città (Ritorno) ---
        else:
            # FIX OPTIMIZATION: Check adiacenza O(1)
            # Questo gestisce i milioni di passi del ritorno istantaneamente
            if graph.has_edge(curr_node, next_node):
                d = graph[curr_node][next_node]['dist']
                if curr_w > 0:
                    cost_leg = d + ((alpha * curr_w * d) ** beta)
                else:
                    cost_leg = d
            else:
                # Fallback Lento (Dijkstra)
                # Si attiva solo se il path salta tra nodi non connessi (raro in Hub&Spoke)
                try:
                    leg_path = nx.shortest_path(graph, curr_node, next_node, weight='dist')
                    leg_dist = 0.0
                    leg_fatigue = 0.0
                    
                    for k in range(len(leg_path)-1):
                        u, v = leg_path[k], leg_path[k+1]
                        d_edge = graph[u][v]['dist']
                        leg_dist += d_edge
                        if curr_w > 0:
                            leg_fatigue += ((alpha * curr_w * d_edge) ** beta)
                            
                    cost_leg = leg_dist + leg_fatigue
                except Exception:
                    return float('inf')

        total_cost += cost_leg
        
        # Aggiornamento Stato
        curr_node = next_node
        if curr_node == 0:
            curr_w = 0.0
        else:
            curr_w += collected_gold
            
    return total_cost


def _calculate_exact_cost(p, path):
    tot, curr, w = 0.0, 0, 0.0
    for node, gold in path:
        tot += _get_cost_matrix(p, curr, node, w)
        w = 0 if node == 0 else w + gold
        curr = node
    return tot


def _get_cost_matrix(p, u, v, w):
    if u == v: return 0.0
    d = p._mat_dist[u][v]
    return d if w == 0 else d + ((p.alpha * w) ** p.beta) * p._mat_beta[u][v]


# * `check_solution_cost(p, solution_path)`
# * `_calculate_exact_cost(p, path)`
# * `_get_cost_matrix(p, u, v, w)`
