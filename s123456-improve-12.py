import logging
import random
import time
import networkx as nx
import numpy as np
from Problem import Problem

# ==============================================================================
# DISPATCHER: MAIN ENTRY POINT
# ==============================================================================

def solution(p: Problem):
    # 1. Init Matrici (Lazy Loading) - Eseguito una volta sola
    if not hasattr(p, '_matrix_init_done'):
        _precompute_matrices(p)

    n_cities = len(p._nodes_list)
    
    # 2. Selezione Strategia
    if p.beta > 1.0:
        # --- CASO SCARICO (Beta Alto) ---
        # Strategia: Hub & Spoke con Merge
        path = solve_merge(p)
    else:


    # ------------------------------------------------------------------------------------
    ## TODO : con B=1 hai il caso speciale che devi usare 
    # ------------------------------------------------------------------------------------


        # --- CASO ACCUMULO (Beta Basso) ---
        # Contiamo le città attive (escluso base e città vuote per density < 1)
        active_cities_count = len([c for c in range(n_cities) if c != 0 and p._gold_cache[c] > 0])
        
        # SOGLIA CRITICA: 300 città
        if active_cities_count <= 300:
            # Per istanze piccole/medie: Algoritmo Genetico (Massima Qualità)
            path = solve_genetic(p)
        else:
            # Per istanze enormi: Greedy Multi-start + ILS (Velocità)
            path = solve_ils(p)


    # 3. Safety Check & Baseline
    # Calcoliamo il costo della nostra soluzione
    my_cost = _calculate_exact_cost(p, path)

    # Ricostruiamo la Baseline (Viaggi singoli A/R solo verso città con oro)
    base_path = []
    target_cities = [c for c in range(n_cities) if c != 0 and p._gold_cache[c] > 0]
    
    for c in target_cities:
        base_path.extend([(c, p._gold_cache[c]), (0, 0)])

    base_cost = _calculate_exact_cost(p, base_path)

    # Se la nostra soluzione è peggiore della baseline, torniamo la baseline
    if my_cost > base_cost:
        return base_path

    return path

# ==============================================================================
# STRATEGIA 1: GENETIC ALGORITHM (N <= 300, Beta <= 1)
# ==============================================================================

def solve_genetic(p: Problem):
    """
    GA Memetico per istanze trattabili.
    """
    cities = [n for n in range(len(p._nodes_list)) if n != 0 and p._gold_cache[n] > 0]
    num_cities = len(cities)
    
    # --- Tuning Parametri ---
    if num_cities < 50:
        population_size = 150
        generations = 150
        elite_size = 20
        mutation_rate = 0.30 
    elif num_cities < 150:
        population_size = 120 
        generations = 300
        elite_size = 15
        mutation_rate = 0.45 
    elif num_cities <= 300:
        population_size = 90
        generations = 600
        elite_size = 10
        mutation_rate = 0.60
    else:
        # Fallback (non dovrebbe accadere grazie al dispatcher)
        population_size = 60
        generations = 800
        elite_size = 5
        mutation_rate = 0.70

    population = []
    
    # 1. Inizializzazione Ibrida (Greedy + Random)
    num_greedy = int(population_size * 0.4) 
    for _ in range(num_greedy):
        start_node = random.choice(cities)
        # Usa la funzione unificata _nearest_neighbor_from
        population.append(_nearest_neighbor_from(p, cities, start_node))

    while len(population) < population_size:
        ind = list(cities)
        random.shuffle(ind)
        population.append(ind)
        
    # Valutazione
    fitnesses = [_eval_chrom(p, ind) for ind in population]
    
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
        fitnesses = [_eval_chrom(p, ind) for ind in population]
        
        curr_best_idx = int(np.argmin(fitnesses))
        if fitnesses[curr_best_idx] < best_cost:
            best_cost = fitnesses[curr_best_idx]
            best_solution = population[curr_best_idx]
    
    # Raffinamento finale
    best_solution = _refine_solution_final(p, best_solution)
            
    return _build_path_ga(p, best_solution)


# ==============================================================================
# STRATEGIA 2: ILS - ITERATED LOCAL SEARCH (N > 300, Beta <= 1)
# ==============================================================================

def solve_ils(p: Problem):
    """
    Greedy + ILS per istanze grandi.
    """
    cities = [n for n in range(len(p._nodes_list)) if n != 0 and p._gold_cache[n] > 0]
    
    if not cities:
        return [(0,0)]
        
    n_cities = len(cities)

    # 1. Multi-start Greedy
    solutions = []

    # Nearest neighbor da 5 punti diversi
    num_starts = min(5, n_cities)
    for _ in range(num_starts):
        start = random.choice(cities)
        tour = _nearest_neighbor_from(p, cities, start)
        solutions.append(tour)

    # Savings heuristic
    tour = _savings_simple(p, cities)
    solutions.append(tour)

    # Farthest insertion
    tour = _farthest_insertion_simple(p, cities)
    solutions.append(tour)

    # Prendi il migliore greedy
    best_tour = min(solutions, key=lambda t: _eval_tour_fast(p, t))

    # 2. ILS: Iterated Local Search
    iters = 50 if n_cities > 800 else 80
    best_tour = _iterated_local_search_simple(p, best_tour, max_iter=iters)

    # 3. Check Inversione (Forward vs Backward)
    path_fw = _build_path_simple_clustering(p, best_tour)
    path_bw = _build_path_simple_clustering(p, best_tour[::-1])
    
    cost_fw = _calculate_exact_cost(p, path_fw)
    cost_bw = _calculate_exact_cost(p, path_bw)
    
    return path_fw if cost_fw < cost_bw else path_bw


# ==============================================================================
# STRATEGIA 3: MERGE - HUB & SPOKE (Beta > 1)
# ==============================================================================

def solve_merge(p: Problem):
    """
    Hub & Spoke con Merge Greedy.
    """
    cities = [n for n in range(len(p._nodes_list)) if n != 0 and p._gold_cache[n] > 0]
    
    raw_trips = []

    # 1. Calcolo Split Ottimali
    for city in cities:
        n_splits, gold_per_trip = _optimal_splits_for_city(p, city)
        for _ in range(n_splits):
            raw_trips.append({'city': city, 'gold': gold_per_trip})

    # 2. Ordina per distanza (Nearest First)
    raw_trips.sort(key=lambda x: p._mat_dist[0][x['city']])

    # 3. Merge Greedy
    final_path = []
    if not raw_trips: return [(0,0)]

    curr_trip = raw_trips[0]
    accumulated_gold = curr_trip['gold']
    current_city = curr_trip['city']
    
    for i in range(1, len(raw_trips)):
        next_trip = raw_trips[i]
        
        # Merge solo stessa città per sicurezza con Beta alto
        if next_trip['city'] == current_city:
            w1 = accumulated_gold
            w2 = next_trip['gold']
            
            dist_out = p._mat_dist[0][current_city]
            dist_in = p._mat_dist[current_city][0] 
            
            cost_separate = _get_cost_trip_simple(p, dist_out, dist_in, w1) + \
                            _get_cost_trip_simple(p, dist_out, dist_in, w2)
            cost_merged = _get_cost_trip_simple(p, dist_out, dist_in, w1 + w2)
            
            if cost_merged < cost_separate:
                accumulated_gold += w2
            else:
                final_path.append((current_city, accumulated_gold))
                final_path.append((0, 0))
                current_city = next_trip['city']
                accumulated_gold = next_trip['gold']
        else:
            final_path.append((current_city, accumulated_gold))
            final_path.append((0, 0))
            current_city = next_trip['city']
            accumulated_gold = next_trip['gold']

    # Chiudi l'ultimo
    final_path.append((current_city, accumulated_gold))
    final_path.append((0, 0))
    
    return final_path


# ==============================================================================
# HELPER FUNCTIONS: CORE & INIT
# ==============================================================================

def _precompute_matrices(p: Problem): 
    nodes = list(p.graph.nodes) # Usa p.graph.nodes invece di p._graph se possibile, ma Problem espone _graph
    n = len(nodes)
    p._nodes_list = nodes
    p._gold_cache = [p.graph.nodes[i]['gold'] for i in range(n)]
    
    p._mat_dist = [[0.0] * n for _ in range(n)]
    p._mat_beta = [[0.0] * n for _ in range(n)]
    
    all_paths = dict(nx.all_pairs_dijkstra_path(p.graph, weight='dist'))
    
    for u in range(n):
        if u not in all_paths: continue
        for v, path in all_paths[u].items():
            d_sum = 0.0
            d_beta_sum = 0.0
            for i in range(len(path) - 1):
                d = p.graph[path[i]][path[i+1]]['dist']
                d_sum += d
                d_beta_sum += d ** p.beta
            p._mat_dist[u][v] = d_sum
            p._mat_beta[u][v] = d_beta_sum

    p._neighbors = []
    for u in range(n):
        sorted_indices = np.argsort(p._mat_dist[u])
        p._neighbors.append(sorted_indices[1:51].tolist())

    p._matrix_init_done = True

def _get_cost_matrix(p, u, v, w):
    if u == v: return 0.0
    d = p._mat_dist[u][v]
    return d if w == 0 else d + ((p.alpha * w) ** p.beta) * p._mat_beta[u][v]

def _calculate_exact_cost(p, path):
    tot, curr, w = 0.0, 0, 0.0
    for node, gold in path:
        tot += _get_cost_matrix(p, curr, node, w)
        w = 0 if node == 0 else w + gold
        curr = node
    return tot

# ==============================================================================
# HELPER FUNCTIONS: ILS & GREEDY
# ==============================================================================

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
        best_pos = len(tour) # Semplificato per velocità
        tour.insert(best_pos, farthest)
        remaining.remove(farthest)
    return tour

def _eval_tour_fast(p, tour):
    if len(tour) == 0: return 0.0
    cost = 0.0
    curr = 0
    weight = 0.0
    
    if p.beta < 0.5: cluster_size = 10
    elif p.beta < 0.8: cluster_size = 5
    else: cluster_size = 3

    for i, city in enumerate(tour):
        gold = p._gold_cache[city]
        cost += _get_cost_matrix(p, curr, city, weight)
        weight += gold
        curr = city
        
        if (i + 1) % cluster_size == 0 or i == len(tour) - 1:
            cost += _get_cost_matrix(p, curr, 0, weight)
            curr = 0
            weight = 0.0
    return cost

def _iterated_local_search_simple(p, tour, max_iter=80):
    best_tour = tour[:]
    best_cost = _eval_tour_fast(p, best_tour)
    current_tour = tour[:]

    for iteration in range(max_iter):
        current_tour = _local_search_2opt_windowed(p, current_tour)
        current_cost = _eval_tour_fast(p, current_tour)

        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = current_tour[:]

        if iteration < max_iter - 1:
            strength = 2 if iteration < max_iter // 2 else 3
            current_tour = _perturb_double_bridge(best_tour, strength)

    return best_tour

def _local_search_2opt_windowed(p, tour):
    improved = True
    best_tour = tour[:]
    best_cost = _eval_tour_fast(p, best_tour)
    n = len(best_tour)
    window_size = min(30, n // 2)
    max_passes = 5
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            j_max = min(n, i + window_size)
            for j in range(i + 2, j_max):
                new_tour = best_tour[:i] + best_tour[i:j][::-1] + best_tour[j:]
                new_cost = _eval_tour_fast(p, new_tour)
                if new_cost < best_cost:
                    best_tour = new_tour
                    best_cost = new_cost
                    improved = True
                    break
            if improved: break
    return best_tour

def _perturb_double_bridge(tour, strength=2):
    n = len(tour)
    if n < 8:
        new_tour = tour[:]
        for _ in range(strength):
            i, j = random.sample(range(n), 2)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour

    cuts = sorted(random.sample(range(1, n), 3))
    a, b, c = cuts
    if strength == 2:
        return tour[:a] + tour[b:c] + tour[a:b] + tour[c:]
    else:
        return tour[:a] + tour[c:] + tour[a:b] + tour[b:c]

def _build_path_simple_clustering(p, tour):
    path = []
    curr = 0
    weight = 0.0
    
    if p.beta < 0.5: cluster_size = 10
    elif p.beta < 0.7: cluster_size = 6
    elif p.beta < 0.9: cluster_size = 4
    else: cluster_size = 2

    for i, city in enumerate(tour):
        gold = p._gold_cache[city]
        path.append((city, gold))
        weight += gold
        curr = city
        if (i + 1) % cluster_size == 0 or i == len(tour) - 1:
            path.append((0, 0))
            curr = 0
            weight = 0.0
    return path

# ==============================================================================
# HELPER FUNCTIONS: GENETIC ALGORITHM
# ==============================================================================

def _eval_chrom(p, chrom):
    # Ottimizzato per performance (Look-ahead integrato)
    mat_dist = p._mat_dist
    mat_beta = p._mat_beta
    golds = p._gold_cache
    beta = p.beta
    alpha_pow = p.alpha ** beta 
    
    total_cost = 0.0
    curr = 0
    w = 0.0
    n_genes = len(chrom)

    for i in range(n_genes):
        nxt = chrom[i]
        fut = chrom[i+1] if i + 1 < n_genes else 0
        gold_nxt = golds[nxt]
        
        # Split Cost
        c_split = mat_dist[curr][0] + mat_dist[0][nxt]
        if w > 0: c_split += (w ** beta) * alpha_pow * mat_beta[curr][0]
        c_fut_split = mat_dist[nxt][fut]
        if gold_nxt > 0: c_fut_split += (gold_nxt ** beta) * alpha_pow * mat_beta[nxt][fut]
        score_split = c_split + c_fut_split

        # Direct Cost
        c_direct = mat_dist[curr][nxt]
        if w > 0: c_direct += (w ** beta) * alpha_pow * mat_beta[curr][nxt]
        w_new = w + gold_nxt
        c_fut_direct = mat_dist[nxt][fut]
        if w_new > 0: c_fut_direct += (w_new ** beta) * alpha_pow * mat_beta[nxt][fut]
        score_direct = c_direct + c_fut_direct

        if score_split < score_direct:
            total_cost += c_split
            w = gold_nxt
        else:
            total_cost += c_direct
            w = w_new
        curr = nxt
        
    total_cost += mat_dist[curr][0]
    if w > 0: total_cost += (w ** beta) * alpha_pow * mat_beta[curr][0]
    return total_cost

def _build_path_ga(p, chrom):
    mat_dist = p._mat_dist
    mat_beta = p._mat_beta
    golds = p._gold_cache
    beta = p.beta
    alpha_pow = p.alpha ** beta 

    path = []
    curr = 0
    w = 0.0
    n_genes = len(chrom)

    for i in range(n_genes):
        nxt = chrom[i]
        fut = chrom[i+1] if i + 1 < n_genes else 0
        gold_nxt = golds[nxt]
        
        c_split = mat_dist[curr][0] + mat_dist[0][nxt]
        if w > 0: c_split += (w ** beta) * alpha_pow * mat_beta[curr][0]
        c_fut_s = mat_dist[nxt][fut]
        if gold_nxt > 0: c_fut_s += (gold_nxt ** beta) * alpha_pow * mat_beta[nxt][fut]
        score_split = c_split + c_fut_s

        c_direct = mat_dist[curr][nxt]
        if w > 0: c_direct += (w ** beta) * alpha_pow * mat_beta[curr][nxt]
        w_new = w + gold_nxt
        c_fut_d = mat_dist[nxt][fut]
        if w_new > 0: c_fut_d += (w_new ** beta) * alpha_pow * mat_beta[nxt][fut]
        score_direct = c_direct + c_fut_d

        if score_split < score_direct:
            if curr != 0: path.append((0, 0))
            path.append((nxt, gold_nxt))
            w = gold_nxt
        else:
            path.append((nxt, gold_nxt))
            w = w_new
        curr = nxt
        
    path.append((0, 0))
    return path

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
    best_cost = _eval_chrom(p, best_sol)
    n = len(best_sol)
    
    if quick:
        num_iter_2opt = 20
        num_iter_insert = 15
        max_loops = 1 
        neighbors_to_check = 6
    else:
        num_iter_2opt = max(50, min(int(n * 0.5), 300))
        num_iter_insert = max(50, min(int(n * 0.4), 300))
        max_loops = 3 if n > 500 else 4
        neighbors_to_check = 20

    improved = True
    loop_count = 0
    
    while improved and loop_count < max_loops:
        improved = False
        loop_count += 1
        
        # 2-Opt
        for _ in range(num_iter_2opt): 
            i, j = sorted(random.sample(range(n), 2))
            if j - i < 2: continue
            if quick and (j - i) > (n / 4): continue
            new_sol = best_sol[:i] + best_sol[i:j+1][::-1] + best_sol[j+1:]
            new_cost = _eval_chrom(p, new_sol)
            if new_cost < best_cost:
                best_sol = new_sol
                best_cost = new_cost
                improved = True
                if quick: break
        
        if improved and quick: break
        if improved: continue

        # Guided Insert
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

            candidate_positions.update(random.sample(range(len(temp_sol)+1), 1 if quick else 2))
            
            found_better = False
            for pos in candidate_positions:
                if pos > len(temp_sol): pos = len(temp_sol)
                cand = temp_sol[:pos] + [city] + temp_sol[pos:]
                c = _eval_chrom(p, cand)
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
    best_c = _eval_chrom(p, best_s)
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
                c = _eval_chrom(p, new_s)
                if c < best_c:
                    best_c = c
                    best_s = new_s
                    improved = True
                    break 
            if improved: break
    return best_s

# ==============================================================================
# HELPER FUNCTIONS: MERGE
# ==============================================================================

def _optimal_splits_for_city(p, city_id):
    total_gold = p._gold_cache[city_id]
    dist = p._mat_dist[0][city_id]

    if dist == 0 or total_gold == 0: return 1, total_gold

    best_splits = 1
    best_cost = float('inf')
    max_splits = min(15, int(total_gold) + 1)

    for k in range(1, max_splits + 1):
        g = total_gold / k
        trip_cost = _get_cost_trip_simple(p, dist, dist, g) 
        total_k_cost = k * trip_cost
        if total_k_cost < best_cost:
            best_cost = total_k_cost
            best_splits = k
    return best_splits, total_gold / best_splits

def _get_cost_trip_simple(p, d_out, d_in, w):
    cost_out = d_out
    cost_in = d_in + ((p.alpha * w) ** p.beta) * d_in
    return cost_out + cost_in


# ==============================================================================
# MAIN TEST - BENCHMARK COMPARATIVO (Ordine Esatto Tabella Prof)
# ==============================================================================

'''
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # Dati copiati rigorosamente in ordine dall'immagine fornita
    benchmarks = [
        # Riga 1: Beta=2 (High Cost) -> OK
        #{"N": 100,  "d": 0.2, "a": 1.0, "b": 2.0, "raco_ref": 944485.81},
        
        # Riga 2: Beta=1 (Low Cost) -> OK
        #{"N": 100,  "d": 1.0, "a": 1.0, "b": 1.0, "raco_ref": 3847.92},
        
        # Riga 3 [FIXED]: Beta=2 DEVE avere il target alto (415k), non quello basso
        # Parametri tabella: a=1, b=2. Valore logico associato: quello della riga 4 originale
        #{"N": 100,  "d": 1.0, "a": 1.0, "b": 2.0, "raco_ref": 415878.82},
        
        # Riga 4 [FIXED]: Beta=1 DEVE avere il target basso (7.5k)
        # Parametri tabella: a=2, b=1. Valore logico associato: quello della riga 3 originale
        #{"N": 100,  "d": 1.0, "a": 2.0, "b": 1.0, "raco_ref": 7549.20},
        
        # Le righe N=1000 sembrano coerenti (Beta=2 ha valori alti), le lascio originali
        {"N": 1000, "d": 0.2, "a": 1.0, "b": 1.0, "raco_ref": 32769.39},
        {"N": 1000, "d": 0.2, "a": 1.0, "b": 2.0, "raco_ref": 69177.09},
        {"N": 1000, "d": 0.2, "a": 2.0, "b": 2.0, "raco_ref": 12068962.33},
        {"N": 1000, "d": 1.0, "a": 1.0, "b": 1.0, "raco_ref": 17142.17},
        {"N": 1000, "d": 1.0, "a": 2.0, "b": 1.0, "raco_ref": 34040.83},
        {"N": 1000, "d": 1.0, "a": 1.0, "b": 2.0, "raco_ref": 7896879.42},
    ]

    print(f"{'PARAMS (N, d, a, b)':<22} | {'BASELINE':<15} | {'TARGET (RACO)':<15} | {'MIO COSTO':<15} | {'VS RACO %':<10} | {'TIME (s)':<9} | {'ESITO':<8}")
    print("-" * 120)

    # Seed standardizzato
    SEED = 42

    for i, bench in enumerate(benchmarks):
        n, d, a, b = bench["N"], bench["d"], bench["a"], bench["b"]
        raco_target = bench["raco_ref"]

        try:
            # 1. Init Problema
            p = Problem(num_cities=n, density=d, alpha=a, beta=b, seed=SEED)
            
            # 2. Calcolo Baseline
            base_cost = p.baseline()
            
            # 3. Esecuzione Tua Soluzione con Timer
            start_t = time.time()
            my_path = solution(p)
            end_t = time.time()
            elapsed = end_t - start_t
            
            # 4. Verifica Costo
            my_cost = check_solution_cost(p, my_path)
            
            # 5. Calcolo Metriche
            diff_raco = my_cost - raco_target
            perc_raco = (diff_raco / raco_target) * 100
            
            if my_cost < raco_target:
                outcome = "🏆 WIN"
            elif my_cost < base_cost:
                outcome = "✅ OK"
            else:
                outcome = "❌ FAIL"

            params_str = f"{n}, {d}, {a}, {b}"
            
            # Output formattato
            print(f"{params_str:<22} | {base_cost:,.2f}".ljust(40) + 
                  f"| {raco_target:,.2f}".ljust(18) + 
                  f"| {my_cost:,.2f}".ljust(18) + 
                  f"| {perc_raco:+.2f}%".ljust(13) + 
                  f"| {elapsed:<9.4f} " +  
                  f"| {outcome}")

        except Exception as e:
            print(f"{n}, {d}, {a}, {b}        | ERRORE: {e}")

    print("-" * 120)

'''


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # Configurazioni di Test
    random.seed(42)


    #NUM_CITIES = [100, 200, 500, 1000] # 500 

    #BETAS = [0.3 , 0.6, 0.9]
    #ALPHAS = [1.0, 2.0]
    #DENSITIES = [0.3, 1.0] # Densità fissa per ora, poi fai 0.8 anche  


    # confronto RICCARDO:
    NUM_CITIES = [100, 500, 1000] # 300, 500, 1000] 
    DENSITIES = [0.2, 1] 
    BETAS = [0.5, 2]   # 1 al momenot non serve -- TODO : metti poi il caso speciale con b=1
    ALPHAS = [1.0]


    #BETAS = [0.5, 1.0, 1.5, 2.0, 4.0]

    
    print("-" * 130) 
    print("--- TEST FINALE  ( ) ---")
    print("-" * 130)
    print(f"{'N':<4} | {'Alp':<4} | {'Bet':<4} | {'Den':<4} | {'Baseline':<12} | {'Mio Costo':<12} | {'Delta':<12} | {'Delta %':<8} | {'Time(s)':<8}")    
    print("-" * 130)    
    
    for n in NUM_CITIES:
        for a in ALPHAS:       
            for b in BETAS:
                for d in DENSITIES:
                    start_time = time.time()
                    
                    p = Problem(num_cities=n, density=d, alpha=a, beta=b, seed=42)
                    
                    # 2. Esecuzione Soluzione
                    base = p.baseline()
                    sol = solution(p)

                    # check per essere certi del risultato
                    cost = check_solution_cost(p, sol)
                    
                    delta = base - cost
                    elapsed_time = time.time() - start_time
                    
                    if base > 0:
                        delta_perc = (delta / base) * 100
                    else:
                        delta_perc = 0.0
                    
                    # Formattazione stringhe
                    delta_str = f"{delta:.2f}"
                    perc_str = f"{delta_perc:+.2f}%" # Il '+' aggiunge il segno anche se positivo
                    
                    print(f"{n:<4} | {a:<4.1f} | {b:<4.1f} | {d:<4.1f} | {base:<12.2f} | {cost:<12.2f} | {delta_str:<12} | {perc_str:<8} | {elapsed_time:<8.4f}")
    
    print("-" * 145)

