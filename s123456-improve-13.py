
import logging
import random
import time
import networkx as nx
import numpy as np
from Problem import Problem


# ==============================================================================
# DISPATCHER: MAIN ENTRY POINT -- CC  con _calculate_route_cost_atomic -- sembra piu lento #2
# ==============================================================================


def solution(p: Problem):
    # 1. Init Matrici (Lazy Loading) - Eseguito una volta sola
    if not hasattr(p, '_matrix_init_done'):
        _precompute_matrices(p)

    n_cities = len(p._nodes_list)
    
    # 2. Selezione Strategia
    if p.beta > 1.0:
        # --- CASO SCARICO (Beta Alto) ---
        path = solve_merge(p)     
    elif p.beta == 1.0:
        # --- CASO SPECIALE BETA=1 (Lineare) --- TODO ... 
        path = solve_beta_one(p)
    else:
        # --- CASO ACCUMULO (Beta Basso) ---
        active_cities_count = len([c for c in range(n_cities) if c != 0 and p._gold_cache[c] > 0])
        
        # SOGLIA CRITICA: 300 città
        if active_cities_count <= 300:
            # Per istanze piccole/medie: Algoritmo Genetico (Massima Qualità)
            path = solve_genetic(p)
        else:
            # Per istanze enormi: ILS con DP Splitting (Qualità + Velocità)
            path = solve_ils(p)

    # 3. Safety Check & Baseline
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
# STRATEGIA 1: GENETIC ALGORITHM (N <= 300, Beta < 1)
# ==============================================================================


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


# ==============================================================================
# STRATEGIA 2: ILS - ITERATED LOCAL SEARCH (N > 300, Beta < 1)
# ==============================================================================


def solve_ils(p: Problem):
    """
    ILS Geometrico + DP Splitting Ottimale.
    1. Ottimizza la geometria (distanza pura) velocemente.
    2. Usa DP per trovare split ottimali basati su peso.
    """
    cities = [n for n in range(len(p._nodes_list)) if n != 0]
    if not cities: return [(0,0)]
    
    # 1. Inizializzazione Multi-Start (Geometry-Aware)
    best_tour = []
    best_dist = float('inf')
    
    # Proviamo 3 partenze diverse
    starts = [random.choice(cities) for _ in range(2)]
    
    # Aggiungi un tentativo deterministico (più lontano dalla base)
    farthest_city = max(cities, key=lambda c: p._mat_dist[0][c])
    starts.append(farthest_city)

    for start_node in starts:
        tour = _nearest_neighbor_from(p, cities, start_node)
        d = _calc_pure_distance(p, tour)
        if d < best_dist:
            best_dist = d
            best_tour = tour

    # 2. ILS Loop (Geometry First)
    max_iter = 60 
    
    current_tour = best_tour[:]
    current_dist = best_dist
    
    for iteration in range(max_iter):
        # A. Local Search (2-Opt Fast con Delta Eval)
        current_tour, current_dist = _local_search_2opt_fast_dist(p, current_tour, current_dist)

        # B. Aggiorna Best
        if current_dist < best_dist:
            best_dist = current_dist
            best_tour = current_tour[:]

        # C. Perturbazione (Double Bridge)
        if iteration < max_iter - 1:
            strength = 2 if iteration < max_iter // 2 else 3
            current_tour = _perturb_double_bridge(best_tour, strength)
            current_dist = _calc_pure_distance(p, current_tour)

    # 3. NUOVO: DP Splitting Ottimale (invece di clustering euristico)
    path_fw = _split_tour_optimal_dp(p, best_tour)
    path_bw = _split_tour_optimal_dp(p, best_tour[::-1])
    
    cost_fw = _calculate_exact_cost(p, path_fw)
    cost_bw = _calculate_exact_cost(p, path_bw)
    
    return path_fw if cost_fw < cost_bw else path_bw


# ==============================================================================
# STRATEGIA 3: MERGE - HUB & SPOKE (Beta > 1)
# ==============================================================================


def solve_merge(p: Problem):
    """
    Advanced Hub & Spoke con Merge Cross-Città.
    1. Genera viaggi atomici ottimizzati (Split).
    2. Multi-pass greedy merge con controllo peso massimo.
    3. Unisce città vicine per ridurre viaggi a vuoto.
    """
    cities = [n for n in range(len(p._nodes_list)) if n != 0 and p._gold_cache[n] > 0]
    
    # --- FASE 1: Creazione Viaggi Atomici ---
    routes = []
    for city in cities:
        n_splits, gold_per_trip = _optimal_splits_for_city(p, city)
        for _ in range(n_splits):
            routes.append([(city, gold_per_trip)])
            
    if not routes: return [(0,0)]

    # Parametri adattativi
    if p.beta > 3.0:
        max_weight_per_trip = sum(p._gold_cache) * 0.05
        max_passes = 2
    elif p.beta > 2.0:
        max_weight_per_trip = sum(p._gold_cache) * 0.15
        max_passes = 3
    else:
        max_weight_per_trip = float('inf')
        max_passes = 3
    
    # Riduci pass per N grande
    if len(cities) > 500:
        max_passes = min(max_passes, 2)

    # --- FASE 2: Greedy Merge (Multi-Pass) ---
    improved = True
    
    for pass_num in range(max_passes):
        if not improved: break
        improved = False
        
        # Ordina per distanza decrescente dalla base
        routes.sort(key=lambda r: p._mat_dist[0][r[0][0]], reverse=True)
        
        i = 0
        while i < len(routes):
            route_A = routes[i]
            last_city_A = route_A[-1][0]
            weight_A = sum(gold for _, gold in route_A)
            
            best_j = -1
            best_saving = 0.0
            
            # Neighbor set per pre-filtro geometrico
            neighbors_set = set(p._neighbors[last_city_A][:20]) 
            
            # Trova candidati ordinati per distanza
            all_candidates = []
            for j in range(len(routes)):
                if j == i: continue
                start_city_B = routes[j][0][0]
                dist_AB = p._mat_dist[last_city_A][start_city_B]
                all_candidates.append((j, dist_AB, start_city_B))
            
            # Ordina e prendi i K più vicini
            all_candidates.sort(key=lambda x: x[1])
            K = min(50, len(all_candidates))
            
            for j, dist_AB, start_city_B in all_candidates[:K]:
                route_B = routes[j]
                weight_B = sum(gold for _, gold in route_B)
                merged_weight = weight_A + weight_B
                
                # Pre-filtro: Salta se supera peso massimo
                if merged_weight > max_weight_per_trip:
                    continue
                
                # Pre-filtro geometrico (opzionale, già ordinato per distanza)
                if start_city_B not in neighbors_set and dist_AB > p._mat_dist[0][last_city_A] * 0.5:
                    continue  # Troppo lontano

                # Calcola saving effettivo
                cost_A = _calculate_route_cost_atomic(p, route_A)
                cost_B = _calculate_route_cost_atomic(p, route_B)
                
                merged_route = route_A + route_B
                cost_merged = _calculate_route_cost_atomic(p, merged_route)
                
                saving = (cost_A + cost_B) - cost_merged
                
                if saving > 0.001 and saving > best_saving:
                    best_saving = saving
                    best_j = j
            
            if best_j != -1:
                routes[i] = routes[i] + routes[best_j]
                routes.pop(best_j)
                improved = True
                if best_j < i: i -= 1
                continue 
            
            i += 1

    # --- FASE 3: Output ---
    final_path = []
    for route in routes:
        for city, gold in route:
            final_path.append((city, gold))
        final_path.append((0, 0))
        
    return final_path


def _calculate_route_cost_atomic(p, route):
    """
    Calcola il costo esatto di un viaggio A/R che visita le città in 'route'.
    """
    cost = 0.0
    curr = 0 
    w = 0.0  
    beta = p.beta
    alpha = p.alpha
    
    # Andata: visita tutte le città in sequenza
    for city, gold in route:
        dist = p._mat_dist[curr][city]
        if w == 0: 
            cost += dist
        else: 
            cost += dist + ((alpha * w) ** beta) * p._mat_beta[curr][city]
        w += gold
        curr = city
        
    # Ritorno alla base
    dist_home = p._mat_dist[curr][0]
    if w == 0: 
        cost += dist_home
    else: 
        cost += dist_home + ((alpha * w) ** beta) * p._mat_beta[curr][0]
        
    return cost



# ==============================================================================
# STRATEGIA 4: BETA=1 (Caso Speciale Lineare)
# ==============================================================================


def solve_beta_one(p: Problem):
    """
    Per Beta=1, il costo è LINEARE: d * (1 + alpha * W).
    Strategia ibrida: Tour geometrico + DP per split ottimali.
    """
    cities = [n for n in range(len(p._nodes_list)) if n != 0]
    if not cities: return [(0,0)]
    
    n_cities = len(cities)
    
    # Per N piccolo, usa GA (alta qualità)
    if n_cities <= 300:
        # Riusa solve_genetic (già ottimizzato)
        return solve_genetic(p)
    else:
        # Per N grande, usa ILS + DP
        return solve_ils(p)


# ==============================================================================
# HELPER FUNCTIONS: CORE & INIT
# ==============================================================================


def _precompute_matrices(p: Problem): 
    nodes = list(p._graph.nodes)
    n = len(nodes)
    p._nodes_list = nodes
    p._gold_cache = [p._graph.nodes[i]['gold'] for i in range(n)]
    
    p._mat_dist = [[0.0] * n for _ in range(n)]
    p._mat_beta = [[0.0] * n for _ in range(n)]
    
    # Calcolo Dijkstra sul grafo originale
    all_paths = dict(nx.all_pairs_dijkstra_path(p._graph, weight='dist'))
    
    for u in range(n):
        if u not in all_paths: continue
        for v, path in all_paths[u].items():
            d_sum = 0.0
            d_beta_sum = 0.0
            for i in range(len(path) - 1):
                d = p._graph[path[i]][path[i+1]]['dist']
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
        best_pos = len(tour)
        tour.insert(best_pos, farthest)
        remaining.remove(farthest)
    return tour


def _calc_pure_distance(p, tour):
    """Calcola la distanza pura totale del tour (senza tornare alla base)"""
    d = 0.0
    for i in range(len(tour) - 1):
        d += p._mat_dist[tour[i]][tour[i+1]]
    return d


def _local_search_2opt_fast_dist(p, tour, current_dist):
    """
    2-Opt velocissimo basato solo sulla DISTANZA.
    Usa la valutazione delta O(1).
    """
    best_tour = tour
    best_d = current_dist
    n = len(tour)
    improved = True
    
    mat = p._mat_dist
    
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                A = best_tour[i]
                B = best_tour[i+1]
                C = best_tour[j]
                D = best_tour[j+1] if j + 1 < n else None
                
                # Vecchi archi
                d_old = mat[A][B]
                if D is not None: d_old += mat[C][D]
                
                # Nuovi archi
                d_new = mat[A][C]
                if D is not None: d_new += mat[B][D]
                
                if d_new < d_old:
                    delta = d_new - d_old
                    new_segment = best_tour[i+1:j+1][::-1]
                    best_tour = best_tour[:i+1] + new_segment + best_tour[j+1:]
                    best_d += delta
                    improved = True
                    break 
            if improved: break
            
    return best_tour, best_d


def _split_tour_optimal_dp(p, tour):
    """
    NUOVO: Trova gli split ottimali con Programmazione Dinamica O(N²).
    Per ogni posizione i, prova tutti i possibili viaggi A/R che terminano lì.
    """
    n = len(tour)
    INF = float('inf')
    
    # dp[i] = costo minimo per servire tour[0:i] con viaggi A/R dalla base
    dp = [INF] * (n + 1)
    split_at = [-1] * (n + 1)
    dp[0] = 0.0
    
    # Parametro chiave: max_trip_length (dipende da beta)
    if p.beta < 0.4:
        max_trip_len = n  # Nessun limite reale
    elif p.beta < 0.7:  # Beta=0.5 cade qui
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


# ==============================================================================
# HELPER FUNCTIONS: GENETIC ALGORITHM - ENHANCED
# ==============================================================================


def _eval_chrom_enhanced(p, chrom):
    """
    NUOVO: Valutazione POTENZIATA con look-ahead multi-step.
    Con beta basso, guarda 2-3 città avanti per decisioni migliori.
    """
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


def _build_path_ga_enhanced(p, chrom):
    """
    NUOVO: Costruisce il path usando la stessa logica di _eval_chrom_enhanced.
    """
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
    best_cost = _eval_chrom_enhanced(p, best_sol)  # UPDATED: usa enhanced
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
            new_cost = _eval_chrom_enhanced(p, new_sol)  # UPDATED
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
                c = _eval_chrom_enhanced(p, cand)  # UPDATED
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
# VALIDATION FUNCTION
# ==============================================================================


def check_solution_cost(p: Problem, solution_path):
    """
    Verifica indipendente del costo totale.
    Non usa le matrici pre-calcolate, ma simula il movimento fisico
    sul grafo originale usando nx.shortest_path e la formula della traccia.
    """
    total_cost = 0.0
    current_w = 0.0
    curr_node = 0
    
    alpha = p.alpha
    beta = p.beta

    for next_node, collected_gold in solution_path:
        try:
            physical_path = nx.shortest_path(p._graph, source=curr_node, target=next_node, weight='dist')
        except nx.NetworkXNoPath:
            print(f"!!! ERRORE CRITICO: Non esiste strada tra {curr_node} e {next_node}")
            return float('inf')

        for i in range(len(physical_path) - 1):
            u = physical_path[i]
            v = physical_path[i+1]
            
            d = p._graph[u][v]['dist']
            fatigue = (alpha * d * current_w) ** beta
            
            total_cost += d + fatigue

        curr_node = next_node
        
        if curr_node == 0:
            current_w = 0.0
        else:
            current_w += collected_gold
            
    return total_cost


# ==============================================================================
# MAIN TEST
# ==============================================================================


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    random.seed(42)

    NUM_CITIES = [100, 500, 1000]
    DENSITIES = [0.2, 1] 
    BETAS = [0.5, 2]
    ALPHAS = [1.0]
    
    print("-" * 130) 
    print("--- TEST FINALE CON MIGLIORAMENTI ---")
    print("-" * 130)
    print(f"{'N':<4} | {'Alp':<4} | {'Bet':<4} | {'Den':<4} | {'Baseline':<12} | {'Mio Costo':<12} | {'Delta':<12} | {'Delta %':<8} | {'Time(s)':<8}")    
    print("-" * 130)    
    
    for n in NUM_CITIES:
        for a in ALPHAS:       
            for b in BETAS:
                for d in DENSITIES:
                    start_time = time.time()
                    
                    p = Problem(num_cities=n, density=d, alpha=a, beta=b, seed=42)
                    
                    base = p.baseline()
                    sol = solution(p)
                    cost = check_solution_cost(p, sol)
                    
                    delta = base - cost
                    elapsed_time = time.time() - start_time
                    
                    if base > 0:
                        delta_perc = (delta / base) * 100
                    else:
                        delta_perc = 0.0
                    
                    delta_str = f"{delta:.2f}"
                    perc_str = f"{delta_perc:+.2f}%"
                    
                    print(f"{n:<4} | {a:<4.1f} | {b:<4.1f} | {d:<4.1f} | {base:<12.2f} | {cost:<12.2f} | {delta_str:<12} | {perc_str:<8} | {elapsed_time:<8.4f}")
    
    print("-" * 130)

