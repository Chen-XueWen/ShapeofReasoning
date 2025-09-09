from collections import defaultdict, deque
import heapq
import math
import numpy as np
from sklearn.cluster import KMeans


def build_graph(X: np.ndarray, k: int = 200):
    k = min(k, X.shape[0])
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    path = [int(label) for label in labels]
    distances = [float(np.linalg.norm(centers[i] - centers[i-1])) for i in path[1:]]
    return path, distances


def analyze_graph_simple(path, distances):
    adj = defaultdict(list)
    for u, v, w in zip(path, path[1:], distances):
        if u != v:
            adj[u].append((v, w))
    # Cycle detection
    seen, has_loop = set(), False
    loop_count = 0
    entry_node = None
    for i, node in enumerate(path):
        if node in seen:
            has_loop = True
            entry_node = node
            loop_count = path.count(node) - 1
            break
        seen.add(node)
    # Diameter and Avg Path Length
    def dijkstra(u):
        dist = {u: 0}
        heap = [(0, u)]
        while heap:
            d, node = heapq.heappop(heap)
            for neighbor, weight in adj[node]:
                new_dist = d + weight
                if neighbor not in dist or new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))
        return dist
    all_distances = [dijkstra(node) for node in list(adj.keys())]
    diameter = max((max(distances.values()) for distances in all_distances), default=0)
    avg_path_length = \
        sum(sum(distances.values()) for distances in all_distances) / sum(len(distances) - 1 for distances in all_distances)
    # Clustering Coefficient
    undirected = defaultdict(set)
    for u, neighbors in adj.items():
        for v, _ in neighbors:
            undirected[u].add(v)
            undirected[v].add(u)
    clustering_sum, count_cc = 0, 0
    for node, nbrs in undirected.items():
        if len(nbrs) < 2:
            continue
        actual_edges = sum(1 for v in nbrs for w in nbrs if v < w and w in undirected[v])
        clustering_sum += actual_edges / (len(nbrs) * (len(nbrs) - 1) / 2)
        count_cc += 1
    avg_clustering = clustering_sum / count_cc if count_cc else 0
    # Small-World IIndex
    N = len(undirected)
    K = sum(len(nbrs) for nbrs in undirected.values()) / N if N else 0
    C_rand = K / (N - 1) if N > 1 else 0
    L_rand = math.log(N) / math.log(K) if N > 1 and K > 1 else float('inf')
    clustering_norm = avg_clustering / C_rand if C_rand else 0
    path_length_norm = avg_path_length / L_rand if L_rand else 0
    small_world_index = clustering_norm / path_length_norm if path_length_norm else 0
    return has_loop, loop_count, diameter, avg_clustering, avg_path_length, small_world_index