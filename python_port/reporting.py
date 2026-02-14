from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np


def calc_agap(
    nw: Dict[str, object],
    results: np.ndarray,
    real_path_cost: np.ndarray,
    real_link_cost: np.ndarray,
    k_shortest_paths_func: Callable[[np.ndarray, int, int, int], tuple[List[List[int]], List[float]]],
) -> Dict[str, float]:
    no_class = int(nw["no_class"])
    no_od = int(nw["no_OD"])
    path_vector = np.asarray(nw["path_vector"], dtype=int)

    total_miss_assign_original = 0.0
    for m in range(no_class):
        for w in range(no_od):
            start = int(np.sum(path_vector[:w]))
            end = start + int(path_vector[w])
            c_min = float(np.min(results[start:end, 2 * no_class + m]))
            for p in range(path_vector[w]):
                idx = start + p
                total_miss_assign_original += (
                    results[idx, m]
                    * float(nw["pce"][m])
                    * (results[idx, 2 * no_class + m] - c_min)
                )

    total_miss_assign_real_cost = 0.0
    for m in range(no_class):
        for w in range(no_od):
            start = int(np.sum(path_vector[:w]))
            end = start + int(path_vector[w])
            c_min = float(np.min(real_path_cost[start:end, m]))
            for p in range(path_vector[w]):
                idx = start + p
                total_miss_assign_real_cost += (
                    results[idx, m] * float(nw["pce"][m]) * (real_path_cost[idx, m] - c_min)
                )

    no_node = int(nw["no_node"])
    no_link = int(nw["no_link"])
    network = nw["network"]
    link_costff_postassignment = np.full((no_node, no_node), np.inf)
    for i in range(no_link):
        u = int(network["fromNode"][i]) - 1
        v = int(network["toNode"][i]) - 1
        link_costff_postassignment[u, v] = float(real_link_cost[i, 0])

    demand = nw["demand"]
    no_dem = len(demand["fromNode"])
    shortest_paths_post: List[List[int]] = []
    for w in range(no_dem):
        paths, _ = k_shortest_paths_func(
            link_costff_postassignment, int(demand["fromNode"][w]), int(demand["toNode"][w]), 1
        )
        shortest_paths_post.append(paths[0] if paths else [])

    shortest_paths_post_link: List[List[int]] = []
    for w in range(no_dem):
        links_for_path: List[int] = []
        path = shortest_paths_post[w]
        for n in range(max(len(path) - 1, 0)):
            u = path[n]
            v = path[n + 1]
            for idx in range(no_link):
                if int(network["fromNode"][idx]) == u and int(network["toNode"][idx]) == v:
                    links_for_path.append(idx)
                    break
        shortest_paths_post_link.append(links_for_path)

    shortest_path_cost = np.zeros((int(nw["no_od_on"]), no_class), dtype=float)
    for m in range(no_class):
        for w in range(int(nw["no_od_on"])):
            for e in shortest_paths_post_link[w]:
                shortest_path_cost[w, m] += real_link_cost[e, m]

    total_miss_schedule = 0.0
    for m in range(no_class):
        for w in range(no_od):
            start = int(np.sum(path_vector[:w]))
            for k in range(path_vector[w]):
                idx = start + k
                total_miss_schedule += (
                    (real_path_cost[idx, m] * float(nw["pce"][m]) - shortest_path_cost[w, m] * float(nw["pce"][m]))
                    * results[idx, m]
                )

    total_flow = 0.0
    for m in range(no_class):
        total_flow += float(np.sum(results[:, m]) * float(nw["pce"][m]))
    total_flow = max(total_flow, 1e-12)

    return {
        "agap": float(total_miss_schedule / total_flow),
        "agap_ori": float(total_miss_assign_original / total_flow),
        "agap_real": float(total_miss_assign_real_cost / total_flow),
    }

