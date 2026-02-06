#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark adaptativo para funções principais.
CLI: python bench_module.py bench <target> [args...] [--min-time T] [--max-iters N] [--clear-cache]
Targets:
  neighbors W H
  bfs W H comp_size
  compute block_mask W H [--random K] [--clear-cache]
  placements N W H
  all W H N ...
"""
import random
import sys
from collections import deque
from functools import lru_cache
from time import perf_counter

from polyominoes import ominoes_dict


# ---------------------- core utilities (from your code) ----------------------
def build_neighbors(w: int, h: int) -> list[list[int]]:
    total = w * h
    nb = [[] for _ in range(total)]
    for y in range(h):
        base = y * w
        up = (y - 1) * w if (y - 1) >= 0 else None
        down = (y + 1) * w if (y + 1) < h else None
        for x in range(w):
            idx = base + x
            if x + 1 < w:
                nb[idx].append(idx + 1)
            if x - 1 >= 0:
                nb[idx].append(idx - 1)
            if down is not None:
                nb[idx].append(down + x)
            if up is not None:
                nb[idx].append(up + x)
    return nb


def bfs_local(
    start_local: int, local_nb: list[list[int]]
) -> tuple[list[int], list[int]]:
    n = len(local_nb)
    dist = [-1] * n
    parent = [-1] * n
    dq = deque()
    dq.append(start_local)
    dist[start_local] = 0

    pop = dq.popleft
    push = dq.append

    while dq:
        u = pop()
        du = dist[u] + 1
        for v in local_nb[u]:
            if dist[v] == -1:
                dist[v] = du
                parent[v] = u
                push(v)
    return dist, parent


def reconstruct_path_local(
    parent_local: list[int], src_local: int, dest_local: int, local_to_global: list[int]
) -> list[int]:
    path_local = []
    cur = dest_local
    while cur != -1:
        path_local.append(cur)
        if cur == src_local:
            break
        cur = parent_local[cur]
    path_local.reverse()
    return [local_to_global[i] for i in path_local]


EXACT_THRESHOLD = 120
_LRU_CACHE_SIZE = 10**6
_neighbors_cache: dict[tuple[int, int], list[list[int]]] = {}


def _compute_diameter_and_path_uncached(
    block_mask: int, w: int, h: int
) -> tuple[int, int | None, int | None, tuple]:
    total = w * h
    if total == 0:
        return 0, None, None, tuple()
    full_mask = (1 << total) - 1
    empty_mask = full_mask & (~block_mask)
    if empty_mask == 0:
        return 0, None, None, tuple()
    neighbors = _neighbors_cache.get((w, h))
    if neighbors is None:
        neighbors = build_neighbors(w, h)
        _neighbors_cache[(w, h)] = neighbors
    nb = neighbors

    remaining = empty_mask
    best_diam = 0
    best_a = None
    best_b = None
    best_path: list[int] = []

    while remaining:
        lsb = remaining & -remaining
        start = lsb.bit_length() - 1

        # flood fill
        comp_nodes: list[int] = []
        q = [start]
        seen = [False] * total
        seen[start] = True
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            comp_nodes.append(u)
            for v in nb[u]:
                if (not seen[v]) and (((empty_mask >> v) & 1) != 0):
                    seen[v] = True
                    q.append(v)

        # remove comp
        comp_mask = 0
        for u in comp_nodes:
            comp_mask |= 1 << u
        remaining &= ~comp_mask

        comp_size = len(comp_nodes)
        if comp_size < 2:
            continue

        total_nodes = total
        map_global_to_local = [-1] * total_nodes
        for i, g in enumerate(comp_nodes):
            map_global_to_local[g] = i
        local_to_global = comp_nodes[:]

        local_nb: list[list[int]] = [[] for _ in range(comp_size)]
        for i, g in enumerate(comp_nodes):
            for gg in nb[g]:
                gg_local = map_global_to_local[gg]
                if gg_local != -1:
                    local_nb[i].append(gg_local)

        if comp_size <= EXACT_THRESHOLD:
            for src_local in range(comp_size):
                dist, parent = bfs_local(src_local, local_nb)
                far_local = src_local
                far_d = 0
                for v_local in range(comp_size):
                    d = dist[v_local]
                    if d > far_d:
                        far_d = d
                        far_local = v_local
                if far_d > best_diam:
                    best_diam = far_d
                    best_a = local_to_global[src_local]
                    best_b = local_to_global[far_local]
                    best_path = reconstruct_path_local(
                        parent, src_local, far_local, local_to_global
                    )
        else:
            a_local = 0
            dist_a, parent_a = bfs_local(a_local, local_nb)
            far_a_local = a_local
            max_da = 0
            for v_local in range(comp_size):
                d = dist_a[v_local]
                if d > max_da:
                    max_da = d
                    far_a_local = v_local

            dist_b, parent_b = bfs_local(far_a_local, local_nb)
            far_b_local = far_a_local
            max_db = 0
            for v_local in range(comp_size):
                d = dist_b[v_local]
                if d > max_db:
                    max_db = d
                    far_b_local = v_local

            if max_db > best_diam:
                best_diam = max_db
                best_a = local_to_global[far_a_local]
                best_b = local_to_global[far_b_local]
                best_path = reconstruct_path_local(
                    parent_b, far_a_local, far_b_local, local_to_global
                )

    return (
        best_diam + 1,
        (best_a if best_a is not None else None),
        (best_b if best_b is not None else None),
        tuple(best_path),
    )


_cached_compute = lru_cache(maxsize=_LRU_CACHE_SIZE)(
    _compute_diameter_and_path_uncached
)


def compute_diameter_and_path(
    block_mask: int, w: int, h: int
) -> tuple[int, int | None, int | None, list[int]]:
    nodes, a, b, path_t = _cached_compute(block_mask, w, h)
    return nodes, a, b, list(path_t)


def clear_caches():
    _neighbors_cache.clear()
    try:
        _cached_compute.cache_clear()
    # trunk-ignore(bandit/B110)
    except Exception:
        pass


# ---------------------- simple polyomino helpers for placements benchmark ----------
# minimal implementations used only for placements benchmark:
_ROTATIONS = (
    lambda x, y: (x, y),
    lambda x, y: (-y, x),
    lambda x, y: (-x, -y),
    lambda x, y: (y, -x),
)


def _normalize_variants(cells):
    pts = tuple(cells)
    variants = []
    for reflect in (False, True):
        for rot_fn in _ROTATIONS:
            transformed = []
            if reflect:
                for x, y in pts:
                    xr = -x
                    yr = y
                    tx, ty = rot_fn(xr, yr)
                    transformed.append((tx, ty))
            else:
                for x, y in pts:
                    tx, ty = rot_fn(x, y)
                    transformed.append((tx, ty))
            minx = min(p[0] for p in transformed)
            miny = min(p[1] for p in transformed)
            norm = tuple(sorted(((p[0] - minx, p[1] - miny) for p in transformed)))
            variants.append(norm)
    return variants


def normalize(cells):
    return min(_normalize_variants(cells))


def all_symmetries(canonical):
    """
    Gera todas as simetrias (reflexões+rotações) normalizadas de `canonical`.
    Minimiza alocações: calcula minx/miny on-the-fly e evita chamadas de função por ponto.
    """
    pts = tuple(canonical)
    syms = set()
    for reflect in (False, True):
        for rot in range(4):
            transformed = []
            # init with large/small sentinels
            minx = 10**9
            miny = 10**9
            if reflect:
                for x, y in pts:
                    x = -x
                    if rot == 0:
                        tx, ty = x, y
                    elif rot == 1:
                        tx, ty = -y, x
                    elif rot == 2:
                        tx, ty = -x, -y
                    else:
                        tx, ty = y, -x
                    transformed.append((tx, ty))
                    if tx < minx:
                        minx = tx
                    if ty < miny:
                        miny = ty
            else:
                for x, y in pts:
                    if rot == 0:
                        tx, ty = x, y
                    elif rot == 1:
                        tx, ty = -y, x
                    elif rot == 2:
                        tx, ty = -x, -y
                    else:
                        tx, ty = y, -x
                    transformed.append((tx, ty))
                    if tx < minx:
                        minx = tx
                    if ty < miny:
                        miny = ty
            # normalize by subtracting minx/miny and sort to canonical order
            norm = tuple(sorted(((tx - minx, ty - miny) for tx, ty in transformed)))
            syms.add(norm)
    return sorted(syms)


def placements_for_shape(shape, w: int, h: int):
    """
    Gera todos os placements (cells + mask) de uma forma em um tabuleiro w x h.
    Otimizações:
      - usa base_mask (bits relativos) e desloca com << offset em vez de calcular 1<<(idx) por célula.
      - evita construir 'cells' até confirmar máscara única.
      - usa variáveis locais para reduzir lookup.
    """
    syms = all_symmetries(shape)
    placements = []
    seen_masks = set()
    app = placements.append
    for s in syms:
        # bbox
        maxx = max(x for x, y in s)
        maxy = max(y for x, y in s)
        width_minus = w - maxx
        height_minus = h - maxy
        if width_minus <= 0 or height_minus <= 0:
            continue
        # base_idxs relativos e base_mask (bits com origem em (0,0))
        base_idxs = [y * w + x for x, y in s]
        base_mask = 0
        for bi in base_idxs:
            base_mask |= 1 << bi

        # iterar offsets possíveis; offset = by*w + bx permite fazer shift único
        for bx in range(width_minus):
            for by in range(height_minus):
                offset = by * w + bx
                mask = base_mask << offset
                if mask in seen_masks:
                    continue
                seen_masks.add(mask)
                # finalmente construir lista de células em coordenadas locais (x+bx,y+by)
                cells = [(x + bx, y + by) for x, y in s]
                app({"cells": cells, "mask": mask})
    return placements


def build_global_placements(shapes, w: int, h: int):
    """
    Constrói lista global de placements e índice por shape.
    Pequenas otimizações: append local, atribuição in-place de campos.
    """
    placements = []
    app = placements.append
    for sid, shape in enumerate(shapes):
        p = placements_for_shape(shape, w, h)
        for item in p:
            # atribui in-place (evita cópias)
            item["shape_id"] = sid
            item["popcount"] = item["mask"].bit_count()
            app(item)
    return placements


# generate simple free polyominoes of size n by naive search (used only for benchmark)
def generate_free_polyominoes(n: int):
    if n <= 0:
        return []
    if ominoes_dict is not None and n in ominoes_dict:
        return [normalize(c) for c in ominoes_dict[n]]

    seen = set()
    results = []

    def recurse(cells: set):
        if len(cells) == n:
            key = normalize(cells)
            if key not in seen:
                seen.add(key)
                results.append(key)
            return
        frontier = set()
        for x, y in cells:
            frontier.update(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))
        for nb in frontier:
            if nb in cells:
                continue
            cells.add(nb)
            recurse(cells)
            cells.remove(nb)

    recurse({(0, 0)})
    return results


# ---------------------- adaptive benchmark runner ----------------------
def adaptive_benchmark(
    func,
    args=(),
    kwargs=None,
    min_time=0.5,
    max_iters=1000000,
    warmup=2,
    clear_cache_each=False,
):
    if kwargs is None:
        kwargs = {}
    # warmup
    for _ in range(warmup):
        if clear_cache_each:
            clear_caches()
        func(*args, **kwargs)
    # timed loop
    times = []
    total = 0.0
    iters = 0
    while iters < max_iters and total < min_time:
        if clear_cache_each:
            clear_caches()
        t0 = perf_counter()
        func(*args, **kwargs)
        dt = perf_counter() - t0
        times.append(dt)
        total += dt
        iters += 1
    if not times:
        return {"iters": 0, "total": 0.0}
    best = min(times)
    avg = sum(times) / len(times)
    import statistics

    stdev = statistics.pstdev(times) if len(times) > 1 else 0.0
    return {
        "iters": len(times),
        "total": total,
        "avg": avg,
        "best": best,
        "stdev": stdev,
        "per_iter": times,
    }


# ---------------------- helpers to build test data ----------------------
def random_block_masks(w: int, h: int, k: int):
    total = w * h
    masks = []
    for _ in range(k):
        # create random blocked fraction between 0 and 0.5
        blocked = 0
        for i in range(total):
            # trunk-ignore(bandit/B311)
            if random.random() < 0.25:
                blocked |= 1 << i
        masks.append(blocked)
    return masks


def connected_component_of_size(w: int, h: int, size: int):
    # BFS from 0 in full grid to get first 'size' nodes connected (simple snake)
    nb = build_neighbors(w, h)
    start = 0
    seen = {start}
    q = [start]
    qi = 0
    while qi < len(q) and len(seen) < size:
        u = q[qi]
        qi += 1
        for v in nb[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
                if len(seen) >= size:
                    break
    nodes = list(sorted(seen))
    # build local adjacency
    map_global_to_local = {g: i for i, g in enumerate(nodes)}
    local_nb = [[] for _ in range(len(nodes))]
    for i, g in enumerate(nodes):
        for gg in nb[g]:
            if gg in map_global_to_local:
                local_nb[i].append(map_global_to_local[gg])
    return nodes, local_nb


# ---------------------- CLI handling ----------------------
def print_stats(name, res):
    print(
        f"[{name}] iters={res['iters']} total={res['total']:.6f}s avg={res['avg']:.6f}s best={res['best']:.6f}s stdev={res['stdev']:.6f}s"
    )


def usage():
    print(
        "Usage: python test.py bench <target> [args...] [--min-time T] [--max-iters N] [--clear-cache]"
    )
    print("Targets:")
    print("  neighbors W H")
    print("  bfs W H comp_size")
    print("  compute block_mask W H | --random K W H [--clear-cache]")
    print("  placements N W H")
    print("  all N W H")
    sys.exit(1)


def main(argv):
    if len(argv) < 2 or argv[1] != "bench":
        print("error 1")
        usage()
    args = argv[2:]
    # defaults
    min_time = 0.5
    max_iters = 1000000
    clear_cache_flag = False

    # parse global flags from tail
    if "--min-time" in args:
        i = args.index("--min-time")
        min_time = float(args[i + 1])
        del args[i : i + 2]
    if "--max-iters" in args:
        i = args.index("--max-iters")
        max_iters = int(args[i + 1])
        del args[i : i + 2]
    if "--clear-cache" in args:
        clear_cache_flag = True
        args.remove("--clear-cache")

    if not args:
        print("error 2")
        usage()
    target = args[0]

    if target == "neighbors":
        if len(args) < 3:
            print("error 3")
            usage()
        w = int(args[1])
        h = int(args[2])
        res = adaptive_benchmark(
            build_neighbors, (w, h), min_time=min_time, max_iters=max_iters, warmup=2
        )
        print_stats(f"build_neighbors {w}x{h}", res)

    elif target == "bfs":
        if len(args) < 4:
            print("error 4")
            usage()
        w = int(args[1])
        h = int(args[2])
        comp_size = int(args[3])
        nodes, local_nb = connected_component_of_size(w, h, comp_size)

        # benchmark bfs_local with random start each iteration via wrapper
        def fn():
            # trunk-ignore(bandit/B311)
            start = random.randrange(len(local_nb))
            bfs_local(start, local_nb)

        res = adaptive_benchmark(
            fn, (), min_time=min_time, max_iters=max_iters, warmup=2
        )
        print_stats(f"bfs_local comp_size={comp_size} ({w}x{h})", res)

    elif target == "compute":
        # two modes: block_mask provided or --random K
        if len(args) >= 4 and args[1] != "--random":
            block_mask = int(args[1])
            w = int(args[2])
            h = int(args[3])

            def fn():
                compute_diameter_and_path(block_mask, w, h)

            res = adaptive_benchmark(
                fn,
                (),
                min_time=min_time,
                max_iters=max_iters,
                warmup=1,
                clear_cache_each=clear_cache_flag,
            )
            print_stats(f"compute mask {block_mask} {w}x{h}", res)
        elif len(args) >= 4 and args[1] == "--random":
            k = int(args[2])
            w = int(args[3])
            h = int(args[4])
            masks = random_block_masks(w, h, k)

            def fn_cycle():
                m = masks.pop()
                compute_diameter_and_path(m, w, h)
                masks.insert(0, m)

            res = adaptive_benchmark(
                fn_cycle,
                (),
                min_time=min_time,
                max_iters=max_iters,
                warmup=1,
                clear_cache_each=clear_cache_flag,
            )
            print_stats(f"compute random {k} masks {w}x{h}", res)
        else:
            print("error 5")
            usage()

    elif target == "placements":
        if len(args) < 4:
            print("error 6")
            usage()
        n = int(args[1])
        w = int(args[2])
        h = int(args[3])
        shapes = generate_free_polyominoes(n)

        def fn():
            build_global_placements(shapes, w, h)

        res = adaptive_benchmark(
            fn, (), min_time=min_time, max_iters=max_iters, warmup=1
        )
        print_stats(f"placements n={n} {w}x{h}", res)

    elif target == "all":
        # run a sequence with sane defaults
        if len(args) < 4:
            print("error 7")
            usage()
        n = int(args[1])
        w = int(args[2])
        h = int(args[3])
        res1 = adaptive_benchmark(
            lambda: build_neighbors(w, h), (), min_time=min_time, max_iters=max_iters
        )
        print_stats(f"build_neighbors {w}x{h}", res1)
        _, local_nb = connected_component_of_size(w, h, 30)
        res2 = adaptive_benchmark(
            lambda: bfs_local(0, local_nb), (), min_time=min_time, max_iters=max_iters
        )
        print_stats(f"bfs_local comp_size=30 ({w}x{h})", res2)
        masks = random_block_masks(w, h, n)
        res3 = adaptive_benchmark(
            lambda: compute_diameter_and_path(masks[0], w, h),
            (),
            min_time=min_time,
            max_iters=max_iters,
        )
        print_stats(f"compute sample mask {w}x{h}", res3)
        shapes = generate_free_polyominoes(n)
        res4 = adaptive_benchmark(
            lambda: build_global_placements(shapes, w, h),
            (),
            min_time=min_time,
            max_iters=max_iters,
        )
        print_stats(f"placements n={n} {w}x{h}", res4)

    else:
        print("error 8")
        usage()


if __name__ == "__main__":
    main(sys.argv)
