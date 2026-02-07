from argparse import ArgumentParser
from collections import deque
from colorsys import hsv_to_rgb
from functools import lru_cache
from io import BytesIO
from multiprocessing import Manager, Pool, cpu_count
from os import makedirs, path
from random import Random, random
from time import perf_counter
from traceback import print_exc
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw

from big_polyominoes import big_ominoes_dict
from polyominoes import ominoes_dict

# ---------------------- constantes ----------------------
_ROTATIONS = (
    lambda x, y: (x, y),
    lambda x, y: (-y, x),
    lambda x, y: (-x, -y),
    lambda x, y: (y, -x),
)


# ---------------------- funções de normalização ----------------------
def _normalize_variants(cells: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Gera todas as variantes (reflexões+rotações) normalizadas sem loops extras."""
    pts = tuple(cells)
    k = len(pts)
    variants = []
    for reflect in (False, True):
        for rot_fn in _ROTATIONS:
            # Pré-alocar lista com tamanho k (micro-otimização)
            transformed = [None] * k
            if reflect:
                # aplicar reflexão em x antes da rotação
                for idx, (x, y) in enumerate(pts):
                    xr = -x
                    yr = y
                    tx, ty = rot_fn(xr, yr)
                    transformed[idx] = (tx, ty)
            else:
                for idx, (x, y) in enumerate(pts):
                    tx, ty = rot_fn(x, y)
                    transformed[idx] = (tx, ty)
            minx = min(p[0] for p in transformed)
            miny = min(p[1] for p in transformed)
            norm = tuple(sorted(((p[0] - minx, p[1] - miny) for p in transformed)))
            variants.append(norm)
    return variants


def normalize(cells: Set[Tuple[int, int]]) -> Tuple[int, int]:
    """Retorna a forma canônica mínima entre todas as variantes."""
    return min(_normalize_variants(cells))


# ---------------------- geração de ominos livres ----------------------
def generate_free_polyominoes(
    n: int, ominoes_dict: Dict[int, List[Set[Tuple[int, int]]]]
) -> List[Tuple[int, int]]:
    """
    Gera ominos livres (semântica igual ao seu código).
    - Usa poda por 'seen' com normalize.
    - evita `sorted(frontier)` (não necessário).
    - evita alocações extras dentro do laço.
    """
    if n <= 0:
        return []
    if n < 7:
        if ominoes_dict is not None and n in ominoes_dict:
            return [normalize(c) for c in ominoes_dict[n]]
    else:
        if big_ominoes_dict is not None and n in big_ominoes_dict:
            return [normalize(c) for c in big_ominoes_dict[n]]

    seen = set()
    results = []

    def recurse(cells, frontier):
        # cells: set de (x,y)
        # frontier: set de candidatos adjacentes (pode conter células já em cells)
        if len(cells) == n:
            key = normalize(cells)
            if key not in seen:
                seen.add(key)
                results.append(key)
            return
        # iterar sobre uma cópia do frontier — sem ordenar
        for nb in list(frontier):
            if nb in cells:
                continue
            # adicionar nb e atualizar frontier incrementalmente
            cells.add(nb)
            added = []
            x0, y0 = nb
            for cand in ((x0 + 1, y0), (x0 - 1, y0), (x0, y0 + 1), (x0, y0 - 1)):
                if cand not in cells and cand not in frontier:
                    frontier.add(cand)
                    added.append(cand)
            # recurse
            recurse(cells, frontier)
            # desfazer adições
            for a in added:
                frontier.remove(a)
            cells.remove(nb)

    recurse({(0, 0)}, set(((0 + 1, 0), (0 - 1, 0), (0, 0 + 1), (0, 0 - 1))))
    return results


# ---------------------- geração de placements ----------------------
def all_symmetries(
    canonical: Tuple[Tuple[int, int], ...],
) -> List[Tuple[Tuple[int, int], ...]]:
    """
    Gera todas as simetrias (reflexões+rotações) normalizadas de `canonical`.
    Minimiza alocações: calcula minx/miny on-the-fly e evita chamadas de função por ponto.
    OTIMIZADO: pré-alocação de listas.
    """
    pts = canonical
    k = len(pts)
    syms = set()
    for reflect in (False, True):
        for rot in range(4):
            # Pré-alocar lista com tamanho k
            transformed = [None] * k
            # init with large/small sentinels
            minx = 10**9
            miny = 10**9
            if reflect:
                for idx, (x, y) in enumerate(pts):
                    x = -x
                    if rot == 0:
                        tx, ty = x, y
                    elif rot == 1:
                        tx, ty = -y, x
                    elif rot == 2:
                        tx, ty = -x, -y
                    else:
                        tx, ty = y, -x
                    transformed[idx] = (tx, ty)
                    if tx < minx:
                        minx = tx
                    if ty < miny:
                        miny = ty
            else:
                for idx, (x, y) in enumerate(pts):
                    if rot == 0:
                        tx, ty = x, y
                    elif rot == 1:
                        tx, ty = -y, x
                    elif rot == 2:
                        tx, ty = -x, -y
                    else:
                        tx, ty = y, -x
                    transformed[idx] = (tx, ty)
                    if tx < minx:
                        minx = tx
                    if ty < miny:
                        miny = ty
            # normalize by subtracting minx/miny and sort to canonical order
            norm = tuple(sorted(((tx - minx, ty - miny) for tx, ty in transformed)))
            syms.add(norm)
    return sorted(syms)


def placements_for_shape(
    shape: Set[Tuple[int, int]], w: int, h: int
) -> List[Dict[str, Any]]:
    """
    Gera todos os placements (cells + mask) de uma forma em um tabuleiro w x h.
    Otimizações:
      - usa base_mask (bits relativos) e desloca com << offset em vez de calcular 1<<(idx) por célula.
      - evita construir 'cells' até confirmar máscara única.
      - usa variáveis locais para reduzir lookup.
      - NOVO: pré-filtra simetrias que claramente não cabem
    """
    # ensure we pass a tuple-of-tuples (canonical) to all_symmetries
    if isinstance(shape, set):
        canonical = tuple(sorted(shape))
    else:
        canonical = tuple(shape)
    syms = all_symmetries(canonical)

    # OTIMIZAÇÃO: Pré-filtrar simetrias que não cabem no tabuleiro
    syms_filtered = [
        s for s in syms if max(x for x, _ in s) < w and max(y for _, y in s) < h
    ]

    placements = []
    seen_masks = set()
    app = placements.append
    for s in syms_filtered:
        # bbox
        maxx = max(x for x, _ in s)
        maxy = max(y for _, y in s)
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


# ---------------------- optimized utilities ----------------------
def build_neighbors(w: int, h: int) -> List[List[int]]:
    """
    Retorna lista de vizinhos para cada índice 0..w*h-1.
    Otimizações:
     - uso direto de aritmética, sem operações de módulo/divisão repetidas dentro do loop.
     - precomputação de limites.
    """
    total = w * h
    nb = [[] for _ in range(total)]
    # percorre por y, x, em vez de idx só
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
    start_local: int, local_nb: List[List[int]]
) -> Tuple[List[int], List[int]]:
    """
    BFS rápido em grafo pequeno representado por local_nb (lista de listas).
    Retorna (dist, parent) onde ambos são listas de tamanho n_local.
    """
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


# ====== Reconstruir caminho em índices locais e mapear para globais ======
def reconstruct_path_local(
    parent_local: List[int], src_local: int, dest_local: int, local_to_global: List[int]
) -> List[int]:
    path_local = []
    cur = dest_local
    while cur != -1:
        path_local.append(cur)
        if cur == src_local:
            break
        cur = parent_local[cur]
    path_local.reverse()
    # map to global indices
    return [local_to_global[i] for i in path_local]


# OTIMIZAÇÃO: Threshold adaptativo baseado no tamanho do grid
def get_adaptive_threshold(w: int, h: int) -> int:
    """
    Retorna threshold adaptativo baseado no tamanho do grid.
    Grids maiores → threshold menor para economizar memória/tempo.
    """
    total = w * h
    if total <= 50:
        return 120  # all-pairs BFS até 120 células
    elif total <= 100:
        return 80
    elif total <= 200:
        return 50
    else:
        return 30  # grids muito grandes: heurística mais agressiva


# tamanho do LRU (ajustável)
_LRU_CACHE_SIZE = 10**15

# neighbor cache global (usado por compute)
_neighbors_cache = {}


# função não-cacheada (mantém a mesma lógica, mas usa _neighbors_cache)
def _compute_diameter_and_path_uncached(
    block_mask: int, w: int, h: int
) -> Tuple[int, Optional[int], Optional[int], Tuple[int, ...]]:
    total = w * h
    if total == 0:
        return 0, None, None, tuple()
    full_mask = (1 << total) - 1
    empty_mask = full_mask & (~block_mask)
    if empty_mask == 0:
        return 0, None, None, tuple()

    # cache neighbors via global
    neighbors = _neighbors_cache.get((w, h))
    if neighbors is None:
        neighbors = build_neighbors(w, h)
        _neighbors_cache[(w, h)] = neighbors
    nb = neighbors

    remaining = empty_mask
    best_diam = 0
    best_a = None
    best_b = None
    best_path = []

    while remaining:
        lsb = remaining & -remaining
        start = lsb.bit_length() - 1

        # flood fill component (global indices)
        comp_nodes = []
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

        # remove comp from remaining
        comp_mask = 0
        for u in comp_nodes:
            comp_mask |= 1 << u
        remaining &= ~comp_mask

        comp_size = len(comp_nodes)
        if comp_size < 2:
            continue

        # build mapping global->local
        map_global_to_local = [-1] * total
        for i, g in enumerate(comp_nodes):
            map_global_to_local[g] = i
        local_to_global = comp_nodes[:]  # index -> global index

        # build local adjacency
        local_nb = [[] for _ in range(comp_size)]
        for i, g in enumerate(comp_nodes):
            for gg in nb[g]:
                gg_local = map_global_to_local[gg]
                if gg_local != -1:
                    local_nb[i].append(gg_local)

        # OTIMIZAÇÃO: usar threshold adaptativo
        threshold = get_adaptive_threshold(w, h)

        if comp_size <= threshold:
            # exact all-pairs BFS in grafo local
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
            # heuristica dupla
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

    # Return como tupla imutável para cache (path como tuple)
    return (
        best_diam + 1,
        (best_a if best_a is not None else None),
        (best_b if best_b is not None else None),
        tuple(best_path),
    )


# wrapper cacheado (LRU) — o cache armazenará as tuplas retornadas acima
_cached_compute = lru_cache(maxsize=_LRU_CACHE_SIZE)(
    _compute_diameter_and_path_uncached
)


# Função pública que converte o path de volta para list (compatibilidade)
def compute_diameter_and_path(
    block_mask: int, w: int, h: int
) -> Tuple[int, Optional[int], Optional[int], List[int]]:
    nodes, a, b, path_t = _cached_compute(block_mask, w, h)
    return nodes, a, b, list(path_t)


# ====== score_selection agora só chama compute_diameter_and_path (LRU ativa) ======
def score_selection(
    occ_mask: int, w: int, h: int
) -> Tuple[int, Optional[int], Optional[int], List[int]]:
    """
    Usa LRU cache via compute_diameter_and_path wrapper.
    """
    return compute_diameter_and_path(occ_mask, w, h)


# ---------------------- search helpers ----------------------
def build_global_placements(shapes, w: int, h: int) -> List[Dict[str, Any]]:
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


def random_feasible_selection(
    placements: List[Dict[str, Any]], max_pieces: int, no_repeat: bool, rng
) -> Tuple[List[int], int]:
    N = len(placements)
    chosen = []
    occ = 0
    used_shape = 0
    pool = list(range(N))
    rng.shuffle(pool)
    for i in pool:
        if len(chosen) >= max_pieces:
            break
        pl = placements[i]
        if occ & pl["mask"]:
            continue
        sid = pl["shape_id"]
        if no_repeat and ((used_shape >> sid) & 1):
            continue
        chosen.append(i)
        occ |= pl["mask"]
        used_shape |= 1 << sid
    return chosen, occ


# OTIMIZAÇÃO: neighbor_move agora recebe e retorna used_shapes
def neighbor_move(
    chosen: List[int],
    occ: int,
    used_shapes: int,
    placements: List[Dict[str, Any]],
    max_pieces: int,
    no_repeat: bool,
    rng,
) -> Tuple[List[int], int, int]:
    """
    OTIMIZADO: used_shapes é passado como parâmetro e atualizado incrementalmente.
    Economiza O(P) operações por iteração.
    """
    N = len(placements)
    choice = rng.random()
    chosen_set = set(chosen)

    # ADD MOVE
    if choice < 0.35 and len(chosen) < max_pieces:
        pool = list(range(N))
        rng.shuffle(pool)
        for i in pool:
            if i in chosen_set:
                continue
            pl = placements[i]
            if occ & pl["mask"]:
                continue
            sid = pl["shape_id"]
            if no_repeat and ((used_shapes >> sid) & 1):
                continue
            new_chosen = chosen + [i]
            new_occ = occ | pl["mask"]
            new_used = used_shapes | (1 << sid)  # ← atualização incremental
            return new_chosen, new_occ, new_used

    # REMOVE MOVE
    if choice < 0.7 and len(chosen) > 0:
        rem = rng.choice(chosen)
        new_chosen = [c for c in chosen if c != rem]
        new_occ = 0
        new_used = 0
        # Reconstruir (inevitável na remoção)
        for c in new_chosen:
            new_occ |= placements[c]["mask"]
            new_used |= 1 << placements[c]["shape_id"]
        return new_chosen, new_occ, new_used

    # SWAP MOVE
    if len(chosen) > 0:
        rem = rng.choice(chosen)
        pool = list(range(N))
        rng.shuffle(pool)
        new_chosen = [c for c in chosen if c != rem]
        occ_tmp = 0
        used_tmp = 0
        for c in new_chosen:
            occ_tmp |= placements[c]["mask"]
            used_tmp |= 1 << placements[c]["shape_id"]
        for i in pool:
            if i in new_chosen:
                continue
            pl = placements[i]
            if occ_tmp & pl["mask"]:
                continue
            sid = pl["shape_id"]
            if no_repeat and ((used_tmp >> sid) & 1):
                continue
            new_chosen2 = new_chosen + [i]
            new_occ2 = occ_tmp | pl["mask"]
            new_used2 = used_tmp | (1 << sid)
            return new_chosen2, new_occ2, new_used2

    # Fallback
    if len(chosen) > 0:
        rem = rng.choice(chosen)
        new_chosen = [c for c in chosen if c != rem]
        new_occ = 0
        new_used = 0
        for c in new_chosen:
            new_occ |= placements[c]["mask"]
            new_used |= 1 << placements[c]["shape_id"]
        return new_chosen, new_occ, new_used

    return chosen, occ, used_shapes


# ---------------------- rendering (palette cached) ----------------------
_palette_cache = {}


def get_palette(num_shapes: int) -> List[Tuple[int, int, int]]:
    if num_shapes in _palette_cache:
        return _palette_cache[num_shapes]
    # trunk-ignore(bandit/B311)
    rng = Random(12345)
    palette = []
    for i in range(num_shapes):
        hval = (i / max(1, num_shapes)) % 1.0
        sat = 0.6 + (rng.random() * 0.35)
        val = 0.7 + (rng.random() * 0.3)
        r, g, b = hsv_to_rgb(hval, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    _palette_cache[num_shapes] = palette
    return palette


def render_maze_and_path_by_shape(
    placements: List[Dict[str, Any]],
    sel: List[int],
    w: int,
    h: int,
    path_board: List[int],
    out_file: str,
    cell: int = 24,
) -> str:
    total = w * h
    cell_shape = [-1] * total
    if sel:
        for pl_idx in sel:
            pl = placements[pl_idx]
            sid = pl["shape_id"]
            for x, y in pl["cells"]:
                cell_shape[y * w + x] = sid

    max_sid = 0
    for p in placements:
        if p["shape_id"] > max_sid:
            max_sid = p["shape_id"]
    shape_count = max_sid + 1
    palette = get_palette(shape_count)

    empty_col = (250, 250, 250)
    path_col = (255, 80, 80)
    end1_col = (20, 160, 20)
    end2_col = (20, 80, 200)

    wpx = w * cell
    hpx = h * cell
    img = Image.new("RGBA", (wpx, hpx), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    for y in range(h):
        y0 = y * cell
        y1 = y0 + cell - 1
        row_base = y * w
        for x in range(w):
            x0 = x * cell
            x1 = x0 + cell - 1
            sid = cell_shape[row_base + x]
            if sid == -1:
                draw.rectangle([x0, y0, x1, y1], fill=empty_col)
            else:
                col = palette[sid] if 0 <= sid < len(palette) else (30, 30, 30)
                draw.rectangle([x0, y0, x1, y1], fill=col)
            draw.rectangle([x0, y0, x1, y1], outline=(200, 200, 200))

    if path_board and len(path_board) >= 1:
        for idx in path_board:
            x = idx % w
            y = idx // w
            x0 = x * cell
            y0 = y * cell
            x1 = x0 + cell - 1
            y1 = y0 + cell - 1
            draw.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], fill=path_col)
        if len(path_board) >= 2:
            a = path_board[0]
            b = path_board[-1]
            xa, ya = a % w, a // w
            xb, yb = b % w, b // w
            draw.ellipse(
                [
                    xa * cell + 4,
                    ya * cell + 4,
                    xa * cell + cell - 5,
                    ya * cell + cell - 5,
                ],
                fill=end1_col,
            )
            draw.ellipse(
                [
                    xb * cell + 4,
                    yb * cell + 4,
                    xb * cell + cell - 5,
                    yb * cell + cell - 5,
                ],
                fill=end2_col,
            )
        else:
            a = path_board[0]
            xa, ya = a % w, a // w
            draw.ellipse(
                [
                    xa * cell + 4,
                    ya * cell + 4,
                    xa * cell + cell - 5,
                    ya * cell + cell - 5,
                ],
                fill=end1_col,
            )

    try:
        d = path.dirname(path.abspath(out_file))
        if d and not path.exists(d):
            makedirs(d, exist_ok=True)
    except Exception:
        print_exc()
    try:
        img.save(out_file, format="PNG")
        return path.abspath(out_file)
    except Exception:
        print_exc()
        try:
            bio = BytesIO()
            img.save(bio, format="PNG")
            bio.seek(0)
            with open(out_file, "wb") as f:
                f.write(bio.read())
            return path.abspath(out_file)
        except Exception:
            print_exc()
            raise


# ============================================================================
# PARALLEL WORKER FUNCTIONS
# ============================================================================


def _optimize_worker(args):
    """
    Worker function para paralelizar optimize_maze.
    Cada worker roda uma cadeia independente de simulated annealing.
    """
    (
        worker_id,
        placements,
        w,
        h,
        max_pieces,
        no_repeat,
        time_limit,
        seed_base,
        out_prefix,
        cell,
        first_greedy,
        shared_best_score,
        max_no_improve,
    ) = args

    # Seed único para cada worker
    seed = seed_base + worker_id if seed_base is not None else None
    # trunk-ignore(bandit/B311)
    rng = Random(seed)
    start_time = perf_counter()

    local_best_sel = None
    local_best_occ = 0
    local_best_score = -1
    local_best_path = []

    # Inicialização greedy
    for _ in range(first_greedy):
        sel, occ = random_feasible_selection(placements, max_pieces, no_repeat, rng)
        score, _, _, path = score_selection(occ, w, h)
        if score > local_best_score:
            local_best_score = score
            local_best_sel, local_best_occ, local_best_path = sel, occ, path

    if local_best_sel is None:
        local_best_sel = []
        local_best_occ = 0
        local_best_path = []
        local_best_score = 0

    current_sel = local_best_sel.copy() if local_best_sel else []
    current_occ = local_best_occ
    current_score = local_best_score

    # Manter used_shapes como estado
    current_used = 0
    for i in current_sel:
        current_used |= 1 << placements[i]["shape_id"]

    T0 = 1.0
    Tmin = 0.001
    step = 0
    no_improve_steps = 0
    last_best_step = 0

    while perf_counter() - start_time < time_limit:
        step += 1
        sel2, occ2, used2 = neighbor_move(
            current_sel.copy(),
            current_occ,
            current_used,
            placements,
            max_pieces,
            no_repeat,
            rng,
        )
        sc2, _, _, path2 = score_selection(occ2, w, h)
        accept = False
        if sc2 > current_score:
            accept = True
        else:
            frac = (perf_counter() - start_time) / max(1.0, time_limit)
            T = T0 * (1 - frac) + Tmin * frac
            delta = sc2 - current_score
            if delta >= 0:
                prob = 1.0
            else:
                prob = min(1.0, (2.718281828459045 ** (delta / max(1e-6, T))))
            if rng.random() < prob:
                accept = True

        if accept:
            current_sel = sel2
            current_occ = occ2
            current_score = sc2
            current_used = used2

            if sc2 > local_best_score:
                local_best_score = sc2
                local_best_sel = sel2.copy()
                local_best_occ = occ2
                local_best_path = path2
                last_best_step = step
                no_improve_steps = 0
            else:
                no_improve_steps = step - last_best_step
        else:
            no_improve_steps = step - last_best_step

        # Early stopping
        if no_improve_steps >= max_no_improve:
            break

    return (
        worker_id,
        local_best_score,
        local_best_sel,
        local_best_occ,
        local_best_path,
        step,
    )


# ---------------------- PARALLEL heuristic optimizer ----------------------
def optimize_maze_parallel(
    placements: List[Dict[str, Any]],
    w: int,
    h: int,
    max_pieces: int = 8,
    no_repeat: bool = True,
    time_limit: Optional[float] = 30,
    seed: int | None = None,
    init_selection: List[int] | None = None,
    init_placement: int | None = None,
    out: str = "best.png",
    cell: int = 30,
    first_greedy: int = 100,
    n_workers: Optional[int] = None,
    max_no_improve: int = 5000,
) -> Tuple[int, List[int], int, List[int]]:
    """
    VERSÃO PARALELA de optimize_maze usando múltiplas cadeias independentes.

    Cada worker roda sua própria cadeia de simulated annealing em paralelo.
    No final, retorna o melhor resultado entre todos os workers.

    Args:
        n_workers: Número de processos paralelos. Se None, usa cpu_count().
    """
    if time_limit is None:
        time_limit = float("inf")

    if n_workers is None:
        n_workers = cpu_count() - 2

    n_workers = max(1, min(n_workers, cpu_count() - 2))  # Limitar ao número de CPUs

    print(f"[parallel optimize] Using {n_workers} workers, time_limit={time_limit}s")

    # Se init_selection ou init_placement estão definidos, usar modo sequencial
    if init_selection or init_placement:
        print(
            "[parallel optimize] init_selection/init_placement set, using sequential mode"
        )
        return optimize_maze(
            placements,
            w,
            h,
            max_pieces,
            no_repeat,
            time_limit,
            seed,
            init_selection,
            init_placement,
            out,
            cell,
            first_greedy,
            max_no_improve,
        )

    start_time = perf_counter()

    # Manager para compartilhar best score (não usado atualmente, mas disponível)
    manager = Manager()
    shared_best = manager.Value("i", -1)

    # Preparar argumentos para workers
    worker_args = []
    for worker_id in range(n_workers):
        args = (
            worker_id,
            placements,
            w,
            h,
            max_pieces,
            no_repeat,
            time_limit,
            seed,
            out,
            cell,
            first_greedy // n_workers + 1,  # Dividir first_greedy entre workers
            shared_best,
            max_no_improve,
        )
        worker_args.append(args)

    # Executar workers em paralelo
    print(f"[parallel optimize] Starting {n_workers} parallel chains...")
    with Pool(n_workers) as pool:
        results = pool.map(_optimize_worker, worker_args)

    # Encontrar o melhor resultado
    best_result = max(results, key=lambda x: x[1])  # x[1] é o score
    worker_id, best_score, best_sel, best_occ, best_path, steps = best_result

    elapsed = perf_counter() - start_time
    total_steps = sum(r[5] for r in results)

    print(f"[parallel optimize] Finished in {elapsed:.1f}s")
    print(f"  Best from worker {worker_id}: score={best_score}, pieces={len(best_sel)}")
    print(f"  Total steps across all workers: {total_steps}")
    print(f"  Average steps per worker: {total_steps/n_workers:.0f}")

    # Renderizar melhor resultado
    if best_sel:
        try:
            render_maze_and_path_by_shape(
                placements, best_sel, w, h, best_path, out_file=out, cell=cell
            )
        # trunk-ignore(bandit/B110)
        except Exception:
            pass

    return best_score, best_sel, best_occ, best_path


# Fallback para modo sequencial
def optimize_maze(
    placements: List[Dict[str, Any]],
    w: int,
    h: int,
    max_pieces: int = 8,
    no_repeat: bool = True,
    time_limit: Optional[float] = 30,
    seed: int | None = None,
    init_selection: List[int] | None = None,
    init_placement: int | None = None,
    out: str = "best.png",
    cell: int = 30,
    first_greedy: int = 100,
    max_no_improve: int = 10000,
) -> Tuple[int, List[int], int, List[int]]:
    """Versão sequencial original (mantida para compatibilidade)."""
    if time_limit is None:
        time_limit = float("inf")
    # trunk-ignore(bandit/B311)
    rng = Random(seed)
    start_time = perf_counter()

    N = len(placements)
    best_sel = None
    best_occ = 0
    best_score = -1
    best_path = []

    print(
        f"[optimize] seed={seed} time_limit={time_limit}s max_pieces={max_pieces} no_repeat={no_repeat}"
    )

    # build from init_selection or placement or greedy seeds
    if init_selection:
        occ = 0
        sel_clean = []
        used_shapes = 0
        for i in init_selection:
            if not (0 <= i < N):
                print(f"[init] skipping invalid placement index {i}")
                continue
            p = placements[i]
            if occ & p["mask"]:
                print(f"[init] skipping overlapping placement index {i}")
                continue
            sid = p["shape_id"]
            if no_repeat and ((used_shapes >> sid) & 1):
                print(f"[init] skipping placement {i} due to repeat shape {sid}")
                continue
            sel_clean.append(i)
            occ |= p["mask"]
            used_shapes |= 1 << sid
            if len(sel_clean) >= max_pieces:
                break
        if not sel_clean:
            raise ValueError("Provided init_selection invalid.")
        best_sel = sel_clean
        best_occ = occ
        best_score, _, _, best_path = score_selection(best_occ, w, h)
        print(f"[init selection] start score={best_score}, pieces={len(best_sel)}")
    elif init_placement is not None:
        pl = placements[init_placement]
        occ = pl["mask"]
        used_shapes = 1 << pl["shape_id"] if no_repeat else 0
        sel = [init_placement]
        for j in range(N):
            if len(sel) >= max_pieces:
                break
            if j == init_placement:
                continue
            pj = placements[j]
            if occ & pj["mask"]:
                continue
            sid = pj["shape_id"]
            if no_repeat and ((used_shapes >> sid) & 1):
                continue
            sel.append(j)
            occ |= pj["mask"]
            used_shapes |= 1 << sid
        best_sel = sel
        best_occ = occ
        best_score, _, _, best_path = score_selection(best_occ, w, h)
        print(
            f"[init placement {init_placement}] start score={best_score}, pieces={len(best_sel)}"
        )
    else:
        print(
            f"[greedy seed] running {first_greedy} random feasible samples to initialize"
        )
        for attempt in range(first_greedy):
            sel, occ = random_feasible_selection(placements, max_pieces, no_repeat, rng)
            score, _, _, path = score_selection(occ, w, h)
            if score > best_score:
                best_score = score
                best_sel, best_occ, best_path = sel, occ, path
                if attempt % 20 == 0:
                    print(
                        f"[seed {attempt+1}] best score={best_score}, pieces={len(best_sel)} (t={perf_counter()-start_time:.1f}s)"
                    )
        if best_sel is None:
            best_sel = []
            best_occ = 0
            best_path = []
            best_score = 0
            print("[seed] no feasible seed found; starting from empty selection")

    print(
        f"[start] initial best score={best_score}, pieces={0 if not best_sel else len(best_sel)}"
    )
    current_sel = best_sel.copy() if best_sel else []
    current_occ = best_occ
    current_score = best_score

    # OTIMIZAÇÃO: manter used_shapes como estado persistente
    current_used = 0
    for i in current_sel:
        current_used |= 1 << placements[i]["shape_id"]

    T0 = 1.0
    Tmin = 0.001
    step = 0

    # OTIMIZAÇÃO: Parâmetros de early stopping
    no_improve_steps = 0
    last_best_step = 0

    while perf_counter() - start_time < time_limit:
        step += 1
        sel2, occ2, used2 = neighbor_move(
            current_sel.copy(),
            current_occ,
            current_used,
            placements,
            max_pieces,
            no_repeat,
            rng,
        )
        sc2, _, _, path2 = score_selection(occ2, w, h)
        accept = False
        if sc2 > current_score:
            accept = True
        else:
            frac = (perf_counter() - start_time) / max(
                1.0, time_limit if time_limit != float("inf") else 1e6
            )
            T = T0 * (1 - frac) + Tmin * frac
            delta = sc2 - current_score
            if delta >= 0:
                prob = 1.0
            else:
                prob = min(1.0, (2.718281828459045 ** (delta / max(1e-6, T))))
            # trunk-ignore(bandit/B311)
            if random() < prob:
                accept = True

        if accept:
            current_sel = sel2
            current_occ = occ2
            current_score = sc2
            current_used = used2

            if sc2 > best_score:
                best_score = sc2
                best_sel = sel2.copy()
                best_occ = occ2
                best_path = path2
                last_best_step = step
                no_improve_steps = 0
                tnow = perf_counter()
                print(
                    f"[{step}] New best: diameter={best_score}, pieces={len(best_sel)} (t={tnow-start_time:.1f}s)"
                )
                try:
                    render_maze_and_path_by_shape(
                        placements, best_sel, w, h, best_path, out_file=out, cell=cell
                    )
                # trunk-ignore(bandit/B110)
                except Exception:
                    pass
            else:
                no_improve_steps = step - last_best_step
        else:
            no_improve_steps = step - last_best_step

        if no_improve_steps >= max_no_improve:
            print(f"[early stop] no improvement in {no_improve_steps} steps")
            break

    print(
        f"[done] elapsed={perf_counter()-start_time:.1f}s steps={step} best_score={best_score} pieces={0 if not best_sel else len(best_sel)}"
    )
    return best_score, best_sel, best_occ, best_path


# ============================================================================
# PARALLEL BRUTEFORCE WORKER FUNCTIONS
# ============================================================================
def _bruteforce_worker(args):
    """
    Worker function para paralelizar bruteforce_search.
    Cada worker explora um subset do espaço de busca.
    """
    (
        worker_id,
        start_idx,
        end_idx,
        placements,
        w,
        h,
        max_pieces,
        no_repeat,
        time_limit,
        order,
        masks,
        maps,
        total,
        shared_best_score,
        shared_lock,
    ) = args

    start_time = perf_counter()

    local_best_score = -1
    local_best_sel = []
    local_best_occ = 0
    local_best_path = []

    nodes_visited = 0
    time_up = False
    seen_canons = set()
    state_memo = {}
    pruned_by_memo = 0

    # Cache de canonização
    @lru_cache(maxsize=None)
    def canonical_mask_cached(mask: int) -> int:
        best = None
        for mapping in maps:
            m = mask
            res = 0
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                res |= 1 << mapping[idx]
                m &= m - 1
            if best is None or res < best:
                best = res
        return best if best is not None else mask

    def dfs(i, sel, occ, used_shapes):
        nonlocal local_best_score, local_best_sel, local_best_occ, local_best_path
        nonlocal nodes_visited, time_up, pruned_by_memo

        if perf_counter() - start_time > time_limit:
            time_up = True
            return

        nodes_visited += 1

        # Memoization
        canon = canonical_mask_cached(occ)
        n_pieces = len(sel)
        state_key = (canon, n_pieces)

        # Verificar contra best global (thread-safe read)
        with shared_lock:
            current_global_best = shared_best_score.value

        if state_key in state_memo:
            if state_memo[state_key] <= max(local_best_score, current_global_best):
                pruned_by_memo += 1
                return

        # Evaluate
        score, _, _, path = score_selection(occ, w, h)

        # Atualizar memo
        if state_key not in state_memo or state_memo[state_key] < score:
            state_memo[state_key] = score

        if score > local_best_score:
            local_best_score = score
            local_best_sel = sel.copy()
            local_best_occ = occ
            local_best_path = path

            # Atualizar global best se necessário
            with shared_lock:
                if score > shared_best_score.value:
                    shared_best_score.value = score

        # Upper bound prune
        blocked = occ.bit_count()
        empty_cells = total - blocked
        if empty_cells <= 1 or (empty_cells - 1) <= max(
            local_best_score, current_global_best
        ):
            return

        if len(sel) >= max_pieces or i >= len(order):
            return

        # Canonical pruning
        if canon in seen_canons:
            return
        seen_canons.add(canon)

        # Explore
        for k in range(i, len(order)):
            if time_up:
                break
            idx = order[k]
            pl = placements[idx]
            sid = pl["shape_id"]
            mask = masks[idx]

            if occ & mask:
                continue
            if no_repeat and ((used_shapes >> sid) & 1):
                continue

            sel.append(idx)
            dfs(k + 1, sel, occ | mask, used_shapes | (1 << sid))
            sel.pop()

            if time_up:
                break

    # Explorar apenas placements no range [start_idx, end_idx)
    for start_placement_idx in range(start_idx, end_idx):
        if time_up:
            break

        idx = order[start_placement_idx]
        pl = placements[idx]
        occ = pl["mask"]
        used = 1 << pl["shape_id"] if no_repeat else 0
        sel = [idx]

        dfs(start_placement_idx + 1, sel, occ, used)

    elapsed = perf_counter() - start_time
    return (
        worker_id,
        local_best_score,
        local_best_sel,
        local_best_occ,
        local_best_path,
        nodes_visited,
        pruned_by_memo,
        elapsed,
    )


# ---------------------- PARALLEL brute-force search ----------------------
def bruteforce_search_parallel(
    placements: List[Dict[str, Any]],
    w: int,
    h: int,
    max_pieces: int = 8,
    no_repeat: bool = True,
    time_limit: Optional[float] = None,
    out: str = "brute_best.png",
    cell: int = 30,
    n_workers: Optional[int] = None,
):
    """
    VERSÃO PARALELA de bruteforce_search.

    Divide o espaço de busca entre múltiplos workers que exploram em paralelo.
    Cada worker começa com um subset diferente de placements iniciais.

    Args:
        n_workers: Número de processos paralelos. Se None, usa cpu_count().
    """
    if time_limit is None:
        time_limit = float("inf")

    if n_workers is None:
        n_workers = cpu_count() - 2

    n_workers = max(1, min(n_workers, cpu_count() - 2))

    start_time = perf_counter()

    N = len(placements)
    total = w * h

    print(f"[parallel brute] Using {n_workers} workers for {N} placements")

    # Precompute
    masks = [p["mask"] for p in placements]

    # Shape frequency
    max_shape_id = max((p["shape_id"] for p in placements), default=-1)
    shape_freq = [0] * (max_shape_id + 1)
    for p in placements:
        shape_freq[p["shape_id"]] += 1

    # Overlap count
    overlap_count = [0] * N
    for i in range(N):
        mi = masks[i]
        cnt = sum(1 for j in range(N) if i != j and (mi & masks[j]))
        overlap_count[i] = cnt

    # Order placements
    order = list(range(N))
    order.sort(
        key=lambda i: (
            shape_freq[placements[i]["shape_id"]],
            -overlap_count[i],
            placements[i]["shape_id"],
            -masks[i].bit_count(),
        )
    )

    # Build symmetry maps
    def valid_transforms():
        trans = []
        trans.append(("id", lambda x, y: (x, y)))
        trans.append(("rot180", lambda x, y: (w - 1 - x, h - 1 - y)))
        trans.append(("flip_x", lambda x, y: (w - 1 - x, y)))
        trans.append(("flip_y", lambda x, y: (x, h - 1 - y)))
        if w == h:
            trans.append(("rot90", lambda x, y: (y, w - 1 - x)))
            trans.append(("rot270", lambda x, y: (h - 1 - y, x)))
            trans.append(("diag", lambda x, y: (y, x)))
            trans.append(("antidiag", lambda x, y: (h - 1 - y, w - 1 - x)))
        return trans

    trans_funcs = valid_transforms()
    maps = []
    for _, f in trans_funcs:
        mapping = [0] * total
        ok = True
        for idx in range(total):
            x = idx % w
            y = idx // w
            nx, ny = f(x, y)
            if not (0 <= nx < w and 0 <= ny < h):
                ok = False
                break
            mapping[idx] = ny * w + nx
        if ok:
            maps.append(tuple(mapping))

    # Manager para compartilhar best score
    manager = Manager()
    shared_best_score = manager.Value("i", -1)
    shared_lock = manager.Lock()

    # Dividir espaço de busca
    chunk_size = max(1, N // n_workers)
    worker_args = []

    for worker_id in range(n_workers):
        start_idx = worker_id * chunk_size
        end_idx = min(start_idx + chunk_size, N) if worker_id < n_workers - 1 else N

        args = (
            worker_id,
            start_idx,
            end_idx,
            placements,
            w,
            h,
            max_pieces,
            no_repeat,
            time_limit,
            order,
            masks,
            maps,
            total,
            shared_best_score,
            shared_lock,
        )
        worker_args.append(args)

    # Executar workers
    print(f"[parallel brute] Starting {n_workers} parallel branches...")
    with Pool(n_workers) as pool:
        results = pool.map(_bruteforce_worker, worker_args)

    # Combinar resultados
    best_result = max(results, key=lambda x: x[1])
    worker_id, best_score, best_sel, best_occ, best_path, nodes, memo_prunes, _ = (
        best_result
    )

    total_nodes = sum(r[5] for r in results)
    total_memo_prunes = sum(r[6] for r in results)
    elapsed = perf_counter() - start_time

    print(f"[parallel brute] Finished in {elapsed:.2f}s")
    print(f"  Best from worker {worker_id}: score={best_score}, pieces={len(best_sel)}")
    print(f"  Total nodes visited: {total_nodes}")
    print(f"  Total memo prunes: {total_memo_prunes}")
    print(f"  Average nodes per worker: {total_nodes/n_workers:.0f}")

    # Renderizar melhor
    if best_sel:
        try:
            render_maze_and_path_by_shape(
                placements, best_sel, w, h, best_path, out_file=out, cell=cell
            )
        # trunk-ignore(bandit/B110)
        except Exception:
            pass

    return best_score, best_sel, best_occ, best_path


# Fallback sequencial para bruteforce
def bruteforce_search(
    placements: List[Dict[str, Any]],
    w: int,
    h: int,
    max_pieces: int = 8,
    no_repeat: bool = True,
    time_limit: Optional[float] = None,
    out: str = "brute_best.png",
    cell: int = 30,
):
    """Versão sequencial (mantida para compatibilidade)."""
    if time_limit is None:
        time_limit = float("inf")
    start_time = perf_counter()

    N = len(placements)
    total = w * h

    best_score = -1
    best_sel = []
    best_occ = 0
    best_path = []

    # Precompute masks and popcounts
    masks = [p["mask"] for p in placements]
    popcounts = [m.bit_count() for m in masks]

    # shape frequency
    max_shape_id = max((p["shape_id"] for p in placements), default=-1)
    shape_freq = [0] * (max_shape_id + 1)
    for p in placements:
        shape_freq[p["shape_id"]] += 1

    # overlap count
    overlap_count = [0] * N
    for i in range(N):
        mi = masks[i]
        cnt = 0
        for j in range(N):
            if i != j and (mi & masks[j]):
                cnt += 1
        overlap_count[i] = cnt

    # order placements
    order = list(range(N))
    order.sort(
        key=lambda i: (
            shape_freq[placements[i]["shape_id"]],
            -overlap_count[i],
            placements[i]["shape_id"],
            -popcounts[i],
        )
    )

    # build board symmetry index maps
    def valid_transforms():
        trans = []
        trans.append(("id", lambda x, y: (x, y)))
        trans.append(("rot180", lambda x, y: (w - 1 - x, h - 1 - y)))
        trans.append(("flip_x", lambda x, y: (w - 1 - x, y)))
        trans.append(("flip_y", lambda x, y: (x, h - 1 - y)))
        if w == h:
            trans.append(("rot90", lambda x, y: (y, w - 1 - x)))
            trans.append(("rot270", lambda x, y: (h - 1 - y, x)))
            trans.append(("diag", lambda x, y: (y, x)))
            trans.append(("antidiag", lambda x, y: (h - 1 - y, w - 1 - x)))
        return trans

    trans_funcs = valid_transforms()
    maps = []
    for _, f in trans_funcs:
        mapping = [0] * total
        ok = True
        for idx in range(total):
            x = idx % w
            y = idx // w
            nx, ny = f(x, y)
            if not (0 <= nx < w and 0 <= ny < h):
                ok = False
                break
            mapping[idx] = ny * w + nx
        if ok:
            maps.append(mapping)

    maps = [tuple(m) for m in maps]

    # caching canonicalization
    @lru_cache(maxsize=None)
    def canonical_mask_cached(mask: int) -> int:
        best = None
        for mapping in maps:
            m = mask
            res = 0
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                res |= 1 << mapping[idx]
                m &= m - 1
            if best is None or res < best:
                best = res
        return best if best is not None else mask

    nodes_visited = 0
    time_up = False
    seen_canons = set()

    state_memo = {}
    pruned_by_memo = 0

    masks_local = masks
    placements_local = placements
    order_local = order
    total_local = total
    perf = perf_counter

    def dfs(i, sel, occ, used_shapes):
        nonlocal best_score, best_sel, best_occ, best_path, nodes_visited
        nonlocal time_up, pruned_by_memo

        if perf() - start_time > time_limit:
            time_up = True
            return

        nodes_visited += 1

        canon = canonical_mask_cached(occ)
        n_pieces = len(sel)
        state_key = (canon, n_pieces)

        if state_key in state_memo:
            if state_memo[state_key] <= best_score:
                pruned_by_memo += 1
                return

        score, _, _, path = score_selection(occ, w, h)

        if state_key not in state_memo or state_memo[state_key] < score:
            state_memo[state_key] = score

        if score > best_score:
            best_score = score
            best_sel = sel.copy()
            best_occ = occ
            best_path = path
            print(
                f"[brute] new best score={best_score} pieces={len(best_sel)}, nodes={nodes_visited}, "
                f"memo_prunes={pruned_by_memo} (t={perf()-start_time:.2f}s)"
            )
            try:
                render_maze_and_path_by_shape(
                    placements_local, best_sel, w, h, best_path, out_file=out, cell=cell
                )
            # trunk-ignore(bandit/B110)
            except Exception:
                pass

        blocked = occ.bit_count()
        empty_cells = total_local - blocked
        if empty_cells <= 1 or (empty_cells - 1) <= best_score:
            return

        if len(sel) >= max_pieces or i >= N:
            return

        if canon in seen_canons:
            return
        seen_canons.add(canon)

        for k in range(i, N):
            if time_up:
                break
            idx = order_local[k]
            pl = placements_local[idx]
            sid = pl["shape_id"]
            mask = masks_local[idx]

            if occ & mask:
                continue
            if no_repeat and ((used_shapes >> sid) & 1):
                continue

            sel.append(idx)
            dfs(k + 1, sel, occ | mask, used_shapes | (1 << sid))
            sel.pop()

            if time_up:
                break

    print(
        f"[brute] starting exhaustive search with memoization: placements={N} max_pieces={max_pieces} "
        f"time_limit={time_limit if time_limit!=float('inf') else 'Infinity'} s"
    )
    dfs(0, [], 0, 0)

    if time_up:
        print("[brute] stopped because time limit reached")
    print(
        f"[brute] finished best_score={best_score}, pieces={len(best_sel)} "
        f"elapsed={perf()-start_time:.2f}s nodes_visited={nodes_visited} "
        f"total_memo_prunes={pruned_by_memo}"
    )
    return best_score, best_sel, best_occ, best_path


# ---------------------- CLI ----------------------
def main() -> None:
    p = ArgumentParser()
    p.add_argument("--w", type=int, default=10)
    p.add_argument("--h", type=int, default=10)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max-pieces", type=int, default=8)
    p.add_argument("--allow-repeat", action="store_true")
    p.add_argument("--time-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=str, default="best_maze.png")
    p.add_argument("--cell", type=int, default=28)
    p.add_argument("--first-greedy", type=int, default=100)
    p.add_argument("--init-pos", type=int, default=None)
    p.add_argument("--init-selection", type=str, default=None)
    p.add_argument("--max-no-improve", type=int, default=10000)
    # brute-force control
    p.add_argument(
        "--bruteforce", action="store_true", help="force exhaustive bruteforce"
    )
    # NOVO: controle de paralelização
    p.add_argument("--parallel", action="store_true", help="enable parallel execution")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="number of parallel workers (default: cpu_count)",
    )
    args = p.parse_args()

    no_repeat = True if not args.allow_repeat else False

    def parse_init_selection(s: Optional[str]) -> Optional[List[int]]:
        if s is None:
            return None
        s = s.strip()
        if path.exists(s):
            try:
                import json

                j = json.load(open(s, "r"))
                if isinstance(j, dict) and "selection" in j:
                    return list(map(int, j["selection"]))
                if isinstance(j, list):
                    return list(map(int, j))
            except Exception as e:
                raise RuntimeError(f"Failed to load {s}: {e}") from e
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        return list(map(int, parts))

    init_selection = (
        parse_init_selection(args.init_selection) if args.init_selection else None
    )
    init_placement = args.init_pos if not init_selection else None

    print("Generating free polyominoes n =", args.n)
    shapes = generate_free_polyominoes(args.n, ominoes_dict)
    print("Found", len(shapes), "free shapes.")
    placements = build_global_placements(shapes, args.w, args.h)
    print("Total placements:", len(placements))

    # decide mode
    use_parallel = args.parallel
    n_workers = args.workers

    if args.bruteforce:
        if use_parallel:
            print(
                f"[mode] using PARALLEL bruteforce search (workers={n_workers if n_workers else 'auto'})"
            )
            best_score, best_sel, _, best_path = bruteforce_search_parallel(
                placements,
                args.w,
                args.h,
                max_pieces=args.max_pieces,
                no_repeat=no_repeat,
                time_limit=args.time_limit,
                out=args.out,
                cell=args.cell,
                n_workers=n_workers,
            )
        else:
            print("[mode] using sequential bruteforce search")
            best_score, best_sel, _, best_path = bruteforce_search(
                placements,
                args.w,
                args.h,
                max_pieces=args.max_pieces,
                no_repeat=no_repeat,
                time_limit=args.time_limit,
                out=args.out,
                cell=args.cell,
            )
    else:
        if use_parallel:
            print(
                f"[mode] using PARALLEL heuristic search (workers={n_workers if n_workers else 'auto'})"
            )
            best_score, best_sel, _, best_path = optimize_maze_parallel(
                placements,
                args.w,
                args.h,
                max_pieces=args.max_pieces,
                no_repeat=no_repeat,
                time_limit=args.time_limit,
                seed=args.seed,
                init_selection=init_selection,
                init_placement=init_placement,
                out=args.out,
                cell=args.cell,
                first_greedy=args.first_greedy,
                n_workers=n_workers,
                max_no_improve=(
                    args.max_no_improve if args.max_no_improve > 0 else float("inf")
                ),
            )
        else:
            print("[mode] using sequential heuristic search")
            best_score, best_sel, _, best_path = optimize_maze(
                placements,
                args.w,
                args.h,
                max_pieces=args.max_pieces,
                no_repeat=no_repeat,
                time_limit=args.time_limit,
                seed=args.seed,
                init_selection=init_selection,
                init_placement=init_placement,
                out=args.out,
                cell=args.cell,
                first_greedy=args.first_greedy,
            )

    print(
        "BEST diameter:",
        best_score,
        "pieces used:",
        0 if not best_sel else len(best_sel),
    )

    if best_sel:
        occ_check = 0
        for i in best_sel:
            if occ_check & placements[i]["mask"]:
                print(
                    "Warning: overlap detected in final selection! This should not happen."
                )
                break
            occ_check |= placements[i]["mask"]
        blocked_cells = occ_check.bit_count()
        total = args.w * args.h
        empty = total - blocked_cells
        print(f"Blocked cells: {blocked_cells} / {total}. Empty cells: {empty}")

    print("Rendering image to", args.out)
    outp = render_maze_and_path_by_shape(
        placements,
        best_sel,
        args.w,
        args.h,
        best_path,
        out_file=args.out,
        cell=args.cell,
    )
    print("Saved:", outp)


if __name__ == "__main__":
    main()
