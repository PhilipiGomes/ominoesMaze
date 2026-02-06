from typing import Dict, List, Set, Tuple, Optional
from polyominoes import ominoes_dict as od


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
    variants = []
    for reflect in (False, True):
        for rot_fn in _ROTATIONS:
            transformed = []
            if reflect:
                # aplicar reflexão em x antes da rotação
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


def normalize(cells: Set[Tuple[int, int]]) -> Tuple[int, int]:
    """Retorna a forma canônica mínima entre todas as variantes."""
    return min(_normalize_variants(cells))

def generate_free_polyominoes(
    n: int, ominoes_dict: Optional[Dict[int, List[Set[Tuple[int, int]]]]]
) -> List[Tuple[int, int]]:
    """
    Gera ominos livres (semântica igual ao seu código).
    - Usa poda por 'seen' com normalize.
    - evita `sorted(frontier)` (não necessário).
    - evita alocações extras dentro do laço.
    """
    if n <= 0:
        return []
    if ominoes_dict is not None and n in ominoes_dict:
        return [normalize(c) for c in ominoes_dict[n]]

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

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(
            "Usage: python create_ominoes.py <omino_size>"
        )
        sys.exit(1)
        
    n = sys.argv[1]

    res = generate_free_polyominoes(int(n), od)

    for om in res:
        if res.index(om) == 0:
            print(om, end=", \n")
        elif res.index(om) != len(res)-1:
            print("\t", om, end=", \n",sep='')
        else:
            print("\t", om, sep='')