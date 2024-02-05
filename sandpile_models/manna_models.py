import numba as nb
import networkx as nx
import numpy as np
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)



class _MannaBaseClass:
    def __init__(
        self, p_boundry: list, length: int, dimension: int, periodic: bool,
        f_random: float, alpha: float, p_fill: float,
    ) -> None:
        self._length = length
        self._dimension = dimension
        size = length ** dimension
        self._size = size
        self._periodic = periodic
        self._f_random = f_random
        self._alpha = alpha
        self._p_fill = p_fill
        
        self.heights = np.random.choice(
            a=(0, 1),
            size=size,
            p=(1 - p_fill, p_fill)
        ).astype('int8')
        _graph = nx.grid_graph(
            [length] * dimension, periodic
        )
        _graph = nx.relabel_nodes(
            G=_graph,
            mapping={
                node: np.sum(node * np.power(
                    length, np.arange(dimension - 1, -1, -1)
                ))
                for node in _graph.nodes
            },
            copy=True
        )
        
        extra_links = int(alpha * size)
        while extra_links > 0:
            node_1, node_2 = np.random.randint(0, size, size=2)
            if not _graph.has_edge(node_1, node_2) and node_1 != node_2:
                _graph.add_edge(node_1, node_2)
                extra_links -= 1
                   
        nx.connected_double_edge_swap(
            _graph, nswap=int(f_random * (dimension + alpha) * size),
        )
        
        neighbors = np.array([
            neigh
            for node in range(size)
            for neigh in nx.neighbors(_graph, node)
        ])
        
        neighbors_len = np.array([
            _graph.degree(node)
            for node in range(size)
        ])

        neighbors_pos = (np.concatenate([
            [0],
            neighbors_len.cumsum()[:-1]
        ])).astype('int')
        
        @nb.njit
        def _nb_drive(heights, steps: int, seed: int):
            if seed >= 0:
                np.random.seed(seed)
            
            size_s = np.zeros(steps)
            area_s = np.zeros(steps)
            time_s = np.zeros(steps)
            is_visisted = np.zeros(size).astype('bool')
            
            for step in range(steps):
                node = np.random.randint(size)
                heights[node] += 1
                avalanche_size = 0
                is_visisted[:] = False
                time = 0
                unstable_points = np.where(heights > 1)[0]
                unstable_heights = heights[unstable_points]
                while len(unstable_points):
                    avalanche_size += len(unstable_points)
                    is_visisted[unstable_points] = True
                    time += 1
                    for cell, cell_height in zip(unstable_points, unstable_heights):
                        p = p_boundry[cell]
                        for _ in range(cell_height):
                            if p == 0 or np.random.rand() > p:
                                neigh = neighbors[
                                    neighbors_pos[cell] + np.random.randint(
                                        neighbors_len[cell]
                                    )
                                ]
                                heights[neigh] += 1
                        heights[cell] -= cell_height

                    unstable_points = np.where(heights > 1)[0]
                    unstable_heights = heights[unstable_points]
                size_s[step] = avalanche_size
                area_s[step] = np.sum(is_visisted)
                time_s[step] = time
            return size_s, area_s, time_s
        self._drive = _nb_drive
        del _graph, neighbors, neighbors_len, neighbors_pos
        
    def drive(self, steps: int, seed: int = -1):
        return self._drive(self.heights, steps, seed)
    
    @property
    def length(self):
        return self._length
    
    @property
    def dimension(self):
        return self._dimension
    
    @property
    def size(self):
        return self._size
    
    @property
    def is_periodic(self):
        return self._periodic
    
    @property
    def f_random(self):
        return self._f_random
    
    @property
    def alpha(self):
        return self._alpha


class Manna(_MannaBaseClass):
    def __init__(
        self, length: int, dimension: int = 2, f_random: float = 0.,
        alpha: float = 0., p_fill: float = 0.,
    ) -> None:
        p_boundry = np.zeros([length] * dimension)
        for d in range(dimension):
            p_boundry[*[
                0 if i == d else slice(0, length)
                for i in range(dimension)
            ]] += 1
            p_boundry[*[
                length - 1 if i == d else slice(0, length)
                for i in range(dimension)
            ]] += 1
        p_boundry /= (2 * dimension)
        p_boundry = p_boundry.flatten()
        super().__init__(
            p_boundry=p_boundry, length=length, dimension=dimension, periodic=False,
            f_random=f_random, alpha=alpha, p_fill=p_fill,
        )

    
class BulkDissipativeManna(_MannaBaseClass):
    def __init__(
        self, p_diss: float, length: int, dimension: int = 2, periodic: bool = False,
        f_random: float = 0., alpha: float = 0., p_fill: float = 0.
    ) -> None:
        self._p_diss = p_diss
        p_boundry = np.ones([length] * dimension) * p_diss
        p_boundry = p_boundry.flatten()
        super().__init__(
            p_boundry=p_boundry, length=length, dimension=dimension, periodic=periodic,
            f_random=f_random, alpha=alpha, p_fill=p_fill,
        )
        
    @property
    def p_diss(self):
        return self._p_diss
        
        
class BoundryDissipativeManna(_MannaBaseClass):
    def __init__(
        self, p_diss: float, length: int, dimension: int = 2, periodic: bool = False,
        f_random: float = 0., alpha: float = 0., p_fill: float = 0.
    ) -> None:
        self._p_diss = p_diss
        p_boundry = np.ones([length] * dimension) * p_diss
        p_boundry[*[slice(1, length - 1)] * dimension] = 0
        p_boundry = p_boundry.flatten()
        super().__init__(
            p_boundry=p_boundry, length=length, dimension=dimension, periodic=periodic,
            f_random=f_random, alpha=alpha, p_fill=p_fill,
        )
        
    @property
    def p_diss(self):
        return self._p_diss
