import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement this method
        self.state_low = state_low
        self.state_high = state_high
        self.num_tilings = num_tilings
        self.tile_width = tile_width

        self.num_tiles = np.ceil((state_high - state_low) / tile_width).astype(int) + 1
        self.offsets = (np.arange(num_tilings).reshape(-1, 1) / num_tilings) * tile_width
        self.weights = np.zeros((num_tilings, *self.num_tiles))

    def get_tile_indices(self, s):
        tile_indices = []
        for tiling in range(self.num_tilings):
            offset = self.offsets[tiling]
            indices = ((s - self.state_low + offset) // self.tile_width).astype(int)
            tile_indices.append(indices)
        return np.array(tile_indices)
    
    def __call__(self,s):
        # TODO: implement this method
        tile_indices = self.get_tile_indices(s)
        tile_values = [self.weights[tiling][tuple(indices)] for tiling, indices in enumerate(tile_indices)]
        return np.sum(tile_values)

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        tile_indices = self.get_tile_indices(s_tau)
        tile_values = [self.weights[tiling][tuple(indices)] for tiling, indices in enumerate(tile_indices)]
        v_hat = np.sum(tile_values)
        delta = alpha * (G - v_hat) / self.num_tilings
        for tiling, indices in enumerate(tile_indices):
            self.weights[tiling][tuple(indices)] += delta
