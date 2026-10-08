import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        exp_z = np.exp(z - np.max(z))
        sm = exp_z / np.sum(exp_z)
        return np.round(sm, 4)
        
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        pass
