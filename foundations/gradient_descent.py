class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        min = init
        for _ in range(iterations):
            der = 2*min
            min = min - learning_rate * der
        
        return round(min, 5)
        


    