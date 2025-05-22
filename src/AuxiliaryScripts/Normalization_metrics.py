import numpy as np



class Normalization_Techniques():
    
    def __init__(self, args):
        self.args = args

    def interp(self, min_value, max_value, value):
        return (((value - min_value)/(max_value - min_value + 1e-6)))
    