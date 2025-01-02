import numpy as np

class Value:
    def __init__(self,data):
        self.data=data

    def __repr__(self):
        return "Value : "+str(self.data)

    def __add__(self,other):
        res=Value(self.data+other.data)
        return res

    def __mul__(self,other):
        res=Value(self.data*other.data)
        return res
