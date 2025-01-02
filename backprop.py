import numpy as np
from graphviz import Digraph

class Value:
    def __init__(self,data,children=(),op='',label=''):
        self.data=data
        self.prev=set(children)
        self.label=label
        self.op=op
        self.grad=0.0
        self.backward=lambda:None

    def __repr__(self):
        return "Value : "+str(self.data)

    def __truediv__(self,other):
        return self*(other**-1)

    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other * self**-1

    def __add__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        res=Value(self.data+other.data,(self,other),'+')
        def backward():
          self.grad+=1*res.grad
          other.grad+=1*res.grad
        res.backward=backward
        return res
    
    def __radd__(self,other):
        return self+other

    def __sub__(self,other):
        return self+(-other)
    
    def __neg__(self):
        return self*-1

    def __mul__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        res=Value(self.data*other.data,(self,other),'*')
        def backward():
          self.grad+=other.data*res.grad
          other.grad+=self.data*res.grad
        res.backward=backward
        return res

    def __rmul__(self,other):
        return self*other

    def __pow__(self,other):
        assert isinstance(other,(int,float)),"only int,float"
        res=Value(self.data**other,(self,),'power')
        def backward():
          self.grad+=other*(self.data**(other-1))*res.grad
        res.backward=backward
        return res

    def tanh(self):
        x=self.data
        t=(np.exp(2*x)-1)/(np.exp(2*x)+1)
        res=Value(t,(self,),'tanh')
        def backward():
            self.grad=(1-(t**2))*res.grad
        res.backward=backward
        return res

    def exp(self):
        x=self.data
        res=Value(np.exp(x),(self,),'exp')
        def backward():
          self.grad+=res.data*res.grad
        res.backward=backward
        return res

    def backward_full(self):
        topo=[]
        visited=set()
        def build_topo(v):
          if v not in visited:
            visited.add(v)
            for child in v.prev:
              build_topo(child)
            topo.append(v)
        build_topo(self)

        self.grad=1
        for node in reversed(topo):
          node.backward()
              

    

