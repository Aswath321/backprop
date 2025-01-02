import numpy as np

class Value:
    def __init__(self,data,children=(),op='',label=''):
        self.data=data
        self.prev=set(children)
        self.label=label
        self.op=op
        self.grad=0
        self.backward=lambda:None

    def __repr__(self):
        return "Value : "+str(self.data)


    def __add__(self,other):
        res=Value(self.data+other.data,(self,other),'+')
        def backward():
          self.grad+=1*res.grad
          other.grad+=1*res.grad
        res.backward=backward
        return res

    def __mul__(self,other):
        res=Value(self.data*other.data,(self,other),'*')
        def backward():
          self.grad+=other.data*res.grad
          other.grad+=self.data*res.grad
        res.backward=backward
        return res

from graphviz import Digraph
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad %.4f  }" % (n.label,n.data,n.grad), shape='record')
        if n.op:
            dot.node(name=str(id(n)) + n.op, label=n.op)
            dot.edge(str(id(n)) + n.op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    return dot
