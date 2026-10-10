import numpy as np
import math
import matplotlib.pyplot as plt

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev= set(_children)
        self._backward = lambda : None
        self._op = _op
        self.label = label
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), _op='+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __radd__(self, other):
        return self + other
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), _op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __rmul__(self, other):
        return self * other
    def __neg__(self):
        return self * (-1)
    def __sub__(self, other):
        return self + (-other)
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting ing/float powers for now"
        out = Value(self.data ** other, (self,), _op='**')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    def __truediv__(self, other):
        return self * (other ** -1)
    def exp(self):
        out = Value(math.exp(self.data), (self,), _op='exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward()
        return out
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), _op='tanh')
        # out = Value((self.exp() - (self**-1).exp) / (self.exp() + (self**-1).exp()))
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                topo.append(v)
                for prev in v._prev:
                    build_topo(prev)
        build_topo(self)
        self.grad = 1.0
        for node in topo:
            node._backward()
    
