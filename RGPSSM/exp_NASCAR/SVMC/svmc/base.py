"""
Abstract classes
"""
from abc import ABCMeta, abstractmethod

import torch
from torch.nn import Module


class Filter(Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def filter(self, *args):
        pass

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Dynamics(Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self):
        """Sample a particle passing through the dynamical model"""
        pass

    @abstractmethod
    def log_weight(self, prev, curr):
        """
        :param prev: previous state
        :param curr: current state
        :return:
        """
        pass

    def forward(self, prev, curr):
        return self.log_weight(prev, curr)

class KnowledgeModel(metaclass=ABCMeta):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

    @abstractmethod
    def fun_tran(self, x, u, f):
        """State transition function
        Args:
            x : system state
            u : system input
            f : GP prediction
        Returns:
            x_pre : state prediction
            Af : partial derivative dx_pre / df
        """
        pass

    @abstractmethod
    def fun_input(self, x, u):
        """Input function
        Args:
            x : system state
            u : system input
        Returns:
            z : GP input
        """
        pass