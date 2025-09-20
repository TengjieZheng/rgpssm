
import torch

class Sys():
    def __init__(self, dt=0.01, std_noise=0.01):
        self.x = torch.ones((1, 1))         # system state
        self.theta = -torch.ones((2, 1))    # model parameters
        self.dt = dt
        self.t = 0.
        self.sigma_noise = std_noise

        self.xd = torch.eye(1) * 2.         # desired state
        self.t_last = -1000.
        self.Delta_t = 1

    def update(self):

        self.xd_update()        # update desired state
        self.u_update()         # update control input
        self.theta_update()     # update model parameters

        # update system state
        self.dot_x = self.dyn()
        self.x = self.x + self.dt * self.dot_x
        self.t = self.t + self.dt
        self.y = self.x + torch.randn(1) * self.sigma_noise # measurement

    def dyn(self):
        """Dynamics"""
        return self.theta[0, 0] * self.x * 1. + self.theta[1, 0] + self.u

    def theta_update(self):
        omg = 1
        self.theta[0, 0] = torch.cos(omg * self.t * torch.ones(1))
        self.theta[1, 0] = torch.cos(0.2 * omg * self.t * torch.ones(1))

    def xd_update(self):
        if self.t -self.t_last > self.Delta_t:
            self.t_last = self.t
            self.xd = -self.xd
            self.Delta_t = self.Delta_t + 1 if self.Delta_t < 4 else 4

    def u_update(self):
        e = self.xd - self.x
        self.u = e * 20.