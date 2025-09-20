import numpy as np

class Earth:
    """Earth model"""

    def __init__(self):
        self.Re = 6371386.8 / 0.3048    # Earth's radius
        self.g = 9.8                    # Gravity acceleration
        self.mu = 3.986e14 / 0.3048**3  # Gravity constant

    @staticmethod
    def rho(h):
        """Get density"""
        rho = 0.00238 * np.exp(-h / 24000)
        return rho

    @staticmethod
    def V_sound(h):
        """Get sound speed"""
        Vs = 8.99e-9 * h ** 2 - 9.16e-4 * h + 996
        return Vs

    @staticmethod
    def dynamic_pressure(h, V):
        """Get dynamic pressure"""
        rho = Earth.rho(h)
        q = 0.5 * rho * V ** 2
        return q