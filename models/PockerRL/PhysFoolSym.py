import math
from typing import Callable, List, Union

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Sym:
    def __init__(self, P, Particles):
        self.P = P
        self.Particles = Particles

class ScalarField2d:
    def __init__(self, a, b, func: Callable):
        self.a = a
        self.b = b
        self.func = func

    def scan_field(self, n=100) -> np.ndarray:
        x = np.linspace(0, self.a, n)
        y = np.linspace(0, self.b, n)

        X, Y = np.meshgrid(x, y)

        Z = self.func(X, Y)
        return Z

    def __call__(self, x) -> Union[float, np.ndarray]:
        x_array = np.asarray(x)
        result = self.func(x_array[0], x_array[1])
        return result.item() if x_array.ndim == 0 else result

def show_func_3d(func: Callable, a, b, n=100) -> None:
    x = np.linspace(0, a, n)
    y = np.linspace(0, b, n)

    X, Y = np.meshgrid(x, y)

    Z = func(X, Y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z = f(X, Y)')
    ax.set_title('3D-график функции f(x, y)')

    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.show()

if __name__ == "__main__":
    #func3 = lambda x, y: 10 - (func1(x, y)) * func2(x, y)
    func1 = lambda x, y: -0.5*y + 10
    func2 = lambda x, y: 0.5*np.cos(x*np.pi + np.pi/2)**2
    func3 = lambda x, y: np.where((0 < x) & (1 > x), 10 + (func1(x, -y)) * func2(x, y), 10 - (func1(x, y)) * func2(x, y))
    show_func_3d(func3, 4, 13)
    P = ScalarField2d(4, 13, func3)

    x = np.array([1, 1])
    print(P(x))
    print(P.scan_field(100))
    print(math.factorial(36)/math.factorial(6)/math.factorial(30) *
          math.factorial(30)/math.factorial(6)/math.factorial(24) *
          math.factorial(24)/math.factorial(6)/math.factorial(18))