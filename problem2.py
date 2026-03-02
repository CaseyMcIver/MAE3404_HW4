import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def f(x):
    """
    Function for Problem 2.
    (Change this ONLY if your HW specifies a different equation.)
    """
    return np.exp(-x**2)


def main():

    print("Problem 2: Numerical Integration")

    a = float(input("Enter lower bound a [0]: ") or 0)
    b = float(input("Enter upper bound b [2]: ") or 2)


    integral, error = quad(f, a, b)

    print("\nIntegral result:")
    print(f"Integral = {integral:.6f}")
    print(f"Estimated error = {error:.6e}")


    x = np.linspace(a, b, 400)
    y = f(x)

    plt.figure()
    plt.plot(x, y)
    plt.title("Function Plot")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()