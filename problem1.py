import numpy as np
from math import log, exp, sqrt, pi
from scipy.integrate import quad
from scipy.optimize import fsolve


def get_float(prompt: str, default: float) -> float:
    """Press Enter to accept the default."""
    s = input(f"{prompt} [{default}]: ").strip()
    return default if s == "" else float(s)


def lognormal_pdf(D: float, mu: float, sigma: float) -> float:
    """Lognormal PDF f(D) for D > 0."""
    if D <= 0:
        return 0.0
    return (1.0 / (D * sigma * sqrt(2.0 * pi))) * exp(-((log(D) - mu) ** 2) / (2.0 * sigma ** 2))


def base_cdf(D: float, mu: float, sigma: float) -> float:
    """Base CDF F(D) = ∫_0^D f(t) dt computed using quad."""
    if D <= 0:
        return 0.0
    val, _ = quad(lambda t: lognormal_pdf(t, mu, sigma), 0.0, D)
    return float(val)


def truncated_cdf(D: float, mu: float, sigma: float, Dmin: float, Dmax: float, Fmin: float, Fmax: float) -> float:
    """
    Truncated CDF on [Dmin, Dmax]:
      F_trunc(D) = (F(D) - F(Dmin)) / (F(Dmax) - F(Dmin))
    """
    if D <= Dmin:
        return 0.0
    if D >= Dmax:
        return 1.0
    FD = base_cdf(D, mu, sigma)
    return (FD - Fmin) / (Fmax - Fmin)


def sample_one(mu: float, sigma: float, Dmin: float, Dmax: float, Fmin: float, Fmax: float) -> float:
    """Draw one sample from the truncated lognormal using inverse CDF with fsolve."""
    u = float(np.random.rand())

    def equation(D_arr):
        # fsolve passes an array even for 1 variable
        D = float(D_arr[0])
        return truncated_cdf(D, mu, sigma, Dmin, Dmax, Fmin, Fmax) - u


    guesses = [
        0.5 * (Dmin + Dmax),
        Dmin + 0.1 * (Dmax - Dmin),
        Dmin + 0.9 * (Dmax - Dmin),
    ]

    for g in guesses:
        sol, info, ier, _msg = fsolve(equation, x0=[g], full_output=True)
        if ier == 1:
            D = float(sol[0])

            if D < Dmin:
                D = Dmin
            if D > Dmax:
                D = Dmax
            return D


    return 0.5 * (Dmin + Dmax)


def main():
    print("Problem 1: Truncated Lognormal Distribution (quad + fsolve)")


    mu = get_float("Enter mu", 0.0)
    sigma = get_float("Enter sigma", 1.0)
    Dmin = get_float("Enter Dmin", 0.5)
    Dmax = get_float("Enter Dmax", 5.0)

    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if Dmin <= 0 or Dmax <= 0 or Dmax <= Dmin:
        raise ValueError("Need 0 < Dmin < Dmax")

    print("\nInputs used:")
    print(f"mu={mu}, sigma={sigma}, Dmin={Dmin}, Dmax={Dmax}")

    # compute CDF values using quad
    Fmin = base_cdf(Dmin, mu, sigma)
    Fmax = base_cdf(Dmax, mu, sigma)

    print("\nBase CDF values (using quad):")
    print(f"F(Dmin) = {Fmin:.8f}")
    print(f"F(Dmax) = {Fmax:.8f}")
    print(f"Normalization (Fmax - Fmin) = {Fmax - Fmin:.8f}")

    # 11 samples, each N=100
    num_samples = 11
    N = 100

    sample_means = []
    sample_vars = []

    print("\nSample results (each sample has N=100 rocks):")
    for k in range(1, num_samples + 1):
        rocks = np.array([sample_one(mu, sigma, Dmin, Dmax, Fmin, Fmax) for _ in range(N)])

        Dbar = float(np.mean(rocks))
        S2 = float(np.var(rocks, ddof=1))

        sample_means.append(Dbar)
        sample_vars.append(S2)

        print(f"Sample {k:02d}: mean(D̅) = {Dbar:.6f}, variance(S^2) = {S2:.6f}")

    sample_means = np.array(sample_means)

    mean_of_means = float(np.mean(sample_means))
    var_of_means = float(np.var(sample_means, ddof=1))

    print("\nSampling mean statistics (across the 11 sample means):")
    print(f"Mean of sample means = {mean_of_means:.6f}")
    print(f"Variance of sample means = {var_of_means:.6f}")


if __name__ == "__main__":
    main()