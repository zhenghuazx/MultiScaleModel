import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp


class CellAggregatePopulation:
    def __init__(self, N0, t, xi0, alpha, M, Q=None):
        self.number_of_classes = N0.shape[0]
        # Kernels setup
        # self.beta = beta  # Daughter particle distribution
        # self.gamma = gamma  # Breakup frequency
        self.Q = Q  #
        # Uniform grid
        self.xi = xi0 + xi0 * np.arange(self.number_of_classes)
        self.delta_xi = xi0
        # Solve procedure
        self.alpha = alpha
        self.M = M
        self.N = solve_ivp(self.RHS, [t[0], t[-1]], N0, t_eval=t, method='RK45')
        # odeint(lambda NN, t: self.RHS(NN, t), N0, t)

    def RHS(
            self, t, N
    ):
        dNdt = np.zeros(self.number_of_classes)
        #
        # if self.gamma is not None and self.beta is not None:
        #     # Death breakup term
        #     dNdt -= N * self.gamma(self.xi)
        #     # Birth breakup term
        #     for i in np.arange(self.number_of_classes):
        #         for j in np.arange(i + 1, self.number_of_classes):
        #             dNdt[i] += \
        #                 self.beta(self.xi[i], self.xi[j]) \
        #                 * self.gamma(self.xi[j]) \
        #                 * N[j] * self.delta_xi

        for i in np.arange(1, self.number_of_classes - 1):
            xi_plus = self.xi[i] + 0.5 * self.delta_xi
            xi_minus = self.xi[i] - 0.5 * self.delta_xi
            dNdt[i] -= self.alpha / 2 / self.delta_xi * ((N[i + 1] + N[i]) * np.log(self.M / xi_plus) * xi_plus -
                                                         (N[i] + N[i-1]) * np.log(self.M / xi_minus) * xi_minus)


        if self.Q is not None:
            for i in np.arange(self.number_of_classes):
                # Birth coalescence term
                for j in np.arange(0, i):
                    dNdt[i] += 0.5 * N[i - j] * N[j] \
                               * self.Q(self.xi[j], self.xi[i - j])  # * self.delta_xi
                # Death coalescence term
                for j in np.arange(self.number_of_classes):
                    dNdt[i] -= N[i] * N[j] * self.Q(self.xi[i], self.xi[j])  # * self.delta_xi
        return dNdt


def set_init_distribution(f, N0, v0):
    interval = float(v0) / len(N0)
    for i in range(len(N0)):
        val = i * interval
        if f.x[0] <= val <= f.x[-1]:
            N0[i] = max(f(val), 0)


if __name__ == '__main__':
    import pandas as pd
    from scipy import interpolate

    import matplotlib.pyplot as plt
    from scipy.stats import norm, lognorm
    ini_dist = pd.read_csv('multi_scale_model/agg_distribution.csv')
    ini_dist.columns = columns = ['radius', 'n']
    ini_dist['radius'] = ini_dist.apply(lambda x: np.exp(x["radius"]), axis=1)
    f = interpolate.interp1d(ini_dist['radius'], ini_dist['n'], 'quadratic')

    v0 = 1
    time = np.arange(0.0, 100.0, 0.1)
    grids = 200  # [10, 20, 40, 80, 160]
    max_radius = 1000
    # grids = [1]
    k = 1.26 * 10E-3
    k_1 = 1.94 * 10E-4
    a = 8.06 * 10E-1
    results = dict()
    # N0 = np.zeros(grids)
    N0 = lognorm(np.arange(0, max_radius + 0.1, 200), 2.15 + np.log(7.5), 0.3)
    set_init_distribution(f, N0, v0)
    print(N0)
    results[grids] = CellAggregatePopulation(
        N0,
        time,
        v0 / grids,
        1.72 * 10E-3,
        9.71 * 10E3,
        Q=lambda x, y: k * np.exp(-k_1 * ((x + y) / 2) ** a) * ((x ** (1 / 3.0) + y ** (1 / 3.0)) ** (7 / 3))
    )


    data = results[600].N.y[:,100]
    plt.plot(np.arange(0, 600, v0 / g), data / np.sum(data))
    plt.show()

    data = results[100].N.y[:,100]
    plt.plot(np.arange(0, 1000, v0 / g), data)
    plt.show()

    data = results[100].N.y[-1,]
    plt.plot(np.arange(0, 1000, v0 / g), data)
    plt.show()

    x = np.arange(0, 4, 0.1)
    x1 = np.exp(x)
    plt.plot(x1, lognorm.pdf(x, 3 + np.log(7.5), 0.1))
    plt.show()

    plt.plot(x1, lognorm.pdf(x, 3, 0.4))


    for i in [0, 24, 48, 72, 96]:
        data = agg_density_function.N.y[:, i*10]
        plt.plot(np.arange(0, 1500, 1), data)
        plt.show()
        # plt.plot(np.arange(0, 1500, 1), data / np.sum(data))
        # plt.show()


