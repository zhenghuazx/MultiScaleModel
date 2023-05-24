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
        self.t = t
        # Uniform grid
        self.xi = xi0 + xi0 * np.arange(self.number_of_classes)
        self.delta_xi = xi0
        # Solve procedure
        self.alpha = alpha
        self.M = M
        self.N = solve_ivp(self.RHS, [t[0], t[-1]], N0, t_eval=t, method='BDF')
        # odeint(lambda NN, t: self.RHS(NN, t), N0, t)

    def RHS(
            self, t, N
    ):
        dNdt = np.zeros(self.number_of_classes)

        for i in np.arange(1, self.number_of_classes - 1):
            xi_plus = self.xi[i] + 0.5 * self.delta_xi
            xi_minus = self.xi[i] - 0.5 * self.delta_xi
            dNdt[i] -= self.alpha / 2 / self.delta_xi * ((N[i + 1] + N[i]) * np.log(self.M / xi_plus) * xi_plus -
                                                         (N[i] + N[i - 1]) * np.log(self.M / xi_minus) * xi_minus)

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
        # if f.x[0] <= val <= f.x[-1]:
        #     N0[i] = max(f(val), 0)
        if val == 0:
            N0[i] = 0
        else:
            N0[i] = max(f(val), 0)


def lognorm(x, mu, sigma):
    pdf = (np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))
           / (x * sigma * np.sqrt(2 * np.pi)))
    return pdf


if __name__ == '__main__':
    import pandas as pd
    from scipy import interpolate
    from scipy.stats import norm

    ini_dist = pd.read_csv('multi_scale_model/agg_distribution.csv')
    ini_dist.columns = columns = ['radius', 'n']
    ini_dist['radius'] = ini_dist.apply(lambda x: np.exp(x["radius"]), axis=1)

    # f = lambda x: norm.pdf(x, 1.8, 0.3)
    f = lambda x: lognorm(x, 2.15 + np.log(7.5), 0.27)

    v0 = 1500.0
    grid = 1500
    hour = 144
    t_step = 0.1
    time = np.arange(0.0, hour, t_step)

    grids = [grid]  # [10, 20, 40, 80, 160]

    # grids = [1]
    k = 1.26 * 10E-3
    k_1 = 1.94 * 10E-4
    a = 8.06 * 10E-1
    results = dict()
    for g in grids:
        N0 = np.zeros(g)
        set_init_distribution(f, N0, v0)
        # print(N0)
        results[g] = CellAggregatePopulation(
            N0,
            time,
            v0 / g,
            2.02 * 10E-4,
            9.71 * 10E3,
            Q=lambda x, y: k * np.exp(-k_1 * ((x + y) / 2) ** a) * ((x ** (1 / 3.0) + y ** (1 / 3.0)) ** (7 / 3))
        )

    import matplotlib.pyplot as plt

    ''' true data '''
    D1 = pd.read_csv('multi_scale_model/data/wu2014D1.tsv', sep='\t')
    D2 = pd.read_csv('multi_scale_model/data/wu2014D2.tsv', sep='\t')
    D2.loc[D2['Y'] < 0, 'Y'] = 0
    D3 = pd.read_csv('multi_scale_model/data/wu2014D3.tsv', sep='\t')
    D4 = pd.read_csv('multi_scale_model/data/wu2014D4.tsv', sep='\t')
    experiment_data = [D1, D2, D3, D4]

    # import dill as pickle
    # try:
    #     with open('multi_scale_model/data/case_final.pkl', 'rb') as f:
    #         agg_density_function = pickle.load(f)
    # except:
    #     with open('data/case_final.pkl', 'rb') as f:
    #         agg_density_function = pickle.load(f)

    if len(results) > 0:
        agg_density_function = results[1500]
    ''' predicted data'''
    # agg_density_function = results[grid]
    plt.rcParams.update({'font.size': 16})
    for i, h in enumerate([0, 240, 480, 720]):
        data = agg_density_function.N.y[:, h]
        if i > 0:
            y = data / np.sum(data)
        else:
            y = data / np.sum(data)
        plt.plot(np.arange(0, 600, 1), y[:600], lw=3)
        f_interp1 = interpolate.interp1d(np.exp(experiment_data[i]['X']) * 7.5,
                                         experiment_data[i]['Y'] / np.sum(experiment_data[i]['Y']), 'quadratic',
                                         bounds_error=False, fill_value=(0, 0))
        x_interp = np.arange(0, 600, 1)
        y_interp = f_interp1(x_interp)
        y_interp /= np.sum(y_interp)
        plt.plot(x_interp[::5], y_interp[::5], linestyle='-', marker='o', markersize=6, lw=2)
        plt.xlabel(r'Radius ($\mu$m)')
        plt.ylabel('Densitiy')
        plt.savefig("multi_scale_model/result/PBM/radius-600-hour-{}.pdf".format(int(h / 10)), bbox_inches='tight')
        plt.savefig("multi_scale_model/result/PBM/radius-600-hour-{}.svg".format(int(h / 10)), bbox_inches='tight')
        plt.savefig("multi_scale_model/result/PBM/radius-600-hour-{}.png".format(int(h / 10)), bbox_inches='tight')
        plt.savefig("multi_scale_model/result/PBM/radius-600-hour-{}.eps".format(int(h / 10)), bbox_inches='tight')
        plt.show()

    # np.save('multi-scale model/data/case1.npy', results[g].N)
    # import pickle
    import dill as pickle

    with open('multi_scale_model/data/case_final.pkl', 'wb') as file:
        pickle.dump(results[g], file, pickle.HIGHEST_PROTOCOL)

    # distr = data/ np.sum(data)
    # mean_val = np.sum(np.arange(0, 1500, v0 / g) * distr)  # mean
    # variance = (np.sum(np.arange(0, 1500, v0 / g) ** 2 * distr) - mean_val**2)
    # sigma2 = np.log(variance/ mean_val ** 2 + 1)
    # mu = np.log(mean_val) - sigma2 / 2
    #
    # plt.plot(x1, lognorm.pdf(x, 3, 0.4))
    #

    ''' '''
    mean_results = []
    for i in range(1000):
        data = agg_density_function.N.y[:, i]
        distr = data / np.sum(data)
        mean_val = np.sum(np.arange(0, 1500, v0 / g) * distr)  # mean
        mean_results.append(mean_val)
    plt.plot(range(100), mean_results[::10], lw=3, label='Aggregate Size Model')
    experiment_aggregate_mean = [
        np.sum(np.exp(experiment_data[i]['X']) * 7.5 * experiment_data[i]['Y'] / np.sum(experiment_data[i]['Y'])) for i
        in
        range(4)]
    # plt.plot([0, 24, 48, 72, 96], np.array(experiment_aggregate_mean + [205]) * 1.2, lw=2, markerfacecolor='none')
    plt.plot([0, 24, 48, 72, 96], [70, 100, 143, 200, 255], linestyle='-', marker='o', markersize=10, lw=3,
             label='Experiments')
    # plt.scatter([0, 24, 48, 72, 96], [70, 100, 143, 200, 255], '-p', s=10,  label=None)
    plt.xlabel('Time (hr)')
    plt.ylabel(r'Aggregate radius ($\mu$m)')
    plt.legend()
    # plt.savefig("multi_scale_model/result/PBM/mean_curve.pdf", bbox_inches='tight')
    # plt.savefig("multi_scale_model/result/PBM/mean_curve.svg", bbox_inches='tight')
    # plt.savefig("multi_scale_model/result/PBM/mean_curve.png", bbox_inches='tight')
    # plt.savefig("multi_scale_model/result/PBM/mean_curve.eps", bbox_inches='tight')
    plt.show()
