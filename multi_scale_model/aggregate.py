import numpy as np
from multi_scale_model.single_cell_model import CellModel
from multi_scale_model.Util import meta_label_index


class Aggregate:
    def __init__(self, cell_growth_rate, agg_density_function, meta_index, ini_intra_metabolites,
                 single_cell_radius, parameters, diffusion_coef, porosity, delta_t=0.5, tortuosity=1.46,
                 deterministic=False):
        self.agg_density = agg_density_function
        self.variance_metabolite = []
        self.variance_metabolite_ = []
        self.mean_metabolites = []
        self.porosity = porosity
        self.tortuosity = tortuosity
        self.delta_t = delta_t
        self.R0 = single_cell_radius
        self.extra_metabolites = np.array(ini_intra_metabolites[meta_index])
        self.meta_index = meta_index
        self.aggregates = []
        self.level_size = int(single_cell_radius * 2)
        self.health = []
        self.t = 0
        self.unhealth_data = []
        self.all_metabolites = np.array(ini_intra_metabolites)
        self.results = [np.array(ini_intra_metabolites)]
        self.cell_density = ini_intra_metabolites[-1] * 1000  # 10^6 cells/L
        self.mu = cell_growth_rate  # 0.04531871789155581
        self.deterministic = deterministic
        for i in range(int(agg_density_function.number_of_classes / self.level_size)):
            self.aggregates.append(
                CellModel(
                    single_cell_radius * 2 * (i + 1),
                    ini_intra_metabolites,
                    int(single_cell_radius * 2),
                    parameters,
                    np.array(diffusion_coef) * self.porosity / self.tortuosity,
                    self.meta_index,
                    self.delta_t,
                    deterministic=self.deterministic
                )
            )

    def _get_total_num_cell_for_given_aggregate(self, R_levels):
        if len(R_levels) == 1:
            return (1 - self.porosity) * np.array([(R_levels[0] ** 3) / self.R0 ** 3])

        temp = []
        for n in range(len(R_levels)):
            if n == 0:
                temp.append((R_levels[0] ** 3) / self.R0 ** 3)
            else:
                temp.append((R_levels[n] ** 3 - R_levels[n - 1] ** 3) / self.R0 ** 3)
        return (1 - self.porosity) * np.array(temp)  # * (np.exp(-self.t) + 2) / 3

    def _get_num_aggregate(self, t, cell_density):
        time_ind = int(t / (self.agg_density.N.t[1] - self.agg_density.N.t[0]))
        distr = self.agg_density.N.y[:, time_ind]
        mean_R = np.sum(self.agg_density.xi * distr)
        return cell_density * self.R0 ** 3 / mean_R ** 3

    def _get_num_aggregate_of_size_l(self, l, t, cell_density):
        # R_index = R_l / self.agg_density.delta_xi
        time_ind = int(t / (self.agg_density.N.t[1] - self.agg_density.N.t[0]))
        M_t = self._get_num_aggregate(t, cell_density)
        start = l * self.level_size
        end = (l + 1) * self.level_size
        return M_t * np.sum(self.agg_density.N.y[start:end, time_ind])

    def get_num_cell_in_ring(self, l, t, cell_density):
        # R_l = self.get_aggregate_size(l)
        cell_num = self._get_total_num_cell_for_given_aggregate(self.aggregates[l].radii)
        # print(self.aggregates[l].radii)
        M_l = self._get_num_aggregate_of_size_l(l, t, cell_density)
        # print(M_l, cell_num)
        return M_l * cell_num

    def get_aggregate_size(self, l):
        return self.agg_density.xi[(l + 1) * self.level_size - 1] + 0.01

    def get_cell_density_for_each_aggregrate_l(self, l, t, cell_density):
        G_l = self.get_num_cell_in_ring(l, t, cell_density)
        G_l = G_l[1:]
        G_l[G_l < 0] = 0  # remove the computation error
        return np.sum(G_l)

    def simulate_single_agg_group(self, l, t, condition, cell_density, vcd_adjustment, cal_variance):
        # R_l = self.get_aggregate_size(l)
        flux_rates, metabolite_conc = self.aggregates[l].get_flux_rate_cell_level(condition)
        G_l = self.get_num_cell_in_ring(l, t, cell_density)
        G_l = G_l[1:]
        _G_l = G_l.copy()
        G_l[G_l < 0] = 0  # remove the computation error
        G_l *= vcd_adjustment
        # quantify unhealthy cells
        unhealthy_cell_l = 0
        total_cell_l = 0
        for k, meta_conc in enumerate(metabolite_conc):
            # total_cell_l += G_l[k]
            if (meta_conc[meta_label_index['GLC'][1]] <= 2.5) or (meta_conc[meta_label_index['ELAC'][1]] > 40):
                unhealthy_cell_l += _G_l[k]  # glucose index is 3 in extracellular metabolite list
                # self.unhealth_data.append([t, l, k, G_l[k], meta_conc[3], meta_conc[6]])
            total_cell_l = np.sum(G_l)
        delta_metabolites = self.aggregates[l].update_intracellular_conditions(flux_rates, self.delta_t, G_l,
                                                                               self.cell_density)
        variance_metabolite = []
        variance_metabolite_ = []
        sample_means = []
        if cal_variance:
            variance_metabolite, variance_metabolite_, sample_means = self.aggregates[l].get_covariance(flux_rates,
                                                                                                        self.delta_t,
                                                                                                        G_l)
        return delta_metabolites, unhealthy_cell_l, total_cell_l, variance_metabolite, variance_metabolite_, sample_means

    def simulate(self, validation=False, cal_variance=False):
        if validation:
            self.cell_density = cell_density_cal(self.t)
        else:
            self.cell_density += self.mu * self.delta_t * self.cell_density

        delta_meta, vcd_adjustment = 0, 0
        for l in range(int(self.agg_density.number_of_classes / self.level_size)):
            vcd_adjustment += self.get_cell_density_for_each_aggregrate_l(l, self.t, self.cell_density)
        vcd_adjustment = self.cell_density / vcd_adjustment

        health_l = [] * int(self.agg_density.number_of_classes / self.level_size)
        for l in range(int(self.agg_density.number_of_classes / self.level_size)):
            delta_meta_l, unhealthy_cell_l, total_cell_l, variance_metabolite, variance_metabolite_, mean_metabolite = \
                self.simulate_single_agg_group(l, self.t, self.extra_metabolites, self.cell_density, vcd_adjustment,
                                               cal_variance)
            if cal_variance:
                self.variance_metabolite.append(variance_metabolite)
                self.variance_metabolite_.append(variance_metabolite_)
                self.mean_metabolites.append(mean_metabolite)
            delta_meta += delta_meta_l
            health_l.append([unhealthy_cell_l, total_cell_l])
            # print("aggregate: {}; total cell: {}; unhealthy pct: {}".format(l, total_cell_l, unhealthy_cell_l/total_cell_l))
        self.health.append(health_l)

        '''
        PYR, ALA, ASP, ASX(Asparagine), GLY, HIS, ILE, LEU, LYS, SER, TYR (30c), VAL, 
        #GLC (https://bionumbers.hms.harvard.edu/bionumber.aspx?id=104089&ver=7), GLN, EGLU, LAC, 
        NH4 (https://bionumbers.hms.harvard.edu/bionumber.aspx?s=n&v=4&id=104437), 
        BIO, X
        '''
        # 1.12, 0.91, 0.741, 0.83, 0.104, 0.96, 0.75, 0.73,	0.659, 0.891, 0.753, 0.679,
        # 0.6, 0.76, 0.708, 1.033, 1.86 (10^-9 m^2/s)
        self.extra_metabolites += delta_meta[self.meta_index]
        self.all_metabolites += delta_meta
        self.all_metabolites[-1] = self.cell_density
        self.results.append(self.all_metabolites.copy())
        # ex_meta = np.zeros(len(self.extra_metabolites))  # [0.91, 0.741, 0.891, 0.6, 0.76, 0.708, 1.033, 1.86]
        # for l in range(int(self.agg_density.number_of_classes / self.level_size)):
        #     for i in range(self.aggregates[l].num_levels):
        #         ex_meta += self.aggregates[l].intra_metabolites[i][self.meta_index]
        # self.extra_metabolites = ex_meta
        self.t += self.delta_t
        return health_l


def cell_density_cal(t):
    mu = cal_growth_rate()
    if t <= 48:
        cell_density = 200 * np.exp(mu[0] * t)
    elif 48 < t <= 72:
        # cell_density = (490 - 243) / 24 * (t - 48) + 243
        cell_density = 243 * np.exp(mu[1] * (t - 48))
    elif 72 < t <= 96:
        # cell_density = (1100 - 490) / 24 * (t - 72) + 490
        cell_density = 450 * np.exp(mu[2] * (t - 72))
    elif 96 < t <= 120:
        # cell_density = (1150 - 1100) / 24 * (t - 96) + 1100
        cell_density = 1050 * np.exp(mu[3] * (t - 96))
    else:
        # cell_density = (1250 - 1150) / 24 * (t - 120) + 1150
        cell_density = 1100 * np.exp(mu[4] * (t - 120))
    return cell_density / 1.5


def cal_growth_rate():
    return [(np.log(243) - np.log(200)) / 48,
            (np.log(490) - np.log(243)) / 24,
            (np.log(1050) - np.log(490)) / 24,
            (np.log(1050) - np.log(1050)) / 24,
            (np.log(1100) - np.log(1050)) / 24
            ]


def run_simulation(hours, test_index, mean_whole, cell_growth_rate, agg_density_function, meta_index,
                   single_cell_radius, parameters, diffusion_coef, porosity, delta_t, validation=False,
                   cal_variance=False):
    from multi_scale_model.Util import S1_0
    health = []
    x0 = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
    if validation:
        x0_LGHL[-1] = 0.2
    agg_simulator = Aggregate(cell_growth_rate, agg_density_function, meta_index, x0,
                              single_cell_radius, parameters, diffusion_coef, porosity, delta_t=delta_t)
    for i in range(hours):
        if not validation:
            agg_simulator.simulate(cal_variance=cal_variance)
        else:
            if i in [48, 96, 120, 144]:
                x0 = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
                x0[-1] = agg_simulator.cell_density / 1000
                _result = agg_simulator.results.copy()
                _health = agg_simulator.health.copy()
                _t = agg_simulator.t
                agg_simulator = Aggregate(cell_growth_rate, agg_density_function, meta_index, x0,
                                          single_cell_radius, parameters, diffusion_coef, porosity, delta_t=1)
                agg_simulator.results = _result
                agg_simulator.health = _health
                agg_simulator.t = _t
            health.append(agg_simulator.simulate(validation, cal_variance=cal_variance))
        print('time', (i + 1) * 1,
              'GLC', agg_simulator.extra_metabolites[meta_label_index['GLC'][1]],
              'Lac', agg_simulator.extra_metabolites[meta_label_index['ELAC'][1]])
    return agg_simulator, health


def get_aggregation_profile(no_aggregate=True, bioreactor=False):
    import dill as pickle

    try:
        with open('multi_scale_model/data/case_final.pkl', 'rb') as f:
            agg_density_function = pickle.load(f)
    except:
        with open('data/case_final.pkl', 'rb') as f:
            agg_density_function = pickle.load(f)

    if no_aggregate:
        average_size = 5
        # agg_density_function.N.y = np.zeros_like(agg_density_function.N.y)
        # agg_density_function.N.y[5:35, :] = 1
        # for i in range(agg_density_function.N.y.shape[1]):
        #     total = np.sum(agg_density_function.N.y[:, i])
        #     agg_density_function.N.y[:, i] = agg_density_function.N.y[:, i] / total
        _temp = np.zeros_like(agg_density_function.N.y)
        mean_0_days = (np.arange(0, 1500, 1) * agg_density_function.N.y[:, 0]).sum()
        for i in range(1500):
            if i * int(mean_0_days / average_size) < 1500:
                _temp[i, :] = agg_density_function.N.y[i * int(mean_0_days / average_size), :]
            else:
                _temp[i, :] = 0

        for i in range(agg_density_function.N.y.shape[1]):
            agg_density_function.N.y[:, i] = _temp[i, 0]  # agg_density_function.N.y[:, 0]
            # total = np.sum(agg_density_function.N.y[:, i])
            # agg_density_function.N.y[:, i] = agg_density_function.N.y[:, i] / total
        for i in range(1440):
            agg_density_function.N.y[:, i] = _temp[:, 0] / np.sum(_temp[:, 0])

    if not no_aggregate and bioreactor:
        _temp = np.zeros_like(agg_density_function.N.y)
        mean_6_days = (np.arange(0, 1500, 1) * agg_density_function.N.y[:, -1]).sum()
        for i in range(1500):
            if i * int(mean_6_days / 180) < 1500:
                _temp[i, :] = agg_density_function.N.y[i * int(mean_6_days / 180), :]
            else:
                _temp[i, :] = 0
        for i in range(1440):
            agg_density_function.N.y[:, i] = _temp[:, i] / np.sum(_temp[:, i])

    return agg_density_function


if __name__ == '__main__':
    import pandas as pd
    import pickle
    import seaborn as sns
    from multi_scale_model.Util import real_data
    from multi_scale_model.metabolic_flux_rate import g
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from multi_scale_model.Util import *
    from multi_scale_model.PBM import CellAggregatePopulation
    np.random.seed(10)
    # S1_0 = np.array([1.40E-07, 4.00E-07, 1.40E-07, 2.00E-08, 1.50E-06, 2.00E-07, 2.00E-08, 4.00E-06, 3.00E-07,
    #                  5.00E-08, 4.00E-07, 7.00E-04, 8.00E-06, 1E-6, 9.00E-07, 2.00E-08, 9.00E-13, 5.30E-12, 5.00E-05,
    #                  8.00E-06,
    #                  4.00E-08, 2.00E-06, 2.00E-06, 2.8, 1.50E-07, 5.30E-07, 9.20E-08])

    porosity = 0.27  # 60 rpm, and 0.229 100 rpm
    cell_growth_rate = 0.043  # 0.04531871789155581
    try:
        data_whole = np.load('data/data_whole.npy')
    except:
        data_whole = np.load("multi_scale_model/data/data_whole.npy")

    data_whole, mean_whole, std_whole = get_data('dynamic_model/iPSC_data.xlsx')

    single_cell_radius = 7.5
    try:
        file = open("data/param.pkl", 'rb')
    except:
        file = open("multi_scale_model/data/param.pkl", 'rb')
    file = open("multi_scale_model/data/result_all.pickle", 'rb')
    parameters = pickle.load(file)
    parameters = parameters.params

    # hour = 144
    # t_step = 0.1
    # v0 = 1500.0
    # g = 1500
    # agg_density_function = CellAggregatePopulation(
    #         np.zeros(g),
    #         np.arange(0.0, hour, t_step),
    #         v0 / g,
    #         3.12 * 10E-4,
    #         9.71 * 10E2,
    #         Q=None
    # )
    # try:
    #     agg_density_function.N = np.load('data/case3.npy', allow_pickle=True)
    # except:
    #     agg_density_function.N = np.load('multi_scale_model/data/case3.npy', allow_pickle=True)

    # Train 0:HGHL, 1:HGLL, 2:LGLL,
    # Test 3:LGHL

    agg_density_function = get_aggregation_profile(no_aggregate=False, bioreactor=True)
    # HGLL (protocol)
    validation_metabolite_consumption = True
    if validation_metabolite_consumption:
        cell_growth_rate_val = 0.044
        hours = 144
        health = []
        test_index = 1
        x0_LGHL = np.hstack([S2_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
        x0_LGHL[-1] = 0.2
        agg_simulator = Aggregate(cell_growth_rate_val, agg_density_function, meta_index, x0_LGHL,
                                  single_cell_radius, parameters, diffusion_coef, porosity, delta_t=1)
        for i in range(hours):
            if i in [48, 96, 120, 144]:
                x0_LGHL = np.hstack([S2_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
                x0_LGHL[-1] = agg_simulator.cell_density / 1000
                _result = agg_simulator.results.copy()
                _health = agg_simulator.health.copy()
                _t = agg_simulator.t
                agg_simulator = Aggregate(cell_growth_rate_val, agg_density_function, meta_index, x0_LGHL,
                                          single_cell_radius, parameters, diffusion_coef, porosity, delta_t=1)
                agg_simulator.results = _result
                agg_simulator.health = _health
                agg_simulator.t = _t
            health.append(agg_simulator.simulate(validation_metabolite_consumption))
            print('time', (i + 1) * 1,
                  'GLC', agg_simulator.extra_metabolites[meta_label_index['GLC'][1]],
                  'Lac', agg_simulator.extra_metabolites[meta_label_index['ELAC'][1]])

        unhealthy_pct = []
        for i in range(144):
            unhealthy = np.sum(np.array(health[i])[:, 0])
            total = np.sum(np.array(health[i])[:, 1])
            unhealthy_pct.append(unhealthy / total)
        fig = plt.figure(figsize=[10, 5])
        ax = fig.add_subplot(111)
        ax.plot(np.array(range(hours)) + 1, unhealthy_pct, color='blue',
                linewidth=3)
        # ax.set_title('Multi-scale model (K3 iPSC)', fontsize=16)
        ax.set_xlabel('Time (h)', fontsize=16)
        ax.set_ylabel('% Unhealthy Cells', fontsize=16)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.tight_layout()
        plt.show(bbox_inches='tight')

        '''Plot '''
        plot_err = True
        N = pd.read_excel('multi_scale_model/data/Simulator_v3.xlsx', sheet_name="N")
        N = N.fillna(0).values
        N = N[:, 1:]
        cases = {0: "HGHL", 1: "HGLL", 2: "LGLL", 3: "LGHL"}

        ''' Lactate '''
        interval1 = (9.544165757906216 - 5.5659760087241) / 2
        interval2 = (15.808069792802616 - 7.694656488549619) / 2
        interval3 = (14.359869138495092 - 12.580152671755725) / 2
        interval4 = (15.389312977099236 - 13.871319520174481) / 2
        err = np.array([0, interval1, interval2, interval3, interval4])
        fig, ax = plt.subplots()
        x0_LGHL = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
        ax.plot(np.array(range(hours)), np.array(agg_simulator.results)[:hours, meta_label_index['ELAC'][0]],
                color=sns.color_palette("tab10", 7)[3],
                linewidth=3, label='Multi-scale model (K3 iPSC)')
        ax.set_title('Lactate Concentration (mM)', fontsize=16)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Predicted by multi-scale model\n(K3 iPSC, Odenwelder et al, 2020)',
                      color=sns.color_palette("tab10", 7)[3], fontsize=14)
        ax2 = ax.twinx()
        plt.xticks(list(range(0, hours + 1, 12)))
        if plot_err:
            ax.set_ylim(-2, 40.5)
        ax2.scatter([0, 48, 96, 120, 144], [0, 7.5, 11.8, 13.5, 14.7], color=sns.color_palette("tab10", 7)[0],
                    marker="X",
                    s=100,
                    label='Measured (FSiPSC, Kwok et al, 2017)')
        if plot_err:
            ax2.errorbar([0, 48, 96, 120, 144], np.array([0, 7.5, 11.8, 13.5, 14.7]), yerr=err, fmt='o',
                         color=sns.color_palette("tab10", 7)[0],
                         capsize=2)
        ax2.set_ylabel("Measured (FSiPSC, Kwok et al, 2017)", color=sns.color_palette("tab10", 7)[0], fontsize=14)
        plt.xticks(list(range(0, hours + 1, 12)))
        if plot_err:
            plt.savefig("multi_scale_model/result/validation-Kwok2017/{}-{}-with-err.svg".format('HGLL', 'Lac'),
                        bbox_inches='tight')
            plt.savefig("multi_scale_model/result/validation-Kwok2017/{}-{}-with-err".format('HGLL', 'Lac'),
                        bbox_inches='tight')
        else:
            plt.savefig("multi_scale_model/result/validation-Kwok2017/{}-{}".format('HGLL', 'Lac'), bbox_inches='tight')
        plt.show()
        plt.clf()


        ''' Glucose '''
        interval1 = (292 - 250.37) / 2
        interval2 = (256.7 - 169.87) / 2
        interval3 = (208.40 - 183.21) / 2
        interval4 = (187.65 - 166.42) / 2
        err = np.array([0, interval1, interval2, interval3, interval4]) / 100
        fig, ax = plt.subplots()
        # i = 42
        x0_LGHL = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
        ax.plot(np.array(range(hours)),
                np.array(agg_simulator.results)[:hours, meta_label_index['GLC'][0]] * 180.156 / 1000, color=sns.color_palette("tab10", 7)[3],
                # * 180.156 / 1000
                linewidth=3, label='Multi-scale model (K3 iPSC)')
        ax.set_title('Glucose Concentration (g/L)', fontsize=16)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Predicted by multi-scale model\n(K3 iPSC, Odenwelder et al, 2020)', color=sns.color_palette("tab10", 7)[3], fontsize=14)

        ax2 = ax.twinx()
        if plot_err:
            ax.set_ylim(-0.1, 3.5)
            # ax.set_ylim(0, 20)

        ax2.scatter([0, 48, 96, 120, 144], np.array([350, 275, 205, 192.5, 180]) / 100, color=sns.color_palette("tab10", 7)[0], marker="X", s=100,
                    label='Measured (FSiPSC, Kwok et al, 2017)')
        if plot_err:
            ax2.errorbar([0, 48, 96, 120, 144], np.array([350, 275, 205, 192.5, 180]) / 100, yerr=err, fmt='o',
                         color=sns.color_palette("tab10", 7)[0], capsize=2)
        ax2.set_ylim(1.6, 3.57)
        ax2.set_ylabel("Measured (FSiPSC, Kwok et al, 2017)", color=sns.color_palette("tab10", 7)[0], fontsize=14)
        plt.xticks(list(range(0, hours + 1, 12)))
        if plot_err:
            plt.savefig("multi_scale_model/result/validation-Kwok2017/{}-{}-with-err.svg".format('HGLL', 'GLC'),
                        bbox_inches='tight')
            plt.savefig("multi_scale_model/result/validation-Kwok2017/{}-{}-with-err".format('HGLL', 'GLC'),
                        bbox_inches='tight')
        else:
            plt.savefig("multi_scale_model/result/validation-Kwok2017/{}-{}".format('HGLL', 'GLC'),
                        bbox_inches='tight')
        plt.show()
        plt.clf()

        ''' Glycolytic efficiency '''
        fig, ax = plt.subplots()
        # i = 42
        delta_lac = np.array(agg_simulator.results)[:hours, meta_label_index['ELAC'][0]][[48, 96, 120, -1]]
        # delta_glc = np.array(agg_simulator.results)[1:hours, meta_label_index['GLC'][0]] - np.array(agg_simulator.results)[:(hours-1), meta_label_index['GLC'][0]]
        delta_glc = x0_LGHL[22] - np.array(agg_simulator.results)[:hours, meta_label_index['GLC'][0]][[48, 96, 120, -1]]
        # delta_glc[[48, 96, 120]] = delta_glc[[48, 96, 120]] + x0_LGHL[22]

        x0_LGHL = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
        ax.plot([48, 96, 120, 144], delta_lac / delta_glc,
                color=sns.color_palette("tab10", 7)[3],
                # * 180.156 / 1000
                linewidth=3, label='Multi-scale model (K3 iPSC)')
        ax.set_title('Glycolytic Efficiency', fontsize=16)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Predicted by multi-scale model\n(K3 iPSC, Odenwelder et al, 2020)', color=sns.color_palette("tab10", 7)[3], fontsize=14)

        ax2 = ax.twinx()
        if plot_err:
            ax.set_ylim(1, 2.5)
            # ax.set_ylim(0, 20)

        ax2.scatter([48, 96, 120, 144], np.array([7.5, 11.8, 13.5, 14.7]) / ((350 - np.array([275, 205, 192.5, 180]))/180.156*10),
                    color=sns.color_palette("tab10", 7)[0], marker="X", s=100,
                    label='Measured (FSiPSC, Kwok et al, 2017)')
        ax2.set_ylabel("Measured (FSiPSC, Kwok et al, 2017)", color=sns.color_palette("tab10", 7)[0], fontsize=14)
        ax2.set_ylim(1, 2.5)
        plt.xticks(list(range(0, hours + 1, 12)))
        # leg = ax.legend()
        # if plot_err:
        #     plt.savefig("multi_scale_model/result/validation-Kwok2017/{}-{}-with-err.svg".format('HGLL', 'GLC'),
        #                 bbox_inches='tight')
        #     plt.savefig("multi_scale_model/result/validation-Kwok2017/{}-{}-with-err".format('HGLL', 'GLC'),
        #                 bbox_inches='tight')
        # else:
        #     plt.savefig("multi_scale_model/result/validation-Kwok2017/{}-{}".format('HGLL', 'GLC'),
        #                 bbox_inches='tight')
        plt.show()
        plt.clf()

    agg_density_function = get_aggregation_profile(no_aggregate=False, bioreactor=True)
    hours = 97
    agg_simulator_HGHL, health = run_simulation(hours, 0, mean_whole, cell_growth_rate, agg_density_function,
                                                meta_index,
                                                single_cell_radius, parameters, diffusion_coef, porosity, 1,
                                                cal_variance=False)
    agg_simulator_HGLL, health = run_simulation(hours, 1, mean_whole, cell_growth_rate, agg_density_function,
                                                meta_index,
                                                single_cell_radius, parameters, diffusion_coef, porosity, 1,
                                                cal_variance=True)
    agg_simulator_LGLL, health = run_simulation(hours, 2, mean_whole, cell_growth_rate, agg_density_function,
                                                meta_index,
                                                single_cell_radius, parameters, diffusion_coef, porosity, 1,
                                                cal_variance=False)
    agg_simulator_LGHL, health = run_simulation(hours, 3, mean_whole, cell_growth_rate, agg_density_function,
                                                meta_index,
                                                single_cell_radius, parameters, diffusion_coef, porosity, 1,
                                                cal_variance=False)

    # variance_result = np.array(agg_simulator_HGLL.variance_metabolite).reshape((96, 100, 45))
    variance_result = np.array(agg_simulator_HGLL.variance_metabolite).reshape((hours, 100, len(meta)-1))
    mean_result = np.array(agg_simulator_HGLL.mean_metabolites).reshape((hours, 100, len(meta)-1))
    # Variance of biomass production

    with plt.style.context('default'):
        colormap = mpl.cm.tab10.colors
        fig, ax = plt.subplots()
        ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='both', useMathText=True)
        total_variance_biomass = []
        total_variance_biomass_std = []
        for i in range(hours):
            total_variance_biomass.append(
                np.nansum(np.sqrt(variance_result[i, :, -1]) / (np.abs(mean_result[i, :, -1]))))
            total_variance_biomass_std.append(np.nanstd(np.sqrt(variance_result[i, :, -1]) / (np.abs(mean_result[i, :, -1]))))
        # sum of variance of metabolite conc change
        total_variance_all = []
        total_variance_all_std = []
        total_variance_all_moving_std = []
        for i in range(hours):
            # total_variance_all.append(np.nansum(np.sqrt(variance_result[i, :, :])) / np.nansum(np.abs(mean_result[i, :, :])))
            total_variance_all.append(
                np.nansum(np.sqrt(variance_result[i, :, :]) / (np.abs(mean_result[i, :, :]) + 1e-6)) / (len(meta)-1))
            total_variance_all_std.append(
                np.nanstd(np.sqrt(variance_result[i, :, :]) / (np.abs(mean_result[i, :, :]))) / (len(meta)-1))
        ax.plot(np.array(range(hours))[:72] + 1, total_variance_biomass[:72], color=colormap[0], alpha=0.9, linewidth=3)
        ax.set_ylabel('Relative Standard Deviation of \n Biomass Production in $\Delta t$ Time', fontsize=14,
                      color=colormap[0], multialignment='center')
        # ax.set_ylim(0, 5e4)
        ax2 = ax.twinx()
        ax2.ticklabel_format(style='sci', scilimits=(-3, 4), axis='both', useMathText=True)
        ax.set_xlabel('Time (h)', fontsize=14)
        ax2.plot(np.array(range(hours))[:72] + 1, total_variance_all[:72], color=colormap[3], alpha=0.9, linewidth=3, )
        ax2.set_ylabel('Average Relative Standard Deviation of \n Metabolite Change in $\Delta t$ Time', fontsize=14,
                       color=colormap[3], multialignment='center')
        # ax2.set_ylim(0, 5e4)
        # ax2.fill_between(np.array(range(hours)) + 1, np.array(total_variance_all) - np.array(total_variance_all_std),
        #                  np.array(total_variance_all) + np.array(total_variance_all_std),
        #                  color='g', alpha=.1)
        # plt.yticks(fontsize=14)
        # plt.xticks(fontsize=14)
        # plt.xticks(list(range(0, hours + 1, 12)))
        plt.tight_layout()
        # plt.savefig("multi_scale_model/result/variance-component-analysis/{}.svg".format('variance-with-time'),
        #             bbox_inches='tight', format='svg')
        # plt.savefig("multi_scale_model/result/variance-component-analysis/{}.pdf".format('variance-with-time'),
        #             bbox_inches='tight', format='pdf')
        plt.show(bbox_inches='tight')
        plt.clf()

    variance_result = np.load('multi_scale_model/result/variance_result.npy')
    stoichiometric_matrix = pd.read_excel('multi_scale_model/data/Simulator_v3.xlsx', sheet_name="N")
    stoichiometric_matrix = stoichiometric_matrix.fillna(0).values
    stoichiometric_matrix = stoichiometric_matrix[:, 1:]
    v_var = []
    for i in range(hours):
        v_var.append(np.abs(np.linalg.pinv(stoichiometric_matrix.astype('float')) @ np.nansum(
            np.sqrt(variance_result[i, :, :]) / (abs(mean_result[i, :, :]) + 1e-10), axis=0)))
    v_var = np.array(v_var)
    v_group_var = np.array([np.mean(v_var[:, :10], axis=1),
                            np.mean(v_var[:, 10:12], axis=1),
                            np.mean(v_var[:, 12:22], axis=1),
                            np.mean(v_var[:, 22:38], axis=1),
                            v_var[:, 38]]).T

    e_var = []
    for i in range(96):
        e_var.append(
            np.abs(np.linalg.pinv(stoichiometric_matrix.astype('float')) @ np.nansum(mean_result[i, :, :], axis=0)))
    e_var = np.array(e_var)
    e_group_var = np.array([np.mean(e_var[:, :10], axis=1),
                            np.mean(e_var[:, 10:12], axis=1),
                            np.mean(e_var[:, 12:22], axis=1),
                            np.mean(e_var[:, 22:38], axis=1),
                            e_var[:, 38]]).T

    # imports
    import cufflinks as cf
    import pandas as pd
    import numpy as np
    import seaborn as sns

    # variance
    # %qtconsole --style vim #src# https://qtconsole.readthedocs.io/en/stable/
    v_group_var_df = pd.DataFrame(v_group_var,
                                  columns=['Glycolysis', 'PPP', 'TCA', 'Anaplerosis and Amino Acid', 'Biomass'])
    v_group_var_df.index = np.arange(1, 97, 1)
    v_group_var_df['total'] = v_group_var_df.sum(1)
    fraction_v_group_var_df = v_group_var_df.apply(lambda x: x / x[-1], axis=1)
    v_group_var_df = v_group_var_df.drop('total', axis=1)
    fraction_v_group_var_df.drop('total', axis=1, inplace=True)
    # make figure fraction bar plots
    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax = v_group_var_df[['PPP', 'TCA', 'Glycolysis', 'Anaplerosis and Amino Acid']].plot(ax=axes[0], kind='bar',
                                                                                         stacked=True, width=0.8,
                                                                                         figsize=(16, 10), fontsize=12)
    ax.set_ylabel('Total mean \n in different Metabolic Pathways', fontsize=14)
    ax.legend(title='Metabolic Pathway', fontsize=14)
    ax.set_xlabel('Hours', fontsize=14)
    ax = fraction_v_group_var_df[['PPP', 'TCA', 'Glycolysis', 'Anaplerosis and Amino Acid']].plot(ax=axes[1],
                                                                                                  kind='bar',
                                                                                                  stacked=True,
                                                                                                  width=0.8,
                                                                                                  figsize=(16, 10),
                                                                                                  fontsize=12,
                                                                                                  legend=False)
    ax.set_xlabel('Hours', fontsize=14)
    # plt.xticks(rotation=0)
    ax.set_ylabel('Fraction of Total mean \n in different Metabolic Pathways', fontsize=14)
    # plt.savefig("multi_scale_model/result/variance-component-analysis/variance-decomposition-with-time.pdf",
    #             bbox_inches='tight', format='pdf')
    plt.show()

    # bar plots
    fig, axes = plt.subplots(nrows=5, ncols=1)
    ax = v_group_var_df[['PPP']].plot(ax=axes[0], kind='bar', stacked=True, width=0.8, figsize=(16, 13), fontsize=12)
    ax.set_ylabel('Averaged RSD \n in PPP Pathways', fontsize=14)
    ax = v_group_var_df[['TCA']].plot(ax=axes[1], kind='bar', stacked=True, width=0.8, figsize=(16, 13), fontsize=12)
    ax.set_ylabel('Averaged RSD \n in TCA Pathways', fontsize=14)
    ax = v_group_var_df[['Glycolysis']].plot(ax=axes[2], kind='bar', stacked=True, width=0.8, figsize=(16, 13),
                                             fontsize=12)
    ax.set_ylabel('Averaged RSD \n in Glycolysis Pathways', fontsize=14)
    ax = v_group_var_df[['Anaplerosis and Amino Acid']].plot(ax=axes[3], kind='bar', stacked=True, width=0.8,
                                                             figsize=(16, 13), fontsize=12)
    ax.set_ylabel('Averaged RSD \n in Anaplerosis and \n Amino Acid Pathways', fontsize=14)
    ax = v_group_var_df[['Biomass']].plot(ax=axes[4], kind='bar', stacked=True, width=0.8, figsize=(16, 13),
                                          fontsize=12)
    ax.set_ylabel('Averaged RSD \n in Biomass Pathways', fontsize=14)
    ax.set_xlabel('Hours', fontsize=14)
    plt.show()

    # line chart
    sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
    sns.set_context("paper", font_scale=1)
    splot = sns.lineplot(
        data=fraction_v_group_var_df[['PPP', 'TCA', 'Glycolysis', 'Anaplerosis and Amino Acid']],
        markers=True, dashes=False, linewidth=3
    )
    sns.scatterplot(
        data=fraction_v_group_var_df[['PPP', 'TCA', 'Glycolysis', 'Anaplerosis and Amino Acid']], s=10, legend=False
    )

    plt.xlabel('Hours', fontsize=14)
    plt.ylabel('Fraction of Averaged RSD \n in different Metabolic Pathways', fontsize=14)
    plt.tight_layout()
    splot.yaxis.grid(True, clip_on=False)
    sns.despine(left=True, bottom=True)

    plt.legend(loc=(0.5, 0.2), fontsize='large')
    # plt.savefig("multi_scale_model/result/variance-component-analysis/expectation-decomposition-with-time.pdf",
    #             bbox_inches='tight', format='pdf')
    plt.show()

    # make figure
    # variance-decomposition-with-time
    # variance
    v_group_var_df = pd.DataFrame(v_group_var,
                                  columns=['Glycolysis', 'PPP', 'TCA', 'Anaplerosis and Amino Acid', 'Biomass'])
    v_group_var_df.index = np.arange(1, 97, 1)
    v_group_var_df['total'] = v_group_var_df.sum(1)
    fraction_v_group_var_df = v_group_var_df.apply(lambda x: x / x[-1], axis=1)
    v_group_var_df = v_group_var_df.drop('total', axis=1)
    fraction_v_group_var_df.drop('total', axis=1, inplace=True)

    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax = v_group_var_df[['PPP', 'TCA', 'Glycolysis', 'Anaplerosis and Amino Acid']].plot(ax=axes[0], kind='bar',
                                                                                         stacked=True, width=0.8,
                                                                                         figsize=(16, 10), fontsize=12)
    ax.set_ylabel('Total mean \n in different Metabolic Pathways', fontsize=14)
    ax.legend(title='Metabolic Pathway', fontsize=14)
    ax.set_xlabel('Hours', fontsize=14)
    ax = fraction_v_group_var_df[['PPP', 'TCA', 'Glycolysis', 'Anaplerosis and Amino Acid']].plot(ax=axes[1],
                                                                                                  kind='bar',
                                                                                                  stacked=True,
                                                                                                  width=0.8,
                                                                                                  figsize=(16, 10),
                                                                                                  fontsize=12,
                                                                                                  legend=False)
    ax.set_xlabel('Hours', fontsize=14)
    # plt.xticks(rotation=0)
    ax.set_ylabel('Fraction of Total mean \n in different Metabolic Pathways', fontsize=14)
    # plt.savefig("multi_scale_model/result/variance-component-analysis/variance-decomposition-with-time.pdf",
    #             bbox_inches='tight', format='pdf')
    plt.show()

    # expectation-decomposition-with-time
    # mean
    e_group_var_df = pd.DataFrame(e_group_var,
                                  columns=['Glycolysis', 'PPP', 'TCA', 'Anaplerosis and Amino Acid', 'Biomass'])
    e_group_var_df.index = np.arange(1, 97, 1)
    e_group_var_df['total'] = e_group_var_df.sum(1)
    fraction_e_group_var_df = e_group_var_df.apply(lambda x: x / x[-1], axis=1)
    e_group_var_df = e_group_var_df.drop('total', axis=1)
    fraction_e_group_var_df.drop('total', axis=1, inplace=True)
    sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
    sns.set_context("paper", font_scale=1)
    splot = sns.lineplot(
        data=fraction_e_group_var_df[['PPP', 'TCA', 'Glycolysis', 'Anaplerosis and Amino Acid']],
        markers=True, dashes=False, linewidth=3
    )
    sns.scatterplot(
        data=fraction_e_group_var_df[['PPP', 'TCA', 'Glycolysis', 'Anaplerosis and Amino Acid']], s=10, legend=False
    )

    plt.xlabel('Hours', fontsize=14)
    plt.ylabel('Fraction of Average Flux Rate \n of different Metabolic Pathways', fontsize=14)
    plt.tight_layout()
    splot.yaxis.grid(True, clip_on=False)
    sns.despine(left=True, bottom=True)

    plt.legend(loc=(0.5, 0.2), fontsize='large')
    # plt.savefig("multi_scale_model/result/variance-component-analysis/expectation-decomposition-with-time.pdf",
    #             bbox_inches='tight', format='pdf')
    # plt.savefig("multi_scale_model/result/variance-component-analysis/expectation-decomposition-with-time.svg",
    #             bbox_inches='tight', format='svg')
    plt.show()
    # correlation
    pearson = fraction_v_group_var_df.corr(method='pearson')
    spearman = fraction_v_group_var_df.corr(method='spearman')

    ''' The following are not presented in the paper'''
    # expectation-decomposition-with-time
    # e_group_var = np.array([v_var[:, 3],
    #                         np.mean(e_var[:, 6:9], axis=1),
    #                         e_var[:, 16],
    #                         np.mean(e_var[:, 17:19], axis=1),
    #                         e_var[:, 33]]).T
    e_group_var = np.array([np.mean(e_var[:, :5], axis=1),
                            np.mean(e_var[:, 6:9], axis=1),
                            np.mean(e_var[:, 9:15], axis=1),
                            np.mean(e_var[:, 15:33], axis=1),
                            e_var[:, 33]]).T
    e_group_var_df = pd.DataFrame(e_group_var,
                                  columns=['Glycolysis', 'PPP', 'TCA', 'Anaplerosis and Amino Acid', 'Biomass'])
    e_group_var_df.index = np.arange(1, 97, 1)
    e_group_var_df['total'] = e_group_var_df.sum(1)
    fraction_e_group_var_df = e_group_var_df.apply(lambda x: x / x[-1], axis=1)
    e_group_var_df = e_group_var_df.drop('total', axis=1)
    fraction_e_group_var_df.drop('total', axis=1, inplace=True)

    sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
    sns.set_context("paper", font_scale=1)
    splot = sns.lineplot(
        data=e_group_var_df[['TCA', 'Anaplerosis and Amino Acid']],
        markers=True, dashes=False, linewidth=3
    )
    sns.scatterplot(
        data=e_group_var_df[['TCA', 'Anaplerosis and Amino Acid']], s=10, legend=False
    )

    plt.xlabel('Hours', fontsize=14)
    plt.ylabel('Fraction of Average Flux Rate \n in different Metabolic Pathways', fontsize=14)
    plt.tight_layout()
    splot.yaxis.grid(True, clip_on=False)
    sns.despine(left=True, bottom=True)

    plt.legend(loc=(0.5, 0.2), fontsize='large')
    # plt.savefig("multi_scale_model/result/variance-component-analysis/expectation-decomposition-with-time.pdf",
    #             bbox_inches='tight', format='pdf')
    plt.show()

    '''Run simulation'''
    # HGHL
    hours = 24
    test_index = 0
    x0_LGHL = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
    agg_simulator = Aggregate(cell_growth_rate, agg_density_function, meta_index, x0_LGHL,
                              single_cell_radius, parameters, diffusion_coef, porosity, delta_t=1)
    for i in range(hours):
        agg_simulator.simulate()
        print('time', (i + 1) * 1, 'GLC', agg_simulator.extra_metabolites[3], 'Lac',
              agg_simulator.extra_metabolites[-2])

    # HGLL
    test_index = 1
    x0_LGHL = np.hstack([S2_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
    # x0_LGHL[[28, 29, 36, 39, 40, 42]] = [real_data[test_index]["Ala.x"][0],
    #                                    real_data[test_index]["Asp.x"][0],
    #                                    real_data[test_index]["Ser.x"][0],
    #                                    real_data[test_index]["Glc.x"][0],
    #                                    real_data[test_index]["Gln.x"][0],
    #                                    real_data[test_index]["Lac.x"][0]]
    agg_simulator = Aggregate(cell_growth_rate, agg_density_function, meta_index, x0_LGHL,
                              single_cell_radius, parameters, diffusion_coef, porosity, delta_t=1)
    for i in range(hours):
        agg_simulator.simulate()
        print('time', (i + 1) * 1, 'GLC', agg_simulator.extra_metabolites[meta_label_index['GLC'][1]],
              'Lac', agg_simulator.extra_metabolites[meta_label_index['ELAC'][1]])

    # LGLL
    test_index = 2
    x0_LGHL = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
    # x0_LGHL[[28, 29, 36, 39, 40, 42]] = [real_data[test_index]["Ala.x"][0],
    #                                    real_data[test_index]["Asp.x"][0],
    #                                    real_data[test_index]["Ser.x"][0],
    #                                    real_data[test_index]["Glc.x"][0],
    #                                    real_data[test_index]["Gln.x"][0],
    #                                    real_data[test_index]["Lac.x"][0]]

    agg_simulator = Aggregate(cell_growth_rate, agg_density_function, meta_index, x0_LGHL,
                              single_cell_radius, parameters, diffusion_coef, porosity, delta_t=1)
    for i in range(hours):
        agg_simulator.simulate()
        print('time', (i + 1) * 1, 'GLC', agg_simulator.extra_metabolites[3], 'Lac',
              agg_simulator.extra_metabolites[-2])

    # LGHL
    test_index = 3
    x0_LGHL = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
    # x0_LGHL[[28, 29, 36, 39, 40, 42]] = [real_data[test_index]["Ala.x"][0],
    #                                    real_data[test_index]["Asp.x"][0],
    #                                    real_data[test_index]["Ser.x"][0],
    #                                    real_data[test_index]["Glc.x"][0],
    #                                    real_data[test_index]["Gln.x"][0],
    #                                    real_data[test_index]["Lac.x"][0]]
    agg_simulator = Aggregate(cell_growth_rate, agg_density_function, meta_index, x0_LGHL,
                              single_cell_radius, parameters, diffusion_coef, porosity, delta_t=1)
    for i in range(hours):
        agg_simulator.simulate()
        print('time', (i + 1) * 1, 'GLC', agg_simulator.extra_metabolites[3], 'Lac',
              agg_simulator.extra_metabolites[-2])

    ''' Generate the inner and outer cell metabolism'''
    test_index = 1
    x0_LGHL = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
    # x0_LGHL[[28, 29, 36, 39, 40, 42]] = [real_data[test_index]["Ala.x"][0],
    #                                    real_data[test_index]["Asp.x"][0],
    #                                    real_data[test_index]["Ser.x"][0],
    #                                    real_data[test_index]["Glc.x"][0],
    #                                    real_data[test_index]["Gln.x"][0],
    #                                    real_data[test_index]["Lac.x"][0]]
    agg_simulator = Aggregate(cell_growth_rate, agg_density_function, meta_index, x0_LGHL,
                              single_cell_radius, parameters, diffusion_coef, porosity, delta_t=1, deterministic=True)
    for i in range(hours):
        agg_simulator.simulate()
        print('time', (i + 1) * 1, 'GLC', agg_simulator.extra_metabolites[meta_label_index['GLC'][1]],
              'Lac', agg_simulator.extra_metabolites[meta_label_index['ELAC'][1]])
        if i in [24, 48, 72]:
            for size in [4, 8, 16, 24]:
                rates, conc = agg_simulator.aggregates[size].get_flux_rate_cell_level(agg_simulator.extra_metabolites)
                outer_cell, inner_cell = rates[-1] * 1e6, rates[0] * 1e6
                # np.savetxt("multi_scale_model/result/flux_rate/inner_cell-hours{}-size{}-cd.csv".format(i, size * 15),
                #            inner_cell, delimiter=", ")
                # np.savetxt("multi_scale_model/result/flux_rate/outer_cell-hours{}-size{}-cd.csv".format(i, size * 15),
                #            outer_cell, delimiter=", ")

    '''Plot the unhealthy trajectories'''
    def plot_unhealth_trajectory(list_agg_simulators, hours, path='multi_scale_model/result/unhealthy'):
        import os
        labels = ['HGHL', 'HGLL', 'LGLL', 'LGHL']
        with plt.style.context('_mpl-gallery'):
            fig = plt.figure(figsize=[10, 5])
            for k, agg_simulator in enumerate(list_agg_simulators):
                colormap = mpl.cm.tab10.colors
                unhealthy_pct = []
                health = np.array(agg_simulator.health)
                for i in range(hours):
                    unhealthy = np.sum(np.array(health[i])[:, 0])
                    total = np.sum(np.array(health[i])[:, 1])
                    unhealthy_pct.append(unhealthy / total)
                plt.plot(np.array(range(hours)) + 1, np.array(unhealthy_pct) * 100, color=colormap[k],
                         linewidth=3, label=labels[k])
            # ax.set_title('HGLL', fontsize=16)
            plt.xlabel('Time (h)', fontsize=16)
            plt.ylabel('% Unhealthy Cells', fontsize=16)
            plt.yticks(fontsize=14)
            plt.xticks(fontsize=14)
            plt.legend(fontsize=15)
            fig.tight_layout(rect=(0, 0, 1, 1))
            plt.savefig(
                os.path.join(path,
                             "batch-unhealthy-{}.pdf".format("time")),
                bbox_inches='tight', format='pdf')
            plt.savefig(
                os.path.join(path,
                             "batch-unhealthy-{}.svg".format("time")),
                bbox_inches='tight', format='svg')
            plt.savefig(
                os.path.join(path,
                             "batch-unhealthy-{}.png".format("time")),
                bbox_inches='tight', format='png')
            plt.show(bbox_inches='tight')


    plot_unhealth_trajectory([agg_simulator_HGHL, agg_simulator_HGLL, agg_simulator_LGLL, agg_simulator_LGHL], hours)

    '''Plot '''
    plot_comparison_under_monolayer_condition(test_index, hours, agg_simulator, mean_whole, std_whole, parameters,
                                              draw_measurement=True,
                                              path='multi_scale_model/result/simulation-with-agg')
    hours = 48
    macro_replication = 10
    agg_sims = []
    for _ in range(macro_replication):
        agg_simulator, health = run_simulation(hours, 1, mean_whole, cell_growth_rate, agg_density_function,
                                                    meta_index,
                                                    single_cell_radius, parameters, diffusion_coef, porosity, 1,
                                                    cal_variance=False)
        agg_sims.append(agg_simulator)
    plot_comparison_under_monolayer_condition_macro(test_index, hours, g, agg_sims, mean_whole, std_whole,
                                              parameters,
                                              draw_measurement=True)



