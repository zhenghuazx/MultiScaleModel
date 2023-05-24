import numpy as np
import pandas as pd
from multi_scale_model.metabolic_flux_rate import f_m
from scipy.stats import poisson


class FluxRate:
    def __init__(self, stoichiometric_matrix, parameters, reaction=None):
        self.N = stoichiometric_matrix
        self.parameters = parameters
        self.reaction = ["HK", "PGI", "PEK/ALD", "PGK", "PK", "LDH", "G6PDH/PGLcDH", "EP", "TK/TA", "PDH", "CS",
                         "CITS/ISOD", "AKGDH", "SDH/FUM", "MLD", "ME", "PC", "GLNS", "GLDH", "AlaTA", "GluT", "resp",
                         "leak", "ATPase", "AK", "CK", "PPRibP", "NADPHox", "SAL", "ASX", "ASTA", "AA1", "AA2",
                         "Biomass"]
        if reaction is not None:
            self.reaction = reaction
            if self.reaction[-1] == 'growth':
                self.reaction[-1] = 'Biomass'

    def estimate_flux_rate(self, metabolite):
        return f_m(0, metabolite, self.parameters, self.N, 0)


class CellModel:
    """
    Model for a single cell with give size
    Args:
        *args: A list of numpy arrays which will be concatenated separately.

    Returns:
        A list of concatenated numpy arrays with each representing the instantaneous flux rate of single cell.
    """

    def __init__(self, size, ini_intra_metabolites, single_cell_radius, parameters, diffusion_coef,
                 meta_index, porosity=0.229, delta_t=0.5, deterministic=False):
        self.intra_metabolites = None
        self.radii = None
        self.num_levels = 0
        self._t = 0
        self.delta_t = delta_t
        self.init_intra_metabolites = ini_intra_metabolites
        self.diffusion_coef = np.array(diffusion_coef) * 3600 * 1e3  # unit from m^2/h to um^2/h
        self.single_cell_radius = single_cell_radius
        self.meta_index = meta_index
        self.porosity = porosity
        self.metabolite_conc_list = []
        self.deterministic = True
        try:
            stoichiometric_matrix = pd.read_excel('data/Simulator_v3.xlsx', sheet_name="N")
        except:
            stoichiometric_matrix = pd.read_excel('multi_scale_model/data/Simulator_v3.xlsx', sheet_name="N")
        reaction_labels = stoichiometric_matrix.columns[1:].to_list()
        stoichiometric_matrix = stoichiometric_matrix.fillna(0).values
        stoichiometric_matrix = stoichiometric_matrix[:, 1:]
        self.stoichiometric_matrix = stoichiometric_matrix.astype(float)
        self.flux_rate_solver = FluxRate(self.stoichiometric_matrix, parameters, reaction_labels)
        self.update_aggregate_property(size, single_cell_radius)

    @staticmethod
    def reaction_diffusion(boundary_condition, rate, diffusion_coef, outer_radius, inner_radius):
        c = rate * diffusion_coef / 6 * (outer_radius ** 2 - inner_radius ** 2) + boundary_condition
        c[c < 0] = 0
        return c

    def update_aggregate_property(self, size, single_cell_radius):
        self.size = size
        self.radii = np.arange(0, self.size + 0.1, single_cell_radius)
        self.num_levels = len(self.radii) - 1
        self.intra_metabolites = [self.init_intra_metabolites.copy() for _ in
                                  range(self.num_levels)]  # num of metabolite per level
        self._intra_metabolites = [self.init_intra_metabolites.copy() for _ in range(self.num_levels)]
        #  [self.init_intra_metabolites] * self.num_levels

    def get_covariance(self, rates, delta_t, spherical_shell_cell_density):
        list_data = []
        list_data_ = []
        for _ in range(100):
            shell_data = []
            shell_data_ = []
            for k, rate in enumerate(rates):
                sign = [-1 if r < 0 else 1 for r in rate]
                # number of reaction occurrences
                # mmol/L = mmol/10^6cells/hour * hour * 1000 * 10^6 cells/L
                if spherical_shell_cell_density is not None:
                    delta_meta = [np.mean(poisson.rvs(abs(r) * 1e3 * delta_t * spherical_shell_cell_density[k],
                                                      size=5)) * sign[i] if r != 0 else 0 for i, r in enumerate(rate)]
                    # delta_meta = [r * 1e3 * delta_t * spherical_shell_cell_density[k] * sign[i] if r != 0 else 0 for i, r in enumerate(rate)]
                    data = np.concatenate([self.flux_rate_solver.N @ delta_meta * 1e-3])  # delta u
                    data_ = data / spherical_shell_cell_density[k]
                else:
                    # delta_meta = [np.mean(poisson.rvs(abs(r) * 1e6 * delta_t, size=10)) * sign[i] if r != 0 else 0 for i, r in enumerate(rate)]
                    delta_meta = [r * 1e3 * delta_t * sign[i] if r != 0 else 0 for i, r in enumerate(rate)]
                    data = np.concatenate([self.flux_rate_solver.N @ delta_meta * 1e-3])  # delta u
                    data_ = data
                shell_data.append(data)
                shell_data_.append(data_)
            list_data.append(shell_data)
            list_data_.append(shell_data_)
        list_data = np.array(list_data)  # num rep * num shells * num metabolite
        list_data_ = np.array(list_data_)

        var_meta = []
        var_meta_ = []
        sample_means = []
        sample_covs = []
        for i in range(list_data.shape[2]):
            sample_cov = np.cov(list_data[:, :, i])
            sample_cov_ = np.cov(list_data_[:, :, i])
            sample_mean = np.mean(list_data[:, :, i])
            var = np.sum(np.diag(sample_cov))
            var_ = np.sum(np.diag(sample_cov_))
            var_meta.append(var)
            var_meta_.append(var_)
            sample_means.append(sample_mean)
            sample_covs.append(np.sum(np.abs(sample_cov_)) - var_)
        return var_meta, var_meta_, sample_means

    def update_intracellular_conditions(self, rates, delta_t, spherical_shell_cell_density, cell_density):
        # delta_metabolites = np.zeros_like(self.intra_metabolites[0])
        # for k, rate in enumerate(rates):
        #     sign = [-1 if r < 0 else 1 for r in rate]
        #     # mmol/L = mmol/10^6cells/hour * hour * 1000 * 10^6 cells/L
        #     delta_meta = [np.mean(poisson.rvs(abs(r) * 1e3 * delta_t * spherical_shell_cell_density[k], size=10)) * sign[i] if r != 0 else 0 for i, r in enumerate(rate)]
        #     data = np.concatenate([self.flux_rate_solver.N @ delta_meta * 1e-3, [cell_density]])
        #     self.intra_metabolites[k] += data
        #     delta_metabolites += data
        delta_metabolites = []
        _delta_metabolites = 0
        for k, rate in enumerate(rates):
            sign = [-1 if r < 0 else 1 for r in rate]
            # number of reaction occurrences
            # mmol/L = mmol/10^6cells/hour * hour * 1000 * 10^6 cells/L
            delta_meta, temp_delta_meta = [], 0
            for i, r in enumerate(rate):
                if 0 < abs(r) <= 1e5:
                    temp_delta_meta = np.mean(poisson.rvs(abs(r) * 1e3 * delta_t * spherical_shell_cell_density[k],
                                                          size=100)) * sign[i]
                elif abs(r) > 1e5:
                    temp_delta_meta = np.mean(poisson.rvs(abs(r) * delta_t * spherical_shell_cell_density[k],
                                                          size=100)) * sign[i] * 1e3
                if not self.deterministic:
                    delta_meta.append(temp_delta_meta)
                else:
                    delta_meta.append(r * 1e3 * delta_t * spherical_shell_cell_density[k])

            data = np.concatenate([self.flux_rate_solver.N @ np.array(delta_meta) * 1e-3, [cell_density]])  # delta u
            # Avoid metabolite concentration to be below zero
            cur_intra_metabolites = np.array(self.intra_metabolites[k])
            next_intra_metabolites = cur_intra_metabolites + data
            next_intra_metabolites[next_intra_metabolites < 0] = 0
            data = next_intra_metabolites - cur_intra_metabolites
            # update the concentrations
            # self._intra_metabolites[k] += data / (spherical_shell_cell_density[k] + 1e-8) * np.sum(spherical_shell_cell_density)
            '''
            Key Idea: use the single-cell model requires us to rescale intracellular metabolite 
            '''
            temp = data / (spherical_shell_cell_density[k] + 1e-12) * cell_density
            temp[-1] = cell_density
            self.intra_metabolites[k] += temp  # np.sum(spherical_shell_cell_density)  # 10^6 cells/ml
            self._intra_metabolites[k] += temp  # np.sum(spherical_shell_cell_density)

            delta_metabolites.append(data)
            _delta_metabolites += data
        self._t += delta_t
        return _delta_metabolites

    def get_flux_rate_cell_level(self, boundary_condition):
        """
        Common training loop shared by subclasses, monitors training status
        and progress, performs all training steps, updates metrics, and logs progress.
        Args:
            :type metabolites: array
            :param metabolites
            :type boundary_condition: array
            :param boundary_condition
        """
        s = boundary_condition
        metabolite_conc = []
        rates = []
        for i in reversed(range(1, len(self.radii))):
            self.intra_metabolites[i - 1][
                self.meta_index] = s  # update the intracelluar metabolites with extracellular conditions
            rate = np.array(self.flux_rate_solver.estimate_flux_rate(self.intra_metabolites[i - 1])[:-1]) / 1000
            rates.append(rate)
            unit_factor = 3 / 4 / np.pi / 7.5 ** 3 / self.porosity * 1.1 * 1e-3  # nmol/cell/h to nmol/um^3/h
            reaction_rate = self.stoichiometric_matrix @ rate * unit_factor
            if len(self.radii) == 1:
                s = self.reaction_diffusion(s, reaction_rate[self.meta_index], self.diffusion_coef, self.radii[i], 0)
            else:
                s = self.reaction_diffusion(s, reaction_rate[self.meta_index], self.diffusion_coef, self.radii[i],
                                            self.radii[i - 1])
            metabolite_conc.append(s)
        # self.update_intracellular_conditions(list(reversed(rates)), self.delta_t)
        self.metabolite_conc_list.append(np.array(list(reversed(metabolite_conc))))
        return np.array(list(reversed(rates))), list(reversed(metabolite_conc))


def _get_total_num_cell_for_given_aggregate(R0, porosity, R_levels):
    """Plot Unhealthy Cells"""
    if len(R_levels) == 1:
        return (1 - porosity) * np.array([(R_levels[0] ** 3) / R0 ** 3])

    temp = []
    for n in range(len(R_levels)):
        if n == 0:
            temp.append((R_levels[0] ** 3) / R0 ** 3)
        else:
            temp.append((R_levels[n] ** 3 - R_levels[n - 1] ** 3) / R0 ** 3)
    return (1 - porosity) * np.array(temp)  # * (np.exp(-self.t) + 2) / 3


def get_fraction_cell_in_agg_l(R0, porosity, R_levels):
    cell_num = _get_total_num_cell_for_given_aggregate(R0, porosity, R_levels)
    fraction_cell_num = cell_num / np.sum(cell_num)
    return fraction_cell_num


if __name__ == '__main__':
    import pickle
    from multi_scale_model.Util import get_data
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from multi_scale_model.aggregate import run_simulation, get_aggregation_profile
    from multi_scale_model.Util import *

    file = open("multi_scale_model/data/result_all.pickle", 'rb')
    parameters = pickle.load(file)
    parameters = parameters.params
    import pickle
    from multi_scale_model.PBM import CellAggregatePopulation

    porosity = 0.229  # 60 rpm, and 0.229 100 rpm
    cell_growth_rate = 0.04531871789155581
    try:
        data_whole = np.load('data/data_whole.npy')
    except:
        data_whole = np.load("multi_scale_model/data/data_whole.npy")
    data_whole, mean_whole, std_whole = get_data('dynamic_model/iPSC_data.xlsx')

    batch_data = data_whole[0]
    single_cell_radius = 7.5  # um
    tortuosity = 1.5
    delta_t = 0.5  # h


    # run 24 hours
    agg_density_function = get_aggregation_profile(no_aggregate=False, bioreactor=True)
    agg_simulator, _ = run_simulation(48, 1, mean_whole, cell_growth_rate, agg_density_function, meta_index,
                                      single_cell_radius, parameters, diffusion_coef, porosity, 1, cal_variance=False)

    '''Plot All'''
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    sizes = 15 * np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    colormap = mpl.cm.tab10.colors
    mpl.cm.get_cmap()
    test_index = 1
    has_legend = True
    has_legend_outside = False
    for l, label in enumerate(meta_label):
        for i, size in enumerate(sizes):
            # ini_intra_metabolites = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
            ini_intra_metabolites = agg_simulator.results[48].copy()
            # ini_intra_metabolites[[28]] = ini_intra_metabolites[[28]] * 2
            # ini_intra_metabolites[meta_label_index['GLC'][0]] = 20
            # ini_intra_metabolites[meta_label_index['ELAC'][0]] = 10
            agg = CellModel(
                size,
                ini_intra_metabolites,
                int(single_cell_radius),
                parameters,
                np.array(diffusion_coef) * porosity / tortuosity,  # 1000 * 60 * 24, # um/h
                meta_index,
                delta_t
            )
            extracellular_metabolites = ini_intra_metabolites[meta_index]
            rates, conc = agg.get_flux_rate_cell_level(extracellular_metabolites)

            x = agg.radii
            y = np.concatenate([np.array(conc)[:(len(x) - 1), l], [extracellular_metabolites[l]]])

            plt.plot(x, y, '-', color=colormap[i], alpha=0.9, label=size, linewidth=3)

        plt.grid(axis='y', color='0.95')
        if has_legend and not has_legend_outside:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=4)
            plt.legend(title='Aggregate Radius', fontsize=11)
        elif has_legend_outside:
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', title='Aggregate Radius')

        plt.xlabel('Distance from Center ' + r'$(\mu m)$', fontsize=14)
        plt.ylabel('Concentration (mM)', fontsize=14)
        plt.title(label, fontsize=14)
        plt.tight_layout()
        # if has_legend and not has_legend_outside:
        #     plt.savefig("multi_scale_model/result/reaction-diffusion/{}.svg".format(label))
        # elif has_legend_outside:
        #     plt.savefig("multi_scale_model/result/reaction-diffusion/{}-legend-outside.svg".format(label))
        # else:
        #     plt.savefig("multi_scale_model/result/reaction-diffusion/{}-no-legend.svg".format(label))
        plt.show()

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    sizes = np.arange(30, 601, 15)  # 15 * np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    lac_conc = list(range(0, 61, 2))
    colormap = mpl.cm.tab10.colors
    mpl.cm.get_cmap()
    heatmap = []
    test_index = 1
    lactate_upper = 40
    glucose_lower = 2.5
    for l, lac in enumerate(lac_conc):
        fraction_unhealth = []
        for i, size in enumerate(sizes):
            # ini_intra_metabolites = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
            ini_intra_metabolites = agg_simulator.results[24].copy()
            ini_intra_metabolites[meta_label_index['GLC'][0]] = 20
            ini_intra_metabolites[meta_label_index['ELAC'][0]] = lac
            agg = CellModel(
                size,
                ini_intra_metabolites,
                int(single_cell_radius),
                parameters,
                np.array(diffusion_coef) * porosity / tortuosity,  # 1000 * 60 * 24, # um/h
                meta_index,
                delta_t
            )
            extracellular_metabolites = ini_intra_metabolites[meta_index].copy()
            rates, conc = agg.get_flux_rate_cell_level(extracellular_metabolites)
            G_l = get_fraction_cell_in_agg_l(7.5, porosity, agg.radii)
            x = agg.radii
            unhealthy_cell_l = 0
            for k, meta_conc in enumerate(conc):
                if meta_conc[meta_label_index['GLC'][1]] <= glucose_lower or meta_conc[meta_label_index['ELAC'][1]] > lactate_upper:
                    unhealthy_cell_l += G_l[k + 1]
            fraction_unhealth.append(unhealthy_cell_l)
        heatmap.append(fraction_unhealth)

    heatmap = np.array(heatmap)
    import seaborn as sns

    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap * 100, cmap=list(reversed(sns.color_palette("RdBu", 10))), annot=False, vmin=0, vmax=100,
                     linewidth=0.5, xticklabels=sizes, yticklabels=lac_conc)
    plt.xlabel(r'Aggregate Radius ($\mu$m)', fontsize=14)
    plt.ylabel('Lactate Concentration (mM)', fontsize=14)
    ax.collections[0].colorbar.set_label("Unhealthy Cells (%)", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("multi_scale_model/result/unhealthy/{}-{}-{}.pdf".format('Lac', lactate_upper, glucose_lower),
                bbox_inches='tight', format='pdf')
    plt.savefig("multi_scale_model/result/unhealthy/{}-{}-{}.svg".format('Lac', lactate_upper, glucose_lower),
                bbox_inches='tight', format='svg')
    plt.show()

    #### Glucose
    sizes = np.arange(30, 601, 15)  # 15 * np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    glc_conc = list(range(0, 31, 1))
    colormap = mpl.cm.tab10.colors
    mpl.cm.get_cmap()
    heatmap = []
    test_index = 1
    lactate_upper = 40
    glucose_lower = 2.5
    for l, glc in enumerate(glc_conc):
        fraction_unhealth = []
        for i, size in enumerate(sizes):
            ini_intra_metabolites = agg_simulator.results[0].copy()
            ini_intra_metabolites[meta_label_index['GLC'][0]] = glc
            ini_intra_metabolites[meta_label_index['ELAC'][0]] = 0
            agg = CellModel(
                size,
                ini_intra_metabolites,
                int(single_cell_radius),
                parameters,
                np.array(diffusion_coef) * porosity / tortuosity,  # 1000 * 60 * 24, # um/h
                meta_index,
                delta_t
            )
            extracellular_metabolites = ini_intra_metabolites[meta_index].copy()
            rates, conc = agg.get_flux_rate_cell_level(extracellular_metabolites)
            G_l = get_fraction_cell_in_agg_l(7.5, porosity, agg.radii)
            x = agg.radii
            unhealthy_cell_l = 0
            for k, meta_conc in enumerate(conc):
                if meta_conc[meta_label_index['GLC'][1]] <= glucose_lower or meta_conc[
                    meta_label_index['ELAC'][1]] > lactate_upper:
                    unhealthy_cell_l += G_l[
                        k + 1]  # glucose index is 3 and lactate index is 6 in extracellular metabolite list
            fraction_unhealth.append(unhealthy_cell_l)
        heatmap.append(fraction_unhealth)

    heatmap = np.array(heatmap)
    import seaborn as sns

    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap * 100, cmap=list(reversed(sns.color_palette("RdBu", 10))), annot=False, vmin=0, vmax=100,
                     linewidth=0.5, xticklabels=sizes,
                     yticklabels=glc_conc)
    plt.xlabel(r'Aggregate Radius ($\mu$m)', fontsize=14)
    plt.ylabel('Glucose Concentration (mM)', fontsize=14)
    ax.collections[0].colorbar.set_label("Unhealthy Cells (%)", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("multi_scale_model/result/unhealthy/{}-{}-{}.pdf".format('Glc', lactate_upper, glucose_lower),
                bbox_inches='tight', format='pdf')
    plt.savefig("multi_scale_model/result/unhealthy/{}-{}-{}.svg".format('Glc', lactate_upper, glucose_lower),
                bbox_inches='tight', format='svg')
    plt.show()

    plt.rcParams.update({'font.size': 12})

    '''Plot biomass flux rate'''
    #### Lactate
    sizes = np.arange(30, 601, 15)  # 15 * np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    lac_conc = list(range(0, 61, 2))
    colormap = mpl.cm.tab10.colors
    mpl.cm.get_cmap()
    heatmap = []
    test_index = 1
    for l, lac in enumerate(lac_conc):
        fraction_unhealth = []
        for i, size in enumerate(sizes):
            # ini_intra_metabolites = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
            ini_intra_metabolites = agg_simulator.results[24].copy()
            ini_intra_metabolites[meta_label_index['GLC'][0]] = 25
            ini_intra_metabolites[meta_label_index['ELAC'][0]] = lac
            ini_intra_metabolites[meta_label_index['EALA'][0]] = 0.1

            agg = CellModel(
                size,
                ini_intra_metabolites,
                int(single_cell_radius),
                parameters,
                np.array(diffusion_coef) * porosity / tortuosity,  # 1000 * 60 * 24, # um/h
                meta_index,
                delta_t
            )
            extracellular_metabolites = ini_intra_metabolites[meta_index].copy()
            rates, conc = agg.get_flux_rate_cell_level(extracellular_metabolites)

            G_l = get_fraction_cell_in_agg_l(7.5, porosity, agg.radii)
            x = agg.radii
            unhealthy_cell_l = 0
            for k, meta_conc in enumerate(conc):
                if meta_conc[3] > 2.5 or meta_conc[6] < 40:
                    unhealthy_cell_l += G_l[k] * rates[
                        k, -1]  # glucose index is 3 and lactate index is 6 in extracellular metabolite list
            fraction_unhealth.append(unhealthy_cell_l)
        heatmap.append(fraction_unhealth)

    heatmap = np.array(heatmap)
    import seaborn as sns

    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap, cmap='coolwarm', annot=False, linewidth=0.5, xticklabels=sizes, yticklabels=lac_conc)
    plt.xlabel(r'Aggregate Radius ($\mu$m)')
    plt.ylabel('Lactate Concentration (mM)')
    plt.tight_layout()
    plt.savefig("multi_scale_model/result/optimal_size/{}-{}-{}.pdf".format('Lac', lactate_upper, glucose_lower),
                bbox_inches='tight', format='pdf')
    plt.savefig("multi_scale_model/result/optimal_size/{}-{}-{}.svg".format('Lac', lactate_upper, glucose_lower),
                bbox_inches='tight', format='svg')
    plt.show()

    #### Glucose
    sizes = np.arange(30, 601, 15)  # 15 * np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    glc_conc = list(range(0, 61, 2))
    colormap = mpl.cm.tab10.colors
    mpl.cm.get_cmap()
    heatmap = []
    test_index = 1
    for l, glc in enumerate(glc_conc):
        fraction_unhealth = []
        for i, size in enumerate(sizes):
            # ini_intra_metabolites = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
            ini_intra_metabolites = agg_simulator.results[24].copy()
            ini_intra_metabolites[meta_label_index['GLC'][0]] = glc
            ini_intra_metabolites[meta_label_index['ELAC'][0]] = 6
            ini_intra_metabolites[meta_label_index['EALA'][0]] = 0.1
            agg = CellModel(
                size,
                ini_intra_metabolites,
                int(single_cell_radius),
                parameters,
                np.array(diffusion_coef) * porosity / tortuosity,  # 1000 * 60 * 24, # um/h
                meta_index,
                delta_t
            )
            extracellular_metabolites = ini_intra_metabolites[meta_index].copy()
            rates, conc = agg.get_flux_rate_cell_level(extracellular_metabolites)

            G_l = get_fraction_cell_in_agg_l(7.5, porosity, agg.radii)
            x = agg.radii
            unhealthy_cell_l = 0
            for k, meta_conc in enumerate(conc):
                # if meta_conc[3] > 2.5 or meta_conc[6] < 40:
                unhealthy_cell_l += G_l[k] * rates[
                    k, -1]  # glucose index is 3 and lactate index is 6 in extracellular metabolite list
            fraction_unhealth.append(unhealthy_cell_l)
        heatmap.append(fraction_unhealth)

    heatmap = np.array(heatmap)
    import seaborn as sns

    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap, cmap='coolwarm', annot=False, linewidth=0.5, xticklabels=sizes, yticklabels=glc_conc)
    plt.xlabel(r'Aggregate Radius ($\mu$m)')
    plt.ylabel('Glucose Concentration (mM)')
    plt.tight_layout()
    plt.savefig("multi_scale_model/result/optimal_size/{}-{}-{}.pdf".format('Glc', lactate_upper, glucose_lower),
                bbox_inches='tight', format='pdf')
    plt.savefig("multi_scale_model/result/optimal_size/{}-{}-{}.svg".format('Glc', lactate_upper, glucose_lower),
                bbox_inches='tight', format='svg')
    plt.show()

    '''Plot flux rate'''
    sizes = np.arange(30, 601, 15)  # 15 * np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    lac_conc = list(range(0, 100, 2))
    colormap = mpl.cm.tab10.colors
    mpl.cm.get_cmap()
    heatmap = []
    ala_conc = 0.1
    test_index = 1
    fraction_unhealth, fraction_ser_avg_l = [], []

    for i, size in enumerate(sizes):
        # ini_intra_metabolites = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
        ini_intra_metabolites = agg_simulator.results[48].copy()
        # ini_intra_metabolites[meta_label_index['GLC'][0]] = 15
        # ini_intra_metabolites[meta_label_index['ELAC'][0]] = 5
        # ini_intra_metabolites[meta_label_index['EALA'][0]] = ala_conc
        agg = CellModel(
            size,
            ini_intra_metabolites,
            int(single_cell_radius),
            parameters,
            np.array(diffusion_coef) * porosity / tortuosity,  # 1000 * 60 * 24, # um/h
            meta_index,
            delta_t
        )
        extracellular_metabolites = ini_intra_metabolites[meta_index]
        rates, conc = agg.get_flux_rate_cell_level(extracellular_metabolites)

        G_l = get_fraction_cell_in_agg_l(7.5, porosity, agg.radii)
        x = agg.radii
        unhealthy_cell_l = np.zeros(rates.shape[-1])
        ser_avg_l = np.zeros(len(meta_label))
        for k, meta_conc in enumerate(conc):
            # if meta_conc[3] > 2.5 or meta_conc[6] < 16:
            ser_avg_l = ser_avg_l + G_l[k + 1] * meta_conc
            # unhealthy_cell_l = unhealthy_cell_l + G_l[k + 1] * rates[k, :]
            unhealthy_cell_l = unhealthy_cell_l + 1 / len(conc) * rates[k, :]
        fraction_unhealth.append(unhealthy_cell_l)
        fraction_ser_avg_l.append(ser_avg_l)

    heatmap = np.array(fraction_unhealth)
    heatmap_ser = np.array(fraction_ser_avg_l)
    heatmap_norm = (heatmap - heatmap.mean(axis=0)) / heatmap.std(axis=0)
    heatmap_ser_norm = (heatmap_ser - heatmap_ser.mean(axis=0)) / heatmap_ser.std(axis=0)

    # fig, ax = plt.subplots()
    # ax = sns.heatmap(heatmap_norm.T,
    #                  cmap='coolwarm',
    #                  annot=False,
    #                  linewidth=0.5,
    #                  xticklabels=sizes,
    #                  yticklabels=list(range(1, heatmap_norm.shape[0] + 1)))
    # plt.xlabel(r'Aggregate Radius ($\mu$m)')
    # plt.title('Average Flux Rates (after standardization)')
    # plt.tight_layout()
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # ax = sns.heatmap(heatmap_norm.T, cmap='coolwarm', annot=False, linewidth=0.5, xticklabels=sizes, yticklabels=np.array(agg.flux_rate_solver.reaction))
    # plt.xlabel(r'Aggregate Radius ($\mu$m)')
    # plt.title('Average Flux Rates (after standardization)')
    # plt.tight_layout()
    # plt.show()

    selected_reaction = [0, 7, 9, 24, 25, 27, 32, 33, 35, 36, 38]
    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap_norm.T[selected_reaction, :], cmap=list(reversed(sns.color_palette("RdBu", 10))),
                     annot=False, linewidth=0.5, xticklabels=sizes,
                     yticklabels=np.array(agg.flux_rate_solver.reaction)[selected_reaction])
    plt.xlabel(r'Aggregate Radius ($\mu$m)', fontsize=14)
    plt.title('Averaged Flux Rates (after standardization)')
    plt.tight_layout()
    plt.savefig("multi_scale_model/result/optimal_size/{}-{}-ala-{}.pdf".format('Biomass', 'reaction', ala_conc),
                bbox_inches='tight', format='pdf')
    plt.savefig("multi_scale_model/result/optimal_size/{}-{}-ala-{}.svg".format('Biomass', 'reaction', ala_conc),
                bbox_inches='tight', format='svg')
    plt.show()

    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap_ser_norm.T, cmap=list(reversed(sns.color_palette("RdBu", 10))), annot=False, linewidth=0.5,
                     xticklabels=sizes, yticklabels=meta_label)
    plt.xlabel(r'Aggregate Radius ($\mu$m)', fontsize=14)
    plt.title('Averaged Extracellular Metabolite Concentrations \n (after standardization)')
    plt.tight_layout()
    plt.savefig("multi_scale_model/result/optimal_size/{}-{}-ala-{}.pdf".format('Biomass', 'metabolites', ala_conc),
                bbox_inches='tight', format='pdf')
    plt.savefig("multi_scale_model/result/optimal_size/{}-{}-ala-{}.svg".format('Biomass', 'metabolites', ala_conc),
                bbox_inches='tight', format='svg')
    plt.show()

    ### Ala experiment
    sizes = np.arange(30, 601, 15)  # 15 * np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    colormap = mpl.cm.tab10.colors
    mpl.cm.get_cmap()
    test_index = 1
    growth = []
    for ala in range(12):
        fraction_unhealth = []
        for i, size in enumerate(sizes):
            ini_intra_metabolites = agg_simulator.results[24].copy()
            ini_intra_metabolites[meta_label_index['GLC'][0]] = 25
            ini_intra_metabolites[meta_label_index['ELAC'][0]] = 5
            ini_intra_metabolites[meta_label_index['EALA'][0]] = 0.03 * (ala + 1)
            agg = CellModel(
                size,
                ini_intra_metabolites,
                int(single_cell_radius),
                parameters,
                np.array(diffusion_coef) * porosity / tortuosity,  # 1000 * 60 * 24, # um/h
                meta_index,
                delta_t
            )
            extracellular_metabolites = ini_intra_metabolites[meta_index].copy()
            rates, conc = agg.get_flux_rate_cell_level(extracellular_metabolites)

            G_l = get_fraction_cell_in_agg_l(7.5, porosity, agg.radii)
            x = agg.radii
            unhealthy_cell_l = 0
            for k, meta_conc in enumerate(conc):
                # if meta_conc[3] > 2.5 or meta_conc[6] < 16:
                unhealthy_cell_l += G_l[k] * rates[
                    k, -1]  # # glucose index is 3 and lactate index is 6 in extracellular metabolite list
            fraction_unhealth.append(unhealthy_cell_l)
        growth.append(fraction_unhealth)

        growth = np.array(growth)
        growth = growth * 1e6  # / np.std(growth)
        fig, ax = plt.subplots()
        ax = sns.heatmap(growth, cmap='coolwarm', annot=False, linewidth=0.5, xticklabels=sizes,
                         yticklabels=['{}'.format(round(0.03 * (ala + 1), 5)) for ala in range(12)])
        plt.xlabel(r'Aggregate Radius ($\mu$m)', fontsize=14)
        plt.ylabel('Bulk Alanine Concentration (mM)', fontsize=14)
        plt.title(r'Averaged Biomass Flux Rate (nmol/$10^6$ cells/h)')
        plt.tight_layout()
        # plt.savefig("multi_scale_model/result/optimal_size/{}-{}.pdf".format('Biomass', 'Alanine'),
        #             bbox_inches='tight', format='pdf')
        # plt.savefig("multi_scale_model/result/optimal_size/{}-{}.svg".format('Biomass', 'Alanine'),
        #             bbox_inches='tight', format='svg')
        plt.show()

    ### EASP experiment
    sizes = np.arange(30, 601, 15)  # 15 * np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    colormap = mpl.cm.tab10.colors
    mpl.cm.get_cmap()
    heatmap = []

    test_index = 1
    growth = []
    for asp in range(12):
        fraction_unhealth = []
        for i, size in enumerate(sizes):
            ini_intra_metabolites = agg_simulator.results[24].copy()
            ini_intra_metabolites[meta_label_index['GLC'][0]] = 25
            ini_intra_metabolites[meta_label_index['ELAC'][0]] = 5
            ini_intra_metabolites[meta_label_index['EASP'][0]] = 0.03 * (asp + 1)
            agg = CellModel(
                size,
                ini_intra_metabolites,
                int(single_cell_radius),
                parameters,
                np.array(diffusion_coef) * porosity / tortuosity,  # 1000 * 60 * 24, # um/h
                meta_index,
                delta_t
            )
            extracellular_metabolites = ini_intra_metabolites[meta_index]
            rates, conc = agg.get_flux_rate_cell_level(extracellular_metabolites)

            G_l = get_fraction_cell_in_agg_l(7.5, porosity, agg.radii)
            x = agg.radii
            unhealthy_cell_l = 0
            for k, meta_conc in enumerate(conc):
                # if meta_conc[3] > 2.5 or meta_conc[6] < 16:
                unhealthy_cell_l += G_l[k] * rates[
                    k, -1]  # # glucose index is 3 and lactate index is 6 in extracellular metabolite list
            fraction_unhealth.append(unhealthy_cell_l)
        growth.append(fraction_unhealth)

        growth = np.array(growth)
        growth = growth * 1e6
        fig, ax = plt.subplots()
        ax = sns.heatmap(growth, cmap='coolwarm', annot=False, linewidth=0.5, xticklabels=sizes,
                         yticklabels=['{}'.format(round(0.03 * (ala + 1), 5)) for ala in range(12)])
        plt.xlabel(r'Aggregate Radius ($\mu$m)', fontsize=14)
        plt.ylabel('Bulk Aspartate Concentration (mM)', fontsize=14)
        plt.title(r'Averaged Biomass Flux Rate (nmol/$10^6$ cells/h)')
        plt.tight_layout()
        plt.savefig("multi_scale_model/result/optimal_size/{}-{}.pdf".format('Biomass', 'Aspartate'),
                    bbox_inches='tight', format='pdf')
        plt.savefig("multi_scale_model/result/optimal_size/{}-{}.svg".format('Biomass', 'Aspartate'),
                    bbox_inches='tight', format='svg')
        plt.show()

    ### porosity experiment
    sizes = np.arange(30, 601, 15)  # 15 * np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    colormap = mpl.cm.tab10.colors
    mpl.cm.get_cmap()
    heatmap = []

    test_index = 1
    growth = []
    porosity_list = [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.29, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34]
    for porosity in porosity_list:
        fraction_unhealth = []
        for i, size in enumerate(sizes):
            ini_intra_metabolites = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
            ini_intra_metabolites[39] = 25
            ini_intra_metabolites[42] = 5
            ini_intra_metabolites[28] = 0.1
            agg = CellModel(
                size,
                ini_intra_metabolites,
                int(single_cell_radius),
                parameters,
                np.array(diffusion_coef) * porosity / tortuosity,  # 1000 * 60 * 24, # um/h
                meta_index,
                delta_t
            )
            extracellular_metabolites = ini_intra_metabolites[meta_index]
            rates, conc = agg.get_flux_rate_cell_level(extracellular_metabolites)

            G_l = get_fraction_cell_in_agg_l(7.5, porosity, agg.radii)
            x = agg.radii
            unhealthy_cell_l = 0
            for k, meta_conc in enumerate(conc):
                # if meta_conc[3] > 2.5 or meta_conc[6] < 16:
                unhealthy_cell_l += G_l[k] * rates[
                    k, -1]  # # glucose index is 3 and lactate index is 6 in extracellular metabolite list
            fraction_unhealth.append(unhealthy_cell_l)
        growth.append(fraction_unhealth)

        growth = np.array(growth)
        growth = growth * 1e6
        fig, ax = plt.subplots()
        ax = sns.heatmap(growth, cmap='coolwarm', annot=False, linewidth=0.5, xticklabels=sizes,
                         yticklabels=porosity_list)
        plt.xlabel(r'Aggregate Radius ($\mu$m)')
        plt.ylabel('Porosity')
        plt.title('Averaged Biomass Flux Rate ')
        plt.tight_layout()
        plt.show()

    ### tortuosity experiment
    sizes = np.arange(30, 601, 15)  # 15 * np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    colormap = mpl.cm.tab10.colors
    mpl.cm.get_cmap()
    heatmap = []

    test_index = 1
    growth = []
    tortuosity_list = [1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7]
    for tortuosity in tortuosity_list:
        fraction_unhealth = []
        for i, size in enumerate(sizes):
            ini_intra_metabolites = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
            ini_intra_metabolites[39] = 25
            ini_intra_metabolites[42] = 5
            ini_intra_metabolites[28] = 0.1
            porosity = 0.27  # 60 rpm, and 0.229 100 rpm
            agg = CellModel(
                size,
                ini_intra_metabolites,
                int(single_cell_radius),
                parameters,
                np.array(diffusion_coef) * porosity / tortuosity,  # 1000 * 60 * 24, # um/h
                meta_index,
                delta_t
            )
            extracellular_metabolites = ini_intra_metabolites[meta_index]
            rates, conc = agg.get_flux_rate_cell_level(extracellular_metabolites)

            G_l = get_fraction_cell_in_agg_l(7.5, porosity, agg.radii)
            x = agg.radii
            unhealthy_cell_l = 0
            for k, meta_conc in enumerate(conc):
                # if meta_conc[3] > 2.5 or meta_conc[6] < 16:
                unhealthy_cell_l += G_l[k] * rates[
                    k, -1]  # # glucose index is 3 and lactate index is 6 in extracellular metabolite list
            fraction_unhealth.append(unhealthy_cell_l)
        growth.append(fraction_unhealth)

        growth = np.array(growth)

        fig, ax = plt.subplots()
        ax = sns.heatmap(growth, cmap='coolwarm', annot=False, linewidth=0.5, xticklabels=sizes,
                         yticklabels=tortuosity_list)
        plt.xlabel(r'Aggregate Radius ($\mu$m)')
        plt.ylabel('Tortuosity')
        plt.title('Averaged Biomass Flux Rate')
        plt.tight_layout()
        plt.show()

        ''' Variance '''
        mr_biomass_mean = []
        rsd = []
        colormap = mpl.cm.tab10.colors
        mpl.cm.get_cmap()
        cur_hour = 24
        for _ in range(10):
            var_metabolite, between_shell_covs, sample_means = [], [], []
            sizes = np.arange(30, 601, 15)  # 15 * np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
            delta_t = 1
            for i, size in enumerate(sizes):
                # ini_intra_metabolites = np.hstack([S2_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
                ini_intra_metabolites = agg_simulator.results[cur_hour].copy()
                # ini_intra_metabolites[meta_label_index['EALA'][0]] = 10
                # ini_intra_metabolites[meta_label_index['SER'][0]] = 0.5
                # ini_intra_metabolites[meta_index] = np.hstack([S2_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])[meta_index]
                # ini_intra_metabolites[meta_label_index['GLC'][0]] = 20
                # ini_intra_metabolites[meta_label_index['ELAC'][0]] = 5
                # np.array(list(meta.values()))[ini_intra_metabolites<0] # check which metabolite are consumed
                agg = CellModel(
                    size,
                    ini_intra_metabolites,
                    int(single_cell_radius),
                    parameters,
                    np.array(diffusion_coef) * porosity / tortuosity,  # 1000 * 60 * 24, # um/h
                    meta_index,
                    delta_t
                )
                extracellular_metabolites = ini_intra_metabolites[meta_index]
                rates, conc = agg.get_flux_rate_cell_level(extracellular_metabolites)
                var, _, sample_mean = agg.get_covariance(rates, delta_t, None)
                var_metabolite.append(var)
                sample_means.append(sample_mean)
                # between_shell_covs.append(between_shell_cov)

            var_metabolite = np.array(var_metabolite) * 1e12
            sample_means = np.array(sample_means) * 1e6
            # between_shell_covs = np.array(between_shell_covs) * 1e6
            biomass_variance = var_metabolite[:, 31]
            # biomass_between_shell_covs = between_shell_covs[:, 44]
            biomass_mean = sample_means[:, 31]
            mr_biomass_mean.append(biomass_mean)
            rsd.append(np.sqrt(biomass_variance) / (biomass_mean) * 100)

        fig, ax = plt.subplots()

        ax.plot(sizes, np.mean(mr_biomass_mean, axis=0), '-', color=colormap[0], alpha=1, linewidth=3,
                label=r'E[$\Delta$ biomass]')
        ax.fill_between(sizes, np.mean(mr_biomass_mean, axis=0) - np.std(mr_biomass_mean, axis=0),
                        np.mean(mr_biomass_mean, axis=0) + np.std(mr_biomass_mean, axis=0),
                        color=colormap[0], alpha=.1)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=4)
        ax.grid(axis='x', color='0.95')
        ax2 = ax.twinx()
        # ax2.ticklabel_format(style='scientific')
        ax2.ticklabel_format(style='sci', scilimits=(-3, 4), axis='both', useMathText=True)
        # ax2.xaxis.major.formatter._useMathText = True
        ax2.plot(sizes, np.mean(rsd, axis=0), '-', color=colormap[3], alpha=1, linewidth=3,
                 label=r'Var[$\Delta$ biomass]')
        ax2.fill_between(sizes, np.mean(rsd, axis=0) - np.std(rsd, axis=0),
                         np.mean(rsd, axis=0) + np.std(rsd, axis=0),
                         color=colormap[3], alpha=.2)
        # ax2.set_ylabel("Measured (FSiPSC, Kwok et al, 2017)", color="blue", fontsize=14)
        ax.set_xlabel('Aggregate Radius ' + r'($\mu$m)', fontsize=14)
        ax.set_ylabel('Mean Biomass Production \n (nmol/$10^6$ cells/h)', color=colormap[0], fontsize=14)
        ax2.set_ylabel('Relative Standard Deviation (%) of \n Biomass Production in One Hour', color=colormap[3],
                       fontsize=14)
        plt.tight_layout()
        # plt.savefig("multi_scale_model/result/variance-component-analysis/{}-{}.pdf".format('mean-variance', cur_hour),
        #             bbox_inches='tight', format='pdf')
        # plt.savefig("multi_scale_model/result/variance-component-analysis/{}-{}.svg".format('mean-variance', cur_hour),
        #             bbox_inches='tight', format='svg')
        plt.show()
