import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from multi_scale_model.metabolic_flux_rate import g

S2_0 = np.array([1.40E-07, 4.00E-07, 2.00E-07, 8.00E-06, 3.00E-07,
                 5.00E-08, 4.00E-07, 7.00E-04, 8.00E-06, 1E-6, 5.50E-03,
                 4.00E-08, 5.00E-05, 1.50E-07, 5.30E-07, 5.30E-07, 5.30E-07, 5.30E-07,
                 5.30E-07, 5.30E-07])
S1_0 = np.array([1.40E-03, 4.00E-03, 2.00E-03, 8.00E-06, 3.00E-03,
                 5.00E-03, 4.00E-03, 7.00E-04, 8.00E-06, 5.50E-03, 5.50E-03,
                 4.00E-08, 5.00E-03, 1.50E-07, 5.30E-03, 5.30E-03, 5.30E-07, 5.30E-07,
                 5.30E-07, 5.30E-07])

meta = {0: 'AcCoA', 1: 'AKG', 2: 'CIT', 3: 'CO2', 4: 'F6P', 5: 'G6P', 6: 'GAP', 7: 'GLU', 8: 'GLY', 9: 'MAL', 10: 'OAA',
        11: 'PEP', 12: 'FUM', 13: 'Ru5P', 14: 'SUC', 15: 'PYR', 16: 'ALA', 17: 'ASP', 18: 'LAC', 19: 'GLN', 20: 'GLY',
        21: 'SER', 22: 'GLC', 23: 'EGLN', 24: 'EGLU', 25: 'EPYR', 26: 'EASP', 27: 'EALA', 28: 'ELAC', 29: 'NH4',
        30: 'LIPID', 31: 'BIO', 32: 'X'}

# meta_index = np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 32])
# meta_index = np.array([27, 26, 21, 22, 23, 24, 28, 29])  # ALA, ASP, SER, GLC, GLN, EGLU, LAC, NH4
meta_label = ['GLY', 'SER', 'GLC', 'EGLN', 'EGLU', 'EPYR', 'EASP', 'EALA', 'ELAC', 'NH4']
meta_index = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
meta_label_index = dict(zip(meta_label, zip(meta_index, range(len(meta_label)))))
diffusion_coef = np.array([1.04, 0.891, 0.6, 0.76, 0.708, 1.12, 0.741, 0.91, 1.033, 1.86])


def get_data(path):
    data_extra_raw = pd.read_excel(path, sheet_name="Extra")
    data_extra = data_extra_raw.values

    #####19 observable state: EPYR, EALA, EASP, EASX, GLY, HIS, ILE, LUE, LYS, SER, TYR, VAL, GLC, EGLN, EGLU, ELAC, NH4, BIO, X
    # data_extra = np.vstack((data_extra[:,6],data_extra[:,10],data_extra[:,12], data_extra[:,11]+data_extra[:,12], data_extra[:,13:18].T, data_extra[:,21], data_extra[:,24:26].T, data_extra[:,4], data_extra[:,7:9].T, data_extra[:,5], data_extra[:,9], data_extra[:,26], data_extra[:,3]))

    # 13 observable state: GLY, SER, GLC, EGLN, EGLU, EPYR, EASP, EALA, ELAC, NH4, LIPID, Bio, X

    data_extra = np.vstack(
        (data_extra[:, 13], data_extra[:, 21], data_extra[:, 4], data_extra[:, 7], data_extra[:, 8], data_extra[:, 6],
         data_extra[:, 12], data_extra[:, 10], data_extra[:, 5], data_extra[:, 9], data_extra[:, 27], data_extra[:, 26],
         data_extra[:, 3]))
    data_extra = data_extra.T
    data_extra = np.array(data_extra, dtype=float)

    # Measurements under different DoE
    data_extra_HGHL = data_extra[:30, :]
    data_extra_HGLL = data_extra[30:60, :]
    data_extra_LGLL = data_extra[60:90, :]
    data_extra_LGHL = data_extra[90:, :]
    data_whole = [data_extra_HGHL, data_extra_HGLL, data_extra_LGLL, data_extra_LGHL]

    # Measurement Time
    t = [0, 12, 24, 36, 48]
    size = len(t)

    # mean and measurement std for each metabolite at each time point
    mean_HGHL = np.zeros([5, 13])
    mean_HGLL = np.zeros([5, 13])
    mean_LGLL = np.zeros([5, 13])
    mean_LGHL = np.zeros([5, 13])

    std_HGHL = np.zeros([5, 13])
    std_HGLL = np.zeros([5, 13])
    std_LGLL = np.zeros([5, 13])
    std_LGHL = np.zeros([5, 13])

    for i in range(0, len(t)):
        index = np.arange(i, data_extra_HGHL[:, 1].size, 5)
        mean_HGHL[i] = np.mean(data_extra_HGHL[index,], axis=0)
        mean_HGLL[i] = np.mean(data_extra_HGLL[index,], axis=0)
        mean_LGLL[i] = np.mean(data_extra_LGLL[index,], axis=0)
        mean_LGHL[i] = np.mean(data_extra_LGHL[index,], axis=0)

        std_HGHL[i] = np.std(data_extra_HGHL[index,], axis=0)
        std_HGLL[i] = np.std(data_extra_HGLL[index,], axis=0)
        std_LGLL[i] = np.std(data_extra_LGLL[index,], axis=0)
        std_LGHL[i] = np.std(data_extra_LGHL[index,], axis=0)

    mean_whole = [mean_HGHL, mean_HGLL, mean_LGLL, mean_LGHL]
    std_whole = [std_HGHL, std_HGLL, std_LGLL, std_LGHL]
    return data_whole, mean_whole, std_whole


def plot_comparison_under_monolayer_condition(test_index, hours, agg_simulator, mean_whole, std_whole, parameters,
                                              draw_measurement=True,
                                              path='multi_scale_model/result/simulation-agg'):
    import os
    N = pd.read_excel('multi_scale_model/data/Simulator_v3.xlsx', sheet_name="N")
    N = N.fillna(0).values
    N = N[:, 1:]
    cases = {0: "HGHL", 1: "HGLL", 2: "LGLL", 3: "LGHL"}
    colormap = mpl.cm.tab10.colors
    t_pred = np.linspace(0, hours, 144)
    initial_states = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
    pred = g((0, hours), initial_states, parameters, N, t_pred).y
    with plt.style.context('_mpl-gallery'):
        # mpl.rcParams['grid.color'] = "black"
        fig, axs = plt.subplots(3, 3, figsize=(10, 8))
        for k, i in enumerate(meta_index[1:]):
            # axs[k // 3, k % 3].plot([0, 12, 24, 36, 48], mean_whole[test_index][:, (i - 27)], 'o', color='blue', label='Measured')
            if draw_measurement:
                axs[k // 3, k % 3].errorbar([0, 12, 24, 36, 48], mean_whole[test_index][:, (i - 20)],
                                            yerr=1.96 * std_whole[test_index][:, (i - 20)] / np.sqrt(6), fmt='o',
                                            color=colormap[0], elinewidth=2,
                                            ms=3, label='Measured')
            axs[k // 3, k % 3].plot(t_pred, pred[i, :], color=colormap[2], label='Mechanistic Model')
            if i == 45:
                axs[k // 3, k % 3].plot(np.array(range(hours)) + 1,
                                        np.array(agg_simulator.results)[:hours, i] * 1e-6, color=colormap[3],
                                        linewidth=3, label='Bio-SoS')
            else:
                axs[k // 3, k % 3].plot(np.array(range(hours)) + 1, np.array(agg_simulator.results)[:hours, i],
                                        color=colormap[3], linewidth=3, label='Bio-SoS')

            axs[k // 3, k % 3].set_title(str(meta[i]))
            axs[k // 3, k % 3].set_xticks(list(range(0, hours + 1, 12)), fontsize=14)

        fig.supxlabel('Time (h)', fontsize=18)
        fig.supylabel('Concentration (mM)', fontsize=18)
        plt.figlegend(*axs[0, 0].get_legend_handles_labels(), bbox_to_anchor=(0.5, 1.), ncol=3, loc='upper center',
                      fancybox=True, shadow=True, fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        plt.savefig(
            os.path.join(path, "{}-{}-draw_measurement-{}.pdf".format(cases[test_index], meta[i], draw_measurement)),
            bbox_inches='tight', format='pdf')
        plt.savefig(
            os.path.join(path, "{}-{}-draw_measurement-{}.svg".format(cases[test_index], meta[i], draw_measurement)),
            bbox_inches='tight', format='svg')
        plt.savefig(
            os.path.join(path, "{}-{}-draw_measurement-{}.png".format(cases[test_index], meta[i], draw_measurement)),
            bbox_inches='tight', format='png')
        plt.show()
        plt.clf()


def plot_comparison_under_monolayer_condition_macro(test_index, hours, g, agg_simulators,
                                                    mean_whole, std_whole, parameters,
                                                    draw_measurement=True,
                                                    path='multi_scale_model/result/simulation-agg'):
    import os
    N = pd.read_excel('multi_scale_model/data/Simulator_v3.xlsx', sheet_name="N")
    N = N.fillna(0).values
    N = N[:, 1:]
    cases = {0: "HGHL", 1: "HGLL", 2: "LGLL", 3: "LGHL"}

    t_pred = np.linspace(0, hours, 144)
    initial_states = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
    pred = g((0, hours), initial_states, parameters, N, t_pred).y
    with plt.style.context('_mpl-gallery'):
        # mpl.rcParams['grid.color'] = "black"
        fig, axs = plt.subplots(3, 3, figsize=(10, 8))
        for k, i in enumerate(meta_index[1:]):
            if draw_measurement:
                axs[k // 3, k % 3].errorbar([0, 12, 24, 36, 48], mean_whole[test_index][:, (i - 20)],
                                            yerr=1.96 * std_whole[test_index][:, (i - 20)] / np.sqrt(6), fmt='o',
                                            color='blue', elinewidth=2,
                                            ms=3, label='Measured')
            axs[k // 3, k % 3].plot(t_pred, pred[i, :], color='green', label='Mechanistic Model')
            trajectories = np.array([np.array(agg_simulator.results)[:hours, i] for agg_simulator in agg_simulators])
            axs[k // 3, k % 3].plot(np.array(range(hours)) + 1, trajectories.mean(axis=0),
                                    color='red', linewidth=3, label='Bio-SoS')
            axs[k // 3, k % 3].fill_between(
                np.array(range(hours)) + 1,
                trajectories.mean(axis=0) - 1.96 * trajectories.std(axis=0) / np.sqrt(len(agg_simulators)),
                trajectories.mean(axis=0) + 1.96 * trajectories.std(axis=0) / np.sqrt(len(agg_simulators)),
                color='red',
                alpha=0.2
            )

            axs[k // 3, k % 3].set_title(str(meta[i]))
            axs[k // 3, k % 3].set_xticks(list(range(0, hours + 1, 12)), fontsize=14)

        fig.supxlabel('Time (h)', fontsize=18)
        fig.supylabel('Concentration (mM)', fontsize=18)
        plt.figlegend(*axs[0, 0].get_legend_handles_labels(), bbox_to_anchor=(0.5, 1.), ncol=3, loc='upper center',
                      fancybox=True, shadow=True, fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        plt.savefig(
            os.path.join(path, "{}-{}-draw_measurement-{}.pdf".format(cases[test_index], meta[i], draw_measurement)),
            bbox_inches='tight', format='pdf')
        plt.savefig(
            os.path.join(path, "{}-{}-draw_measurement-{}.svg".format(cases[test_index], meta[i], draw_measurement)),
            bbox_inches='tight', format='svg')
        plt.savefig(
            os.path.join(path, "{}-{}-draw_measurement-{}.png".format(cases[test_index], meta[i], draw_measurement)),
            bbox_inches='tight', format='png')
        plt.show()
        plt.clf()


def generate_inner_outer_cell_metabolism(agg_simulator, size=16):
    s = agg_simulator.extra_metabolites
    metabolite_conc = []
    flux_rates = []
    for i in reversed(range(1, len(agg_simulator.aggregates[size].radii))):
        print(s)
        agg_simulator.aggregates[size].intra_metabolites[i - 1][
            agg_simulator.aggregates[
                size].meta_index] = s  # update the intracelluar metabolites with extracellular conditions
        rate = np.array(agg_simulator
                        .aggregates[size]
                        .flux_rate_solver
                        .estimate_flux_rate(agg_simulator.aggregates[size].intra_metabolites[i - 1])[:-1])
        flux_rates.append(rate)
        unit_factor = 3 / 4 / np.pi / 7.5 ** 3 / agg_simulator.aggregates[size].porosity * 1.1 * 1e-3
        reaction_rate = agg_simulator.aggregates[size].stoichiometric_matrix @ rate * unit_factor
        if len(agg_simulator.aggregates[size].radii) == 1:
            s = agg_simulator.aggregates[size].reaction_diffusion(s, reaction_rate[
                agg_simulator.aggregates[size].meta_index],
                                                                  agg_simulator.aggregates[size].diffusion_coef,
                                                                  agg_simulator.aggregates[size].radii[i], 0)
        else:
            s = agg_simulator.aggregates[size].reaction_diffusion(s, reaction_rate[
                agg_simulator.aggregates[size].meta_index],
                                                                  agg_simulator.aggregates[size].diffusion_coef,
                                                                  agg_simulator.aggregates[size].radii[i],
                                                                  agg_simulator.aggregates[size].radii[i - 1])
        metabolite_conc.append(s)
    # self.update_intracellular_conditions(list(reversed(rates)), self.delta_t)
    outer_cell = flux_rates[0] * 1e3
    inner_cell = flux_rates[-1] * 1e3
    return outer_cell, inner_cell


hgHL = {'Ser.x': [0.357, 0.308, 0.315, 0.240, 0.193],
        'CD': [0.034, 0.070, 0.128, 0.193, 0.260],
        'Glc.x': [18.690,
                  17.542,
                  16.308,
                  15.117,
                  13.702],
        'Gln.x': [2.513,
                  2.323,
                  2.122,
                  1.925,
                  1.683],
        'Pyr.x': [0.422,
                  0.350,
                  0.300,
                  0.244,
                  0.214],
        'Asp.x': [0.025,
                  0.020,
                  0.020,
                  0.022,
                  0.020],
        'Ala.x': [0.020,
                  0.052,
                  0.082,
                  0.133,
                  0.168],
        'Glu.x': [0.033,
                  0.063,
                  0.103,
                  0.147,
                  0.182],
        'Lac.x': [21.352,
                  22.108,
                  24.182,
                  26.458,
                  27.772]}
hgLL = {'Ser.x': [0.392, 0.367, 0.313, 0.265, 0.202],
        'CD': [0.034, 0.065, 0.123, 0.200, 0.285],
        'Glc.x': [18.715, 17.395, 15.910, 14.267, 12.617],
        'Gln.x': [2.543, 2.318, 2.092, 1.835, 1.537],
        'Pyr.x': [0.415, 0.327, 0.218, 0.137, 0.132],
        'Asp.x': [0.037, 0.030, 0.030, 0.027, 0.022],
        'Ala.x': [0.032, 0.058, 0.103, 0.153, 0.182],
        'Glu.x': [0.032, 0.070, 0.118, 0.170, 0.208],
        'Lac.x': [0.000, 1.610, 4.070, 7.123, 9.325]}

lgLL = {'Ser.x': [0.392, 0.317, 0.282, 0.247, 0.185],
        'CD': [0.034, 0.067, 0.107, 0.188, 0.322],
        'Glc.x': [5.993, 4.905, 3.470, 1.720, 0.242],
        'Gln.x': [2.547, 2.338, 2.113, 1.820, 1.533],
        'Pyr.x': [0.428, 0.346, 0.224, 0.140, 0.127],
        'Asp.x': [0.033, 0.030, 0.025, 0.020, 0.023],
        'Ala.x': [0.027, 0.055, 0.097, 0.143, 0.198],
        'Glu.x': [0.042, 0.075, 0.153, 0.167, 0.210],
        'Lac.x': [0.000, 1.647, 4.072, 6.735, 8.953]}

lgHL = {'Ser.x': [0.332, 0.292, 0.232, 0.177, 0.178],
        'CD': [0.034, 0.063, 0.112, 0.180, 0.329],
        'Glc.x': [5.977, 5.078, 3.942, 2.450, 1.212],
        'Gln.x': [2.548, 2.347, 2.165, 1.918, 1.695],
        'Pyr.x': [0.434, 0.352, 0.301, 0.254, 0.223],
        'Asp.x': [0.020, 0.020, 0.018, 0.015, 0.018],
        'Ala.x': [0.020, 0.048, 0.082, 0.123, 0.182],
        'Glu.x': [0.032, 0.068, 0.105, 0.147, 0.182],
        'Lac.x': [21.387, 22.108, 24.310, 26.422, 28.253]}

real_data = {0: hgHL, 1: hgLL, 2: lgLL, 3: lgHL}
