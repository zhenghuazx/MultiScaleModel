import os
import numpy as np

from datetime import date
from argparse import ArgumentParser
import logging
import pandas as pd
import seaborn as sns

from multi_scale_model import Aggregate, get_aggregation_profile, run_simulation, CellModel, get_fraction_cell_in_agg_l, \
    dot_plot
from multi_scale_model.Util import real_data
from multi_scale_model.metabolic_flux_rate import g
from multi_scale_model.Util import *
from multi_scale_model.PBM import CellAggregatePopulation, set_init_distribution, lognorm
from scipy import interpolate
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
logger = logging.getLogger()


def get_health_condition(agg_simulator, parameters, arg):
    sizes = np.arange(30, 601, 15)
    lac_conc = list(range(0, 61, 2))
    colormap = mpl.cm.tab10.colors
    mpl.cm.get_cmap()
    heatmap = []
    lactate_upper = 40
    glucose_lower = 2.5
    for l, lac in enumerate(lac_conc):
        fraction_unhealth = []
        for i, size in enumerate(sizes):
            ini_intra_metabolites = agg_simulator.results[24].copy()
            ini_intra_metabolites[meta_label_index['GLC'][0]] = 20  # set the glucose concentration in medium: 20 mM
            ini_intra_metabolites[meta_label_index['ELAC'][0]] = lac
            agg = CellModel(
                size,
                ini_intra_metabolites,
                int(arg.single_cell_radius),
                parameters,
                np.array(diffusion_coef) * arg.porosity / arg.tortuosity,  # 1000 * 60 * 24, # um/h
                meta_index,
                arg.delta_t
            )
            extracellular_metabolites = ini_intra_metabolites[meta_index].copy()
            rates, conc = agg.get_flux_rate_cell_level(extracellular_metabolites)
            G_l = get_fraction_cell_in_agg_l(7.5, arg.porosity, agg.radii)
            unhealthy_cell_l = 0
            for k, meta_conc in enumerate(conc):
                if meta_conc[meta_label_index['GLC'][1]] <= glucose_lower or meta_conc[
                    meta_label_index['ELAC'][1]] > lactate_upper:
                    unhealthy_cell_l += G_l[k + 1]
            fraction_unhealth.append(unhealthy_cell_l)
        heatmap.append(fraction_unhealth)

    heatmap = np.array(heatmap)

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
            ini_intra_metabolites[meta_label_index['ELAC'][0]] = 0  # remove the lactate in medium
            agg = CellModel(
                size,
                ini_intra_metabolites,
                int(arg.single_cell_radius),
                parameters,
                np.array(diffusion_coef) * arg.porosity / arg.tortuosity,  # 1000 * 60 * 24, # um/h
                meta_index,
                arg.delta_t
            )
            extracellular_metabolites = ini_intra_metabolites[meta_index].copy()
            rates, conc = agg.get_flux_rate_cell_level(extracellular_metabolites)
            G_l = get_fraction_cell_in_agg_l(7.5, arg.porosity, agg.radii)
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

    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap * 100, cmap=list(reversed(sns.color_palette("RdBu", 10))), annot=False, vmin=0,
                     vmax=100,
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


def reaction_diffusion(agg_simulator, parameters, arg):
    # consider 60, 120, ..., 600 um
    sizes = 15 * np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    colormap = mpl.cm.tab10.colors
    mpl.cm.get_cmap()
    has_legend = True
    has_legend_outside = False
    for l, label in enumerate(meta_label):
        for i, size in enumerate(sizes):
            ini_intra_metabolites = agg_simulator.results[48].copy()
            agg = CellModel(
                size,
                ini_intra_metabolites,
                int(arg.single_cell_radius),
                parameters,
                np.array(diffusion_coef) * arg.porosity / arg.tortuosity,  # 1000 * 60 * 24, # um/h
                meta_index,
                arg.delta_t
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
        if has_legend and not has_legend_outside:
            plt.savefig("multi_scale_model/result/reaction-diffusion/{}.svg".format(label))
        elif has_legend_outside:
            plt.savefig("multi_scale_model/result/reaction-diffusion/{}-legend-outside.svg".format(label))
        else:
            plt.savefig("multi_scale_model/result/reaction-diffusion/{}-no-legend.svg".format(label))
        plt.show()


def get_standardized_flux_rate(agg_simulator, parameters, arg):
    sizes = np.arange(30, 601, 15)  # 15 * np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    lac_conc = list(range(0, 100, 2))
    colormap = mpl.cm.tab10.colors
    mpl.cm.get_cmap()
    heatmap = []
    ala_conc = 0.1
    fraction_unhealth, fraction_ser_avg_l = [], []

    for i, size in enumerate(sizes):
        ini_intra_metabolites = agg_simulator.results[48].copy()
        agg = CellModel(
            size,
            ini_intra_metabolites,
            int(arg.single_cell_radius),
            parameters,
            np.array(diffusion_coef) * arg.porosity / arg.tortuosity,  # 1000 * 60 * 24, # um/h
            meta_index,
            arg.delta_t
        )
        extracellular_metabolites = ini_intra_metabolites[meta_index]
        rates, conc = agg.get_flux_rate_cell_level(extracellular_metabolites)

        G_l = get_fraction_cell_in_agg_l(7.5, arg.porosity, agg.radii)
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


def get_biomass_yield_variance_analysis(agg_simulator, parameters, arg, cur_hour=48):
    mr_biomass_mean = []
    rsd = []
    colormap = mpl.cm.tab10.colors
    mpl.cm.get_cmap()
    for _ in range(10):
        var_metabolite, between_shell_covs, sample_means = [], [], []
        sizes = np.arange(30, 601, 15)  # 15 * np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
        delta_t = 1
        for i, size in enumerate(sizes):
            ini_intra_metabolites = agg_simulator.results[cur_hour].copy()
            agg = CellModel(
                size,
                ini_intra_metabolites,
                int(arg.single_cell_radius),
                parameters,
                np.array(diffusion_coef) * arg.porosity / arg.tortuosity,  # 1000 * 60 * 24, # um/h
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
        return mr_biomass_mean, rsd


def run(arg):
    import pickle
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    np.random.seed(10)

    porosity = 0.27  # 60 rpm, and 0.229 100 rpm
    cell_growth_rate = 0.043  # 0.04531871789155581
    data_whole, mean_whole, std_whole = get_data('dynamic_model/iPSC_data.xlsx')

    file = open("multi_scale_model/data/result_all.pickle", 'rb')
    parameters = pickle.load(file)
    parameters = parameters.params
    colormap = mpl.cm.tab10.colors
    if arg.validation_metabolite_consumption:
        hours = 144
        agg_density_function = get_aggregation_profile(no_aggregate=False, bioreactor=True)
        health = []
        test_index = 1  # HGLL case
        x0_LGHL = np.hstack([S2_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
        x0_LGHL[-1] = 0.2
        agg_simulator = Aggregate(cell_growth_rate, agg_density_function, meta_index, x0_LGHL,
                                  arg.single_cell_radius, parameters, diffusion_coef, porosity, delta_t=1)
        for i in range(hours):
            # medium exchange after 48, 96, 120, 144 hours
            if i in [48, 96, 120, 144]:
                x0_LGHL = np.hstack([S2_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
                x0_LGHL[-1] = agg_simulator.cell_density / 1000
                _result = agg_simulator.results.copy()
                _health = agg_simulator.health.copy()
                _t = agg_simulator.t
                agg_simulator = Aggregate(cell_growth_rate, agg_density_function, meta_index, x0_LGHL,
                                          arg.single_cell_radius, parameters, diffusion_coef, porosity, delta_t=1)
                agg_simulator.results = _result
                agg_simulator.health = _health
                agg_simulator.t = _t
            health.append(agg_simulator.simulate(arg.validation_metabolite_consumption))
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
    if arg.variance_component_analysis:
        hours = 97
        agg_density_function = get_aggregation_profile(no_aggregate=False, bioreactor=True)
        agg_simulator_HGLL, health = run_simulation(hours, 1, mean_whole, cell_growth_rate, agg_density_function,
                                                    meta_index,
                                                    arg.single_cell_radius, parameters, diffusion_coef, porosity, 1,
                                                    cal_variance=True)
        variance_result = np.array(agg_simulator_HGLL.variance_metabolite).reshape((hours, 100, len(meta) - 1))
        mean_result = np.array(agg_simulator_HGLL.mean_metabolites).reshape((hours, 100, len(meta) - 1))

        with plt.style.context('default'):
            fig, ax = plt.subplots()
            ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='both', useMathText=True)
            total_variance_biomass = []
            total_variance_biomass_std = []
            for i in range(hours):
                total_variance_biomass.append(
                    np.nansum(np.sqrt(variance_result[i, :, -1]) / (np.abs(mean_result[i, :, -1]))))
                total_variance_biomass_std.append(
                    np.nanstd(np.sqrt(variance_result[i, :, -1]) / (np.abs(mean_result[i, :, -1]))))
            # sum of variance of metabolite conc change
            total_variance_all = []
            total_variance_all_std = []
            for i in range(hours):
                # total_variance_all.append(np.nansum(np.sqrt(variance_result[i, :, :])) / np.nansum(np.abs(mean_result[i, :, :])))
                total_variance_all.append(
                    np.nansum(np.sqrt(variance_result[i, :, :]) / (np.abs(mean_result[i, :, :]) + 1e-6)) / (
                            len(meta) - 1))
                total_variance_all_std.append(
                    np.nanstd(np.sqrt(variance_result[i, :, :]) / (np.abs(mean_result[i, :, :]))) / (len(meta) - 1))
            ax.plot(np.array(range(hours))[:72] + 1, total_variance_biomass[:72], color=colormap[0], alpha=0.9,
                    linewidth=3)
            ax.set_ylabel('Relative Standard Deviation of \n Biomass Production in $\Delta t$ Time', fontsize=14,
                          color=colormap[0], multialignment='center')
            # ax.set_ylim(0, 5e4)
            ax2 = ax.twinx()
            ax2.ticklabel_format(style='sci', scilimits=(-3, 4), axis='both', useMathText=True)
            ax.set_xlabel('Time (h)', fontsize=14)
            ax2.plot(np.array(range(hours))[:72] + 1, total_variance_all[:72], color=colormap[3], alpha=0.9,
                     linewidth=3, )
            ax2.set_ylabel('Average Relative Standard Deviation of \n Metabolite Change in $\Delta t$ Time',
                           fontsize=14,
                           color=colormap[3], multialignment='center')
            # ax2.set_ylim(0, 5e4)
            # ax2.fill_between(np.array(range(hours)) + 1, np.array(total_variance_all) - np.array(total_variance_all_std),
            #                  np.array(total_variance_all) + np.array(total_variance_all_std),
            #                  color='g', alpha=.1)
            # plt.yticks(fontsize=14)
            # plt.xticks(fontsize=14)
            # plt.xticks(list(range(0, hours + 1, 12)))
            plt.tight_layout()
            plt.savefig("multi_scale_model/result/variance-component-analysis/{}.svg".format('variance-with-time'),
                        bbox_inches='tight', format='svg')
            plt.savefig("multi_scale_model/result/variance-component-analysis/{}.pdf".format('variance-with-time'),
                        bbox_inches='tight', format='pdf')
            plt.show(bbox_inches='tight')
            plt.clf()
    if arg.steady_state_concentration_profiles:
        agg_density_function = get_aggregation_profile(no_aggregate=False, bioreactor=True)
        agg_simulator, _ = run_simulation(48, 1, mean_whole, cell_growth_rate, agg_density_function, meta_index,
                                          arg.single_cell_radius, parameters, diffusion_coef, porosity, 1,
                                          cal_variance=False)
        reaction_diffusion(agg_simulator, parameters, arg)

    if arg.health_condition:
        agg_density_function = get_aggregation_profile(no_aggregate=False, bioreactor=True)
        agg_simulator, _ = run_simulation(48, 1, mean_whole, cell_growth_rate, agg_density_function, meta_index,
                                          arg.single_cell_radius, parameters, diffusion_coef, porosity, 1,
                                          cal_variance=False)
        get_health_condition(agg_simulator, parameters, arg)

    if arg.flux_rates_and_metabolite_conc:
        agg_density_function = get_aggregation_profile(no_aggregate=False, bioreactor=True)
        agg_simulator, _ = run_simulation(48, 1, mean_whole, cell_growth_rate, agg_density_function, meta_index,
                                          arg.single_cell_radius, parameters, diffusion_coef, porosity, 1,
                                          cal_variance=False)
        get_standardized_flux_rate(agg_simulator, parameters, arg)
    if arg.get_biomass_yield_variance_analysis:
        cur_hour_list = [24, 48, 72]
        agg_density_function = get_aggregation_profile(no_aggregate=False, bioreactor=True)
        agg_simulator, _ = run_simulation(cur_hour_list[-1] + 1, 1, mean_whole, cell_growth_rate, agg_density_function,
                                          meta_index, arg.single_cell_radius, parameters, diffusion_coef, porosity, 1,
                                          cal_variance=False)
        for cur_hour in cur_hour_list:
            sizes = np.arange(30, 601, 15)
            mr_biomass_mean, rsd = get_biomass_yield_variance_analysis(agg_simulator, parameters, arg, cur_hour)
            fig, ax = plt.subplots()
            ax.plot(sizes, np.mean(mr_biomass_mean, axis=0), '-', color=colormap[0], alpha=1, linewidth=3,
                    label=r'E[$\Delta$ biomass]')
            ax.fill_between(sizes, np.mean(mr_biomass_mean, axis=0) - np.std(mr_biomass_mean, axis=0),
                            np.mean(mr_biomass_mean, axis=0) + np.std(mr_biomass_mean, axis=0),
                            color=colormap[0], alpha=.1)
            ax.grid(axis='x', color='0.95')
            ax2 = ax.twinx()
            ax2.ticklabel_format(style='sci', scilimits=(-3, 4), axis='both', useMathText=True)
            ax2.plot(sizes, np.mean(rsd, axis=0), '-', color=colormap[3], alpha=1, linewidth=3,
                     label=r'Var[$\Delta$ biomass]')
            ax2.fill_between(sizes, np.mean(rsd, axis=0) - np.std(rsd, axis=0),
                             np.mean(rsd, axis=0) + np.std(rsd, axis=0),
                             color=colormap[3], alpha=.2)
            ax.set_xlabel('Aggregate Radius ' + r'($\mu$m)', fontsize=14)
            ax.set_ylabel('Mean Biomass Production \n (nmol/$10^6$ cells/h)', color=colormap[0], fontsize=14)
            ax2.set_ylabel('Relative Standard Deviation (%) of \n Biomass Production in One Hour', color=colormap[3],
                           fontsize=14)
            plt.tight_layout()
            plt.savefig("multi_scale_model/result/variance-component-analysis/{}-{}.pdf".format('mean-variance', cur_hour),
                        bbox_inches='tight', format='pdf')
            plt.savefig("multi_scale_model/result/variance-component-analysis/{}-{}.svg".format('mean-variance', cur_hour),
                        bbox_inches='tight', format='svg')
            plt.show()

    if arg.comparison_under_monolayer_condition:
        hours = 48
        agg_density_function = get_aggregation_profile(no_aggregate=False, bioreactor=True)
        agg_simulator, health = run_simulation(hours, 1, mean_whole, cell_growth_rate, agg_density_function,
                                               meta_index,
                                               arg.single_cell_radius, parameters, diffusion_coef, arg.porosity, 1,
                                               cal_variance=False)
        plot_comparison_under_monolayer_condition(1, hours, agg_simulator, mean_whole, std_whole, parameters,
                                                  draw_measurement=True,
                                                  path='multi_scale_model/result/simulation-with-agg')
    if arg.inner_outer_cell_metabolism:
        ''' Generate the inner and outer cell metabolism. The data will generate and save to 'multi_scale_model/result/flux_rate/'
        '''
        hours = 73
        agg_density_function = get_aggregation_profile(no_aggregate=False, bioreactor=True)
        test_index = 1
        x0_LGHL = np.hstack([S1_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
        agg_simulator = Aggregate(cell_growth_rate, agg_density_function, meta_index, x0_LGHL,
                                  arg.single_cell_radius, parameters, diffusion_coef, porosity, delta_t=1, deterministic=True)
        for i in range(hours):
            agg_simulator.simulate()
            print('time', (i + 1) * 1, 'GLC', agg_simulator.extra_metabolites[meta_label_index['GLC'][1]],
                  'Lac', agg_simulator.extra_metabolites[meta_label_index['ELAC'][1]])
            if i in [24, 48, 72]:
                for size in [4, 8, 16, 24]:
                    rates, conc = agg_simulator.aggregates[size].get_flux_rate_cell_level(agg_simulator.extra_metabolites)
                    outer_cell, inner_cell = rates[-1] * 1e6, rates[0] * 1e6
                    np.savetxt("multi_scale_model/result/flux_rate/inner_cell-hours{}-size{}-cd.csv".format(i, size * 15),
                               inner_cell, delimiter=", ")
                    np.savetxt("multi_scale_model/result/flux_rate/outer_cell-hours{}-size{}-cd.csv".format(i, size * 15),
                               outer_cell, delimiter=", ")
        try:
            data_extra_raw = pd.read_excel('multi_scale_model/result/flux_rate/result.xlsx', sheet_name="no-sum")
            data_extra_raw.columns = ['Process Time'] + ['24 Hours'] * 8 + ['48 Hours'] * 8

            with plt.style.context('tableau-colorblind10'):
                fig = plt.figure(figsize=(10, 8))
                ax = plt.subplot(1, 1, 1)
                dot_plot(data_extra_raw.iloc[2:-1, 8].values, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                         color=colormap[4], label=r'Inner Cell (360 $\mu$m)', markersize=10)
                dot_plot(data_extra_raw.iloc[2:-1, 6].values, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                         color=colormap[3], label=r'Inner Cell (240 $\mu$m)', marker='^', markersize=6)
                dot_plot(data_extra_raw.iloc[2:-1, 4].values, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                         color=colormap[2], label=r'Inner Cell (120 $\mu$m)', marker='*', markersize=8)
                dot_plot(data_extra_raw.iloc[2:-1, 2].values, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                         color=colormap[1], label=r'Inner Cell (60 $\mu$m)', marker='s', markersize=5)
                dot_plot(data_extra_raw.iloc[2:-1, 1].values, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                         color=colormap[0], marker='v', markersize=6, label='Outer Cell')

                ax.axvline(x=2200, color='black', ls='--')
                outer_cell = data_extra_raw.iloc[2:-1, 1].values
                dot_plot(data_extra_raw.iloc[2:-1, 9].values + 2500, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                         marker='v', markersize=8, color=colormap[0])
                dot_plot(data_extra_raw.iloc[2:-1, 10].values + 2500, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                         color=colormap[1])
                dot_plot(data_extra_raw.iloc[2:-1, 12].values + 2500, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                         color=colormap[2])
                dot_plot(data_extra_raw.iloc[2:-1, 14].values + 2500, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                         color=colormap[3])
                dot_plot(data_extra_raw.iloc[2:-1, 16].values + 2500, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                         color=colormap[4])
                ax.set_xticks(labels=[0, 1000, 2000, 0, 1000, 2000, 3000], ticks=[0, 1000, 2000, 2500, 3500, 4500, 5500],
                              fontsize=12)
                plt.title("24 Hours                                    48 Hours", fontsize=16)
                plt.xlabel(r"Flux Rates (nmol/$10^6$ cells/h)", fontsize=16)
                plt.legend(bbox_to_anchor=(1., 1.15), ncol=5, columnspacing=0.2, labelspacing=0.2, handletextpad=0.2)
                # plt.savefig("multi_scale_model/result/flux_rate/flux_comparison.svg",
                #             bbox_inches='tight')
                # plt.savefig("multi_scale_model/result/flux_rate/flux_comparison.pdf",
                #             bbox_inches='tight')
                plt.show()

            data_extra_raw = pd.read_excel('multi_scale_model/result/flux_rate/result.xlsx', sheet_name="Sheet1")
            data_extra_raw.columns = ['Process Time'] + ['24 Hours'] * 8 + ['48 Hours'] * 8
            with plt.style.context('seaborn-v0_8-colorblind'):
                fig = plt.figure(figsize=(10, 8))
                ax = plt.subplot(1, 1, 1)
                outer_cell = data_extra_raw.iloc[2:-1, 1].values
                # dot_plot([1] * len(data_extra_raw.iloc[2:-1, 1].values), datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax, color=colormap[0], marker='*', markersize=14)
                dot_plot(data_extra_raw.iloc[2:-1, 8].values / outer_cell, datalabels=data_extra_raw.iloc[2:-1, 0].values,
                         ax=ax,
                         color=colormap[4], label=r'Inner Cell (360 $\mu$m)', markersize=10)
                dot_plot(data_extra_raw.iloc[2:-1, 6].values / outer_cell, datalabels=data_extra_raw.iloc[2:-1, 0].values,
                         ax=ax,
                         color=colormap[3], label=r'Inner Cell (240 $\mu$m)', marker='^', markersize=6)
                dot_plot(data_extra_raw.iloc[2:-1, 4].values / outer_cell, datalabels=data_extra_raw.iloc[2:-1, 0].values,
                         ax=ax,
                         color=colormap[2], label=r'Inner Cell (120 $\mu$m)', marker='*', markersize=8)
                dot_plot(data_extra_raw.iloc[2:-1, 2].values / outer_cell, datalabels=data_extra_raw.iloc[2:-1, 0].values,
                         ax=ax,
                         color=colormap[1], label=r'Inner Cell (60 $\mu$m)', marker='s', markersize=5)
                ax.axvline(x=1.6, color='black', ls='-')
                ax.axvline(x=1, color=colormap[0], ls='--', label='Outer Cell')
                ax.axvline(x=3, color=colormap[0], ls='--')
                outer_cell = data_extra_raw.iloc[2:-1, 9].values
                dot_plot(data_extra_raw.iloc[2:-1, 10].values / outer_cell + 2,
                         datalabels=data_extra_raw.iloc[2:-1, 0].values,
                         ax=ax, color=colormap[1], marker='s', markersize=5)
                dot_plot(data_extra_raw.iloc[2:-1, 12].values / outer_cell + 2,
                         datalabels=data_extra_raw.iloc[2:-1, 0].values,
                         ax=ax, color=colormap[2], marker='*', markersize=8)
                dot_plot(data_extra_raw.iloc[2:-1, 14].values / outer_cell + 2,
                         datalabels=data_extra_raw.iloc[2:-1, 0].values,
                         ax=ax, color=colormap[3], marker='^', markersize=6)
                dot_plot(data_extra_raw.iloc[2:-1, 16].values / outer_cell + 2,
                         datalabels=data_extra_raw.iloc[2:-1, 0].values,
                         ax=ax, color=colormap[4])
                ax.set_xticks(labels=[0, 50, 100, 150, 0, 50, 100], ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], fontsize=12)
                plt.title("24 Hours                                    48 Hours", fontsize=16)
                plt.xlabel("Relative Flux Rates (%)", fontsize=16)
                plt.legend(bbox_to_anchor=(1., 1.12), ncol=5, columnspacing=0.2, labelspacing=0.5, handletextpad=0.)
                plt.savefig("multi_scale_model/result/flux_rate/relative_flux.svg",
                            bbox_inches='tight')
                plt.savefig("multi_scale_model/result/flux_rate/relative_flux.pdf",
                            bbox_inches='tight')
                plt.show()
        except:
            print("The flux rates data are located in 'multi_scale_model/result/flux_rate/'. "
                  "It is necessary to merge them first.")

    if arg.PBM:
        import dill as pickle
        v0 = 1500.0
        if arg.pretrained_PBM:
            try:
                with open('multi_scale_model/data/case_final.pkl', 'rb') as f:
                    agg_density_function = pickle.load(f)
            except:
                with open('data/case_final.pkl', 'rb') as f:
                    agg_density_function = pickle.load(f)
        else:
            f = lambda x: lognorm(x, 2.15 + np.log(7.5), 0.27)


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
            agg_density_function = results[1500]
        ''' true data '''
        D1 = pd.read_csv('multi_scale_model/data/wu2014D1.tsv', sep='\t')
        D2 = pd.read_csv('multi_scale_model/data/wu2014D2.tsv', sep='\t')
        D2.loc[D2['Y'] < 0, 'Y'] = 0
        D3 = pd.read_csv('multi_scale_model/data/wu2014D3.tsv', sep='\t')
        D4 = pd.read_csv('multi_scale_model/data/wu2014D4.tsv', sep='\t')
        experiment_data = [D1, D2, D3, D4]

        ''' predicted data'''
        # agg_density_function = results[grid]
        plt.rcParams.update({'font.size': 16})
        for i, h in enumerate([240, 480, 720]):
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

        mean_results = []
        for i in range(1000):
            data = agg_density_function.N.y[:, i]
            distr = data / np.sum(data)
            mean_val = np.sum(np.arange(0, 1500, 1) * distr)  # mean
            mean_results.append(mean_val)
        plt.plot(range(100), mean_results[::10], lw=3, label='Aggregate Size Model')
        experiment_aggregate_mean = [
            np.sum(
                np.exp(experiment_data[i]['X']) * 7.5 * experiment_data[i]['Y'] / np.sum(experiment_data[i]['Y']))
            for i
            in
            range(4)]
        # plt.plot([0, 24, 48, 72, 96], np.array(experiment_aggregate_mean + [205]) * 1.2, lw=2, markerfacecolor='none')
        plt.plot([0, 24, 48, 72, 96], [70, 100, 143, 200, 255], linestyle='-', marker='o', markersize=10, lw=3,
                 label='Experiments')
        # plt.scatter([0, 24, 48, 72, 96], [70, 100, 143, 200, 255], '-p', s=10,  label=None)
        plt.xlabel('Time (hr)')
        plt.ylabel(r'Aggregate radius ($\mu$m)')
        plt.legend()
        plt.savefig("multi_scale_model/result/PBM/mean_curve.pdf", bbox_inches='tight')
        plt.savefig("multi_scale_model/result/PBM/mean_curve.svg", bbox_inches='tight')
        plt.savefig("multi_scale_model/result/PBM/mean_curve.png", bbox_inches='tight')
        plt.savefig("multi_scale_model/result/PBM/mean_curve.eps", bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-val", "--validation-metabolite-consumption",
                        dest="validation_metabolite_consumption",
                        help="validation metabolite consumption.",
                        default=False, type=bool)

    parser.add_argument("-ss", "--steady-state-concentration-profiles",
                        dest="steady_state_concentration_profiles",
                        help="Generate steady state concentration profiles in 48 hours",
                        default=False, type=bool)

    parser.add_argument("-f", "--flux-rates-metabolite-conc", dest="flux_rates_and_metabolite_conc",
                        help="Reaction flux rates and extracellular metabolite concentrations with various aggregates "
                             "ranging from 30 μm to 600 μm in radius.",
                        default=False, type=bool)

    parser.add_argument("-c", "--comparison-under-monolayer-condition", dest="comparison_under_monolayer_condition",
                        help="Comparison between predicted metabolite concentrations from a single-cell mechanistic "
                             "model and a multi-scale model with experimentally measured bulk metabolite concentrations.",
                        default=False, type=bool)

    parser.add_argument("-v", "--variance-component-analysis", dest="variance_component_analysis",
                        help="Perform variance component analysis.",
                        default=False, type=bool)

    parser.add_argument("-m", "--inner-outer-cell-metabolism", dest="inner_outer_cell_metabolism",
                        help="Generate the figure for metabolic heterogeneity of aggregates of varying sizes.",
                        default=False, type=bool)

    parser.add_argument("-ch", "--cell-health-condition", dest="health_condition",
                        help="Analyze iPS cell health condition.",
                        default=False, type=bool)

    parser.add_argument("-size", "--get-biomass-yield-variance-analysis", dest="get_biomass_yield_variance_analysis",
                        help="Biomass yield and variance analysis.",
                        default=False, type=bool)

    parser.add_argument("-pbm", "--population-balance-model", dest="PBM",
                        help="Run population balance model. PBM takes long time (about 12 hours).",
                        default=False, type=bool)

    parser.add_argument("-pretrained_pbm", "--pretrained-population-balance-model", dest="pretrained_PBM",
                        help="Load the pretrained PBM model.",
                        default=True, type=bool)

    parser.add_argument("-r", "--single-cell-radius",
                        dest="single_cell_radius",
                        help="Set single cell raidus.",
                        default=7.5, type=float)

    parser.add_argument("-t", "--tortuosity",
                        dest="tortuosity",
                        help="Set tortuosity.",
                        default=1.5, type=float)

    parser.add_argument("-dt", "--delta-t", dest="delta_t",
                        help="Set simulation time interval (h)",
                        default=0.5, type=float)

    parser.add_argument("-p", "--porosity", dest="porosity",
                        help="Set porosity",
                        default=0.229)


    options = parser.parse_args()

    print('OPTIONS ', options)

    run(options)
