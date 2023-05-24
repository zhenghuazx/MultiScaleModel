import numpy as np

from multi_scale_model.aggregate import Aggregate, get_aggregation_profile
from multi_scale_model.single_cell_model import CellModel
from multi_scale_model.Util import meta_label_index

if __name__ == '__main__':
    import pickle
    from multi_scale_model.Util import get_data, S1_0, meta_index, meta, diffusion_coef, meta_label

    # S1_0 = np.array([1.40E-07, 4.00E-07, 1.40E-07, 2.00E-08, 1.50E-06, 2.00E-07, 2.00E-08, 4.00E-06, 3.00E-07,
    #                  5.00E-08, 4.00E-07, 7.00E-04, 8.00E-06, 1E-6, 9.00E-07, 2.00E-08, 9.00E-13, 5.30E-12, 5.00E-05,
    #                  8.00E-06,
    #                  4.00E-08, 2.00E-06, 2.00E-06, 2.8, 1.50E-07, 5.30E-07, 9.20E-08])

    porosity = 0.229  # 60 rpm, and 0.229 100 rpm
    cell_growth_rate = 0.043  # 0.04531871789155581
    try:
        data_whole = np.load('data/data_whole.npy')
    except:
        data_whole = np.load("multi_scale_model/data/data_whole.npy")
    try:
        data_whole, mean_whole, std_whole = get_data('dynamic_model/iPSC_data.xlsx')
    except:
        data_whole, mean_whole, std_whole = get_data('../dynamic_model/iPSC_data.xlsx')
    single_cell_radius = 7.5
    try:
        file = open("data/result_all.pickle", 'rb')
    except:
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
    hours = 48
    agg_density_function = get_aggregation_profile(no_aggregate=False, bioreactor=False)
    test_index = 1
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
        print('time', (i + 1) * 1, 'GLC', agg_simulator.extra_metabolites[meta_label_index['GLC'][1]],
              'Lac', agg_simulator.extra_metabolites[meta_label_index['ELAC'][1]])
