RNA-LifeTIme: A deep learning framework for RNA lifetime prediction.
===========
[![zheng](https://img.shields.io/badge/Author-Zheng.H-yellow)](https://zhenghuazx.github.io/hua.zheng/)

**Cite us**: Hua Zheng, Wei Xie, Paul Whitford, Ailun Wang, Chunsheng Fang, Wandi Xu. (2023). _Structure-Function Dynamics Hybrid Modeling: RNA Degradation_. arXiv preprint arXiv:2305.03925.



RNA-LifeTIme is a package for RNA lifetime prediction task written in pytorch. **The MD simulation data will be available in June.**

Predicting RNA degradation dynamics based on its 3D conformation remains a challenge due to the underdeveloped 
methodologies for modeling tertiary structure input. Current state-of-the-art RNA degradation predictive models 
primarily concentrate on primary and secondary structure information. Our paper provides a novel architecture for 
modeling the RNA unfolding process.
![](assets/degradtion_processing.png)

The lifetime of a native contact quantifies the importance of an individual inter-residue contact in the 
protein/RNA folding mechanism. We adopt this concept and apply it to RNA degradation process
(Figure above) by studying the trajectory of fraction of native contacts, i.e., the ratio of the number of 
contacts present in the current structure to the number of contacts in the native structure.


Installation
======================================
If you would like to build the library from the source code, then at the root `/RNA-LifeTime`, run
```shell
pip install build
python -m build
```
Then install RNA-LifeTime and clean up the directory
```shell
pip install dist/RNA_LifeTime-1.0.0-py3-none-any.whl
rm -r dist
```
Usage
======================================
### Usage Help
the main.py script provides useful help messages that describe the available command-line arguments and their usage.
```shell
python main.py --help
```
Read the help messages to check out hyperparamters:
```shell
usage: main.py [-h] [-val VALIDATION_METABOLITE_CONSUMPTION] [-ss STEADY_STATE_CONCENTRATION_PROFILES] [-f FLUX_RATES_AND_METABOLITE_CONC] [-c COMPARISON_UNDER_MONOLAYER_CONDITION] [-v VARIANCE_COMPONENT_ANALYSIS] [-m INNER_OUTER_CELL_METABOLISM] [-ch HEALTH_CONDITION]
               [-size GET_BIOMASS_YIELD_VARIANCE_ANALYSIS] [-pbm PBM] [-r SINGLE_CELL_RADIUS] [-t TORTUOSITY] [-dt DELTA_T] [-p POROSITY]

optional arguments:
  -h, --help            show this help message and exit
  -val VALIDATION_METABOLITE_CONSUMPTION, --validation-metabolite-consumption VALIDATION_METABOLITE_CONSUMPTION
                        validation metabolite consumption.
  -ss STEADY_STATE_CONCENTRATION_PROFILES, --steady-state-concentration-profiles STEADY_STATE_CONCENTRATION_PROFILES
                        Biomass yield and variance analysis.
  -pbm PBM, --population-balance-model PBM
                        Run population balance model. PBM takes long time (about 12 hours).
  -r SINGLE_CELL_RADIUS, --single-cell-radius SINGLE_CELL_RADIUS
                        Set single cell raidus.
  -t TORTUOSITY, --tortuosity TORTUOSITY
                        Set tortuosity.
  -dt DELTA_T, --delta-t DELTA_T
                        Set simulation time interval (h)
  -p POROSITY, --porosity POROSITY
                        Set porosity
```

Example
===========

```python
python main.py -e 15 -b 512 -lr 0.003 -c ./MD-simulation/models/ -m RNA-LifeTime -p ./MD-simulation/ -f False -g 3 -l 72 -t False -d 0.2 -r 1 --step 3 --gamma 0.3
```
![img.png](assets/training.png)


## Bio-SoS Simulation example in Python Console
```python
import pickle
from multi_scale_model.aggregate import Aggregate, get_aggregation_profile
from multi_scale_model.Util import *
# HGLL
# simulation parameters
test_index = 1
porosity = 0.27  # 0.27 at 60 rpm, and 0.229 at 100 rpm of agitation rate
cell_growth_rate = 0.043
single_cell_radius = 7.5
hours = 48
bioreactor = True
# Load PBM model for aggregation size distribution and configure the culture setting (monolayer versus bioreactor)
file = open("multi_scale_model/data/result_all.pickle", 'rb')
parameters = pickle.load(file)
parameters = parameters.params
if bioreactor:
    agg_density_function = get_aggregation_profile(no_aggregate=False, bioreactor=True) # bioreactor
else:
    agg_density_function = get_aggregation_profile(no_aggregate=True, bioreactor=False) # monolayer

# Load initial metabolite concentrations of iPSC cell culture
data_whole, mean_whole, std_whole = get_data('dynamic_model/iPSC_data.xlsx')


x0_LGHL = np.hstack([S2_0 * 1000 * mean_whole[test_index][0, -1], mean_whole[test_index][0]])
agg_simulator = Aggregate(cell_growth_rate, agg_density_function, meta_index, x0_LGHL,
                          single_cell_radius, parameters, diffusion_coef, porosity, delta_t=1)
for i in range(hours):
    agg_simulator.simulate()
    print('time', (i + 1) * 1, 'GLC', agg_simulator.extra_metabolites[meta_label_index['GLC'][1]],
          'Lac', agg_simulator.extra_metabolites[meta_label_index['ELAC'][1]])
```

The solution object is the same as class attribute `results`. We can plot the glucose concentration trajectory using matplotlib:
```python
import numpy as np
import matplotlib.pyplot as plt
from multi_scale_model.Util import meta_label_index
plt.plot(np.array(agg_simulator.results)[:hours, meta_label_index['GLC'][0]]) # plot simulated trajectory of glucose
plt.show()
```
![img.png](asset/glc.png)
Lactate concentration trajectory:
```python
import numpy as np
import matplotlib.pyplot as plt
from multi_scale_model.Util import meta_label_index
plt.plot(np.array(agg_simulator.results)[:hours, meta_label_index['ELAC'][0]]) # plot simulated trajectory of lactate
plt.show()
```
![img.png](asset/lac.png)

## Bio-SoS Result Reproduction
1(a). Generate temporal size distribution of aggregation population balance model (PBM) by using pretrained model (`-pretrained_pbm True`)
```shell
python main.py -pbm True -pretrained_pbm True
```
1(b). Generate temporal size distribution of aggregation population balance model (PBM) without 
using pretrained model (`-pretrained_pbm False`). The parameter estimation can take more than 12 hours.
```shell
python main.py -pbm True -pretrained_pbm True
```
2. Perform Biomass yield and variance analysis.
```shell
python main.py -size True
```
3. Analyze iPS cell health condition.
```shell
python main.py -ch True
```
4. Generate the figure for metabolic heterogeneity of aggregates of varying sizes.
```shell
python main.py -m True
```
5. Perform variance component analysis.
```shell
python main.py -v True
```
6. Comparison between predicted metabolite concentrations from a single-cell mechanistic model and a multi-scale model with experimentally measured bulk metabolite concentrations.
```shell
python main.py -c True
```
7. Metabolic heterogeneity. Reaction flux rates and extracellular metabolite concentrations with various aggregates ranging from 30 μm to 600 μm in radius.
```shell
python main.py -f True
```
8. Generate steady state concentration profiles at 48 hours (Fig. 2c).
```shell
python main.py -ss True
```
9. validation metabolite consumption in a bioreactor culture of iPSCs (Fig. 4).
```shell
python main.py -val True
```

# Contact
If you have any questions, or just want to chat about using the package,
please feel free to contact me in the [Website](https://zhenghuazx.github.io/hua.zheng/).
For bug reports, feature requests, etc., please submit an issue to the email <zheng.hua1@northeastern.edu>.
