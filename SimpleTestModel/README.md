
## example with synthetic data

#### ``synth_fns.py`` has generic functions for this example

### jupyter notebooks

note : indicative plots are inline in notebooks but the figs for the article are made in separate ``.py`` files, see below.  In some indicative plots, the time is rescaled by a factor of 4 with respect to the final results.

#### ``synthInfTest-pop1e8.ipynb``

Inference for an example model with large population (approximated likelihood is accurate)

* estimate of MAP parameters based on a single trajectory, population 10^8
* MCMC for posterior
* output ``dataSynthInfTest-pop1e8-mcmc.pik`` from MCMC and ``dataSynthInfTest-pop1e8-stochTraj.npy`` with synthetic data
(there is an option to load the synthetic data, if it has already been generated)

run-time for MCMC is a few hours 

#### ``synthInfTest-pop1e4.ipynb``

similar to corresponding notebook with filename ending ``pop1e8``, this one has population 10^4.  (In this case the approximated lilkelihood is less accurate but results are still very reasonable).

additional output ``figScatter_pop1e4.pdf`` shows scatter plots of selected posterior params

#### ``synthInfTest-pop1e6-win5.ipynb`` and similar filenenames ending  ``win1.ipynb``-``win4.ipynb``

for filename ending winX, perform inference over a period of 4X days (including MCMC).   The computations indicate that the inference methodology can be used to generate accurate forecasts.

* population is 10^6,
computation is similar to synthInfTest for populations 10^4 and 10^8 but with additional stochastic forecast.

* Additional output ``dataSynthInfTest-pop1e6-winX-foreTraj.npy`` that includes trajectory forecasts

* for filename ending win5, also collate and show the forecasts

#### ``synthInfQuality-pop1e6.ipynb`` and similar filenames ending ``pop1e5.ipynb`` and ``pop1e4.ipynb``

generate multiple stochastic trajectories and run inference (including MCMC) for all of them.  (This will take a few hours.)  The are tests for bias on the inferred parameter esimates (they show that the credible intervals capture the true parameters in all cases)

output : ``dataSynthInfQuality-pop1eX-runY-mcmcAll.pik``  and ``dataSynthInfQuality-pop1eX-runY-stochTrajZZ.npy`` for the individual stochastic trajectories

### ``.py`` files for producing figures

* ``figs_pop1e8.py`` produces ``figPostHistos_pop1e8.pdf`` and ``figInfTraj_pop1e8.pdf`` using data from ``synthInfTest-pop1e8.ipynb``
* ``figs_pop1e4.py`` similar to ``figs_pop1e8.py`` with population 1e4
* ``figs_forecast.py`` produces ``figForecast_pop1e6.pdf`` using data from ``synthInfTest-pop1e6-win5.ipynb`` etc
* ``figs_quality.py`` produces ``figQuality.pdf`` using data from ``synthInfQuality-pop1e6.ipynb`` etc

