# Improving Government/National Energy Modelling

## Project Description

Government policy relies on credible economic modelling of the required energy transition. But current tools make it difficult to build large-scale bottom-up economic models. This project will develop open-source software for energy-economy modelling at a much larger scale and realism than previously possible, through better optimization and data management. We will also provide tools to ensure the validity, transparency, and overall trustworthiness of a model. University College London (UCL) energy-economy models are used worldwide and frequently cited in UK government reports (https://www.ucl.ac.uk/drupal/site\_energy-models/policy-impacts-of-our-models, https://www.sciencedirect.com/science/article/pii/S0306261916301672).

The free and open source energy-economic models we build will result in cheaper, more detailed, and more trustworthy data about the energy economy. This can be used by national governments to formulate more precise and ambitious strategies to decarbonize and achieve a sustainable future. The models we will develop can also be used to evaluate decarbonization progress and our project can increase the accuracy of carbon accounting.

The project primarily assists decision makers, e.g. governments/legislatures, by providing them with more accurate and trustworthy predictions. We expect this to accelerate net zero commitments from countries.

Energy-economic models were used to justify and evaluate UKs 2019 amendment to the Climate Change Act that, by committing to net zero, leads to a reduction of 160 MtC02e from the atmosphere in 2050 and beyond. This project will open source such models and enable other countries and corporations to adapt them and accelerate their transitions.

Modelling allows decision makers (government or corporate) to make better policy decisions, which affect environmental outcomes. The tools and techniques we develop can be transferred to model policy outcomes in other areas such as water.

## Project objective

Accelerate net-zero transitions by improving the economic evaluation of interventions in the energy sector (government policies and technology changes).

## Key results

1. Improve the transparency and accessibility of UCLs UK TIMES model (https://www.ucl.ac.uk/drupal/site\_energymodels/sites/energy-models/files/uk-times-overview.pdf) by transitioning to open source. In particular the data management, least-cost optimization, and output formatting are currently closed-source. We will design the system with modern software engineering practices. Switching to free and open source software will reduce costs, increase transparency and accelerate modelling efforts through a snowballing effect.<p>Key Metrics:
    - Replication - Are the sample outputs replicated in the FOSS implementation?
    - Efficiency - Are the outputs produced in time comparable to the proprietary system?
    - Usability - Is the system usable by existing UK TIMES clients?
    </p>

2. Provide efficiency improvements that enable the model to operate at an hourly timescale (8760 timeslices per year instead of the current 16). These will come from our knowledge of machine learning, compilers, and optimization. It may involve a custom machine learning algorithm and/or custom compiler.

3. Provide further efficiency improvements that enable the model to operate over multiple countries and regions per country.

4. Enable propagation of uncertainty through the model, in order to accurately assess risk (of not meeting carbon targets, or not meeting energy demand). This is an opportunity to add significant value to the model.

5. Provide tools to ensure the trustworthiness of a model (in terms of explainability, uncertainty presentation).

6. Apply our software stack to other similar energy-economy models (https://www.ucl.ac.uk/energy-models/), to show that it generalizes.


## Project milestones/deliverables

### Q1

Set up open source project(s) on GitHub for various components, define interfaces, and build benchmarks.

### Q2

Develop infrastructure to populate the input/output databases and inspect/validate them.

### Q3

Develop open source component that creates model scenarios that are accepted by currently used solvers.

### Q4

Minimum viable product tying in the above, but still using GAMS/CPLEX solvers (stretch goal: use open solver).
