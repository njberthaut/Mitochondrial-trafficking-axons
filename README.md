# Mitochondrial-trafficking-axons
Stochastic model of mitochondrial trafficking in axons

## Description:

A stochastic single-particle model of axonal mitochondrial trafficking was developed and fitted to in vivo two-photon imaging data of mitochondria in distal cortical axons. Simulations can reveal the behaviour of individual mitochondria with heterogeneous dynamics governed by probabilistic states. The model can be used to explore questions and inform hypotheses relating to pausing, direction of transport
and effects of morphology. Parameters relating to all these notions can be easily altered to test their effects on mitochondrial transport dynamics.

![Mito_trafficking_concept](https://github.com/user-attachments/assets/83228ff9-4166-451b-95fd-eaa9030cfb0b)

from:
[Mitochondrial trafficking dynamics in axons: insights from in vivo two-photon imaging and computational modelling
(Berthaut, N. J. J.) 10 Dec 2024](https://research-information.bris.ac.uk/en/studentTheses/mitochondrial-trafficking-dynamics-in-axons)

## Table of contents: 


Scirpt for simulation on a model axonal tree:
- Mitochondrial-trafficking_simulation_model_tree.ipynb

Scirpt for optimisation of parameters (state transition probabilities):
- Parameter_optimisation_transition_probabilities.ipynb

Modules: 
- modmorf.py : Functions to compute axon morphology features needed for simulation.
- modsim.py : Functions for simulation.
- modout.py : Functions for analysing outputs of suimulation (not required to run simulation).
- modplot.py : Functions to set plotting parameters (not required to run simualtion).

Morphologies folder 
- tree_10_mm_path.swc : Model axon used for simulation. _Characteristics:
    total length:10 mm, max. soma-endpoint distance: 10mm, inter-node distance: 0.2 um, number of nodes: 116656, unmber of branchpoints: 3, max.branch order: 3, number of endpoints: 4._
- nobranchtree.swc : Straight cable used for optimisation.

## How to run:
Simulations were run in Python 3.8.8. The following package is needed: 
- MorphoPy (0.7.1) [github.com/berenslab/MorphoPy](https://github.com/berenslab/MorphoPy) to import and
extract morphology parameters from reconstructed axons (Laturnus et al. 2020)




