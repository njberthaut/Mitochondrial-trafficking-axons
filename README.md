# Mitochondrial-trafficking-axons
Stochastic model of mitochondrial trafficking in axons

**Description:
**

![Mito_trafficking_concept](https://github.com/user-attachments/assets/83228ff9-4166-451b-95fd-eaa9030cfb0b)

**Table of contents: 
**

Scirpt for simulation on a model axonal tree:
- Mitochondrial-trafficking_simulation_model_tree.ipynb

Modules: 
- modmorf.py : Functions to compute axon morphology features needed for simulation.
- modsim.py : Functions for simulation.
- modout.py : Functions for analysing outputs of suimulation (not required to run simulation).
- modplot.py : Functions to set plotting parameters (not required to run simualtion).

Morphologies folder 
- tree_10_mm_path.swc : Model axon used for simulation. _Characteristics:
    total length:10 mm, max. soma-endpoint distance: 10mm, inter-node distance: 0.2 um, number of nodes: 116656, unmber of branchpoints: 3, max.branch order: 3, number of endpoints: 4._


How to run:
Simulations were run in Python 3.8.8. The following packages are needed: 

