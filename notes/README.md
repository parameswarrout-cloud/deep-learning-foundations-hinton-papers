# Deep Learning Foundations: Hinton Papers — Research Notes & Visual Guides

This directory contains line-by-line mathematical notes, pedagogical explanations, high-resolution diagrams, and animations replicating and analyzing landmark papers by Geoffrey Hinton and collaborators.

---

## Directory Organization & Standard

To keep the workspace scalable and maintainable as more papers are studied, each paper has its own self-contained directory named after the paper identifier:

```text
notes/
├── README.md                                    # Master index & roadmap (this file)
│
├── 01_1986_Backpropagation/                     # Paper 01
│   ├── 01_1986_Backpropagation_Line_by_Line_Explanation.md # Full line-by-line guide
│   └── images/                                  # All visual assets & animations for Paper 01
│       ├── backprop_flow_animation.gif
│       ├── bptt_recurrent_equivalence.png
│       ├── family_tree_architecture.png
│       ├── momentum_vs_gd_animation.gif
│       ├── neuron_forward_pass.jpg
│       └── symmetry_network_diagram.png
│
├── 02_1986_Distributed_Representations/         # Paper 02 (Next)
│   ├── 02_1986_Distributed_Representations_Line_by_Line_Explanation.md
│   └── images/
│
└── ...
```

---

## Paper Index & Study Progress

| # | Paper Identifier | Year | Paper Title | Original PDF | Code Notebook | Notes & Visual Guide | Status |
|:---:|:---|:---:|:---|:---:|:---:|:---:|:---:|
| **01** | `01_1986_Backpropagation` | 1986 | *Learning representations by back-propagating errors* | [PDF](../../Papers/01_1986_Backpropagation.pdf) | [Notebook](../../Learning_representations_by_back_propagating_errors.ipynb) | [Notes & Visuals](01_1986_Backpropagation/01_1986_Backpropagation_Line_by_Line_Explanation.md) | ✅ Complete |
| **02** | `02_1986_Distributed_Representations` | 1986 | *Distributed Representations* | [PDF](../../Papers/02_1986_Distributed_Representations.pdf) | [Notebook](../../Distributed_Representations.ipynb) | `02_1986_Distributed_Representations/` | ⏳ Planned |
| **03** | `03_2013_Initialization_and_Momentum` | 2013 | *On the importance of initialization and momentum in deep learning* | [PDF](../../Papers/03_2013_Initialization_and_Momentum.pdf) | [Notebook](../../On_the_importance_of_initialization_and_momentum_in_deep_learning.ipynb) | `03_2013_Initialization_and_Momentum/` | ⏳ Planned |
| **04** | `04_2010_ReLU_Improve_RBMs` | 2010 | *Rectified Linear Units Improve Restricted Boltzmann Machines* | [PDF](../../Papers/04_2010_ReLU_Improve_RBMs.pdf) | [Notebook](../../Rectified_Linear_Units_Improve_Restricted_Boltzmann_Machines.ipynb) | `04_2010_ReLU_Improve_RBMs/` | ⏳ Planned |
| **05** | `05_2014_Dropout` | 2014 | *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* | [PDF](../../Papers/05_2014_Dropout.pdf) | [Notebook](../../Dropout_A_Simple_Way_to_Prevent_Neural_Networks_from_overfitting.ipynb) | `05_2014_Dropout/` | ⏳ Planned |
| **06** | `06_2016_Layer_Normalization` | 2016 | *Layer Normalization* | [PDF](../../Papers/06_2016_Layer_Normalization.pdf) | [Notebook](../../Layer_Normalization.ipynb) | `06_2016_Layer_Normalization/` | ⏳ Planned |
| **07** | `07_1991_Adaptive_Mixtures_of_Local_Experts_MoE` | 1991 | *Adaptive Mixtures of Local Experts* | [PDF](../../Papers/07_1991_Adaptive_Mixtures_of_Local_Experts_MoE.pdf) | [Notebook](../../Adaptive_Mixtures_of_Local_Experts.ipynb) | `07_1991_Adaptive_Mixtures_of_Local_Experts_MoE/` | ⏳ Planned |
| **08** | `08_2012_AlexNet` | 2012 | *ImageNet Classification with Deep Convolutional Neural Networks* | [PDF](../../Papers/08_2012_AlexNet.pdf) | [Notebook](../../ImageNet_Classification_with_Deep_Convolutional_Neural_Networks_AlexNet.ipynb) | `08_2012_AlexNet/` | ⏳ Planned |
| **09** | `09_2008_t_SNE` | 2008 | *Visualizing Data using t-SNE* | [PDF](../../Papers/09_2008_t_SNE.pdf) | [Notebook](../../Visualizing_Data_using_t_SNE.ipynb) | `09_2008_t_SNE/` | ⏳ Planned |
| **10** | `10_2015_Distilling_the_Knowledge_in_a_Neural_Network` | 2015 | *Distilling the Knowledge in a Neural Network* | [PDF](../../Papers/10_2015_Distilling_the_Knowledge_in_a_Neural_Network.pdf) | [Notebook](../../Distilling_the_Knowledge_in_a_Neural_Network.ipynb) | `10_2015_Distilling_the_Knowledge_in_a_Neural_Network/` | ⏳ Planned |
| **11a** | `11a_1985_Boltzmann_Machines` | 1985 | *A Learning Algorithm for Boltzmann Machines* | [PDF](../../Papers/11a_1985_Boltzmann_Machines.pdf) | [Notebook](../../A_Learning_Algorithm_for_Boltzmann_Machines.ipynb) | `11a_1985_Boltzmann_Machines/` | ⏳ Planned |
| **11b** | `11b_2002_Contrastive_Divergence` | 2002 | *Training Products of Experts by Minimizing Contrastive Divergence* | [PDF](../../Papers/11b_2002_Contrastive_Divergence.pdf) | [Notebook](../../Training_Products_of_Experts_by_Minimizing_Contrastive_Divergence.ipynb) | `11b_2002_Contrastive_Divergence/` | ⏳ Planned |
| **12** | `12_2006_Deep_Belief_Nets` | 2006 | *A Fast Learning Algorithm for Deep Belief Nets* | [PDF](../../Papers/12_2006_Deep_Belief_Nets.pdf) | [Notebook](../../A_fast_learning_algorithm_for_deep_belief_nets.ipynb) | `12_2006_Deep_Belief_Nets/` | ⏳ Planned |

---

## Instructions for Adding Future Paper Notes

When adding notes for a new paper:
1. Create a dedicated folder matching the paper naming convention: `notes/<XX_YEAR_PaperName>/`
2. Create an `images/` subfolder: `notes/<XX_YEAR_PaperName>/images/` for all GIFs, PNGs, and illustrations.
3. Write `README.md` inside `notes/<XX_YEAR_PaperName>/` referencing its images using relative paths (`images/filename.ext`).
4. Update the Master Index table above.

## Focused equation guides

- [Weighted inputs and worked examples](01_1986_Backpropagation/01_1986_Backpropagation_Line_by_Line_Explanation.md#example-1-three-inputs-feeding-two-neurons)
- [Weighted-sum animation code](01_1986_Backpropagation/Aimation-code/animate_weighted_sum.py)
