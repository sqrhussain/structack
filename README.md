# structack
Structure-aware attacks on GNNs [Paper](https://dl.acm.org/doi/abs/10.1145/3465336.3475110)

This repository implements global black-box adversarial attacks on the node classification task with graph neural networks.
The attacks have only access to the adjacency matrix and not the nodes feature vectors.
These attacks are currenlty based on the knowledge of the degree distribution and community structure.

### Run
The main testin code is in the file `evaluate_attacks.py`.
Please run `python -u -m evaluate_attacks` from the project root directory.

* The `--model` argument chooses the GNN model to train after the perturbation (`gcn, sgc, gat`).

* With the `--dataset` argument, you can list the datasets to perturb (`citeseer, cora, cora_ml, polblogs, pubmed`).

* This file can run in 3 modes
	1. Evaluation on clean graphs with the argument `--approach_type clean`
	2. Evaluation on perturbed graphs using baseline attacks (i.e., random, DICE, Metattack and PGD) with the argument `--approach_type baseline`
	3. Evaluation on perturbed graphs using structack combinations with the argument `--approach_type structack`

### Dependencies
`python >= 3.7`

Please install [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), then run `pip install -r requirements.txt`.

### Citation

```
@inproceedings{10.1145/3465336.3475110,
author = {Hussain, Hussain and Duricic, Tomislav and Lex, Elisabeth and Helic, Denis and Strohmaier, Markus and Kern, Roman},
title = {Structack: Structure-Based Adversarial Attacks on Graph Neural Networks},
year = {2021},
isbn = {9781450385510},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3465336.3475110},
doi = {10.1145/3465336.3475110},
booktitle = {Proceedings of the 32nd ACM Conference on Hypertext and Social Media},
pages = {111–120},
numpages = {10},
keywords = {network centrality, network similarity, adversarial attacks, graph neural networks},
location = {Virtual Event, USA},
series = {HT '21}
}

```