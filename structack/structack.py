import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from deeprobust.graph import utils
from deeprobust.graph.global_attack import BaseAttack


class Structack(BaseAttack):
	def __init__(self, model=None, nnodes=None, degree_percentile_range=[0,1], device='cpu'):
		super(Structack, self).__init__(model, nnodes, attack_structure=True, attack_features=False, device=device)
		
		self.modified_adj = None
		self.degree_percentile_range = degree_percentile_range

	def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=True, ll_cutoff=0.004):
		pass