import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter

from layers import fake_persistence_computation
from torch_persistent_homology.persistent_homology_cpu import compute_persistence_homology_batched_mt

import coord_transforms as coord_transforms


class TopologyLayer(torch.nn.Module):
    """Topological Aggregation Layer."""

    def __init__(self, features_in, features_out, num_filtrations=8,
                 num_coord_funs=3, filtration_hidden=24, num_coord_funs1=3,
                 dim1=False, residual_and_bn=False,
                 share_filtration_parameters=False, fake=False,
                 tanh_filtrations=False, swap_bn_order=False, dist_dim1=False):
        """
        num_coord_funs is a dictionary with the numbers of coordinate functions of each type.
        dim1 is a boolean. True if we have to return dim1 persistence.
        """
        super().__init__()

        self.num_coord_funs = {"Triangle_transform": num_coord_funs,
                               "Gaussian_transform": num_coord_funs,
                               "Line_transform": num_coord_funs,
                               "RationalHat_transform": num_coord_funs
                               }

        self.dim1 = dim1

        self.features_in = features_in
        self.features_out = features_out

        self.num_filtrations = num_filtrations

        self.filtration_hidden = filtration_hidden
        self.residual_and_bn = residual_and_bn
        self.share_filtration_parameters = share_filtration_parameters
        self.fake = fake
        self.swap_bn_order = swap_bn_order
        self.dist_dim1 = dist_dim1

        self.total_num_coord_funs = np.array(
            list(self.num_coord_funs.values())).sum()

        self.coord_fun_modules = torch.nn.ModuleList([
            getattr(coord_transforms, key)(output_dim=self.num_coord_funs[key])
            for key in self.num_coord_funs
        ])

        if self.dim1:
            coord_funs1 = {"Triangle_transform": num_coord_funs1,
                           "Gaussian_transform": num_coord_funs1,
                           "Line_transform": num_coord_funs1,
                           "RationalHat_transform": num_coord_funs1
                           }
            self.coord_fun_modules1 = torch.nn.ModuleList([
                getattr(coord_transforms, key)(output_dim=coord_funs1[key])
                for key in coord_funs1
            ])

        final_filtration_activation = nn.Tanh() if tanh_filtrations else nn.Identity()
        if self.share_filtration_parameters:
            self.filtration_modules = torch.nn.Sequential(
                torch.nn.Linear(self.features_in, self.filtration_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filtration_hidden, num_filtrations),
                final_filtration_activation
            )
        else:
            self.filtration_modules = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(self.features_in, self.filtration_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.filtration_hidden, 1),
                    final_filtration_activation
                ) for _ in range(num_filtrations)
            ])

        if self.residual_and_bn:
            in_out_dim = self.num_filtrations * self.total_num_coord_funs
            features_out = features_in
            self.bn = nn.BatchNorm1d(features_out)
            if self.dist_dim1 and self.dim1:
                self.out1 = torch.nn.Linear(self.num_filtrations * self.total_num_coord_funs, features_out)
        else:
            if self.dist_dim1:
                in_out_dim = self.features_in + 2 * self.num_filtrations * self.total_num_coord_funs
            else:
                in_out_dim = self.features_in + self.num_filtrations * self.total_num_coord_funs

        self.out = torch.nn.Linear(in_out_dim, features_out)

    def compute_persistence(self, x, edge_index, batch, return_filtration=False):
        """
        Returns the persistence pairs as a list of tensors with shape [X.shape[0],2].
        The lenght of the list is the number of filtrations.
        """
        if edge_index is None:
            edge_index = batch.edge_index
        if self.share_filtration_parameters:
            filtered_v_ = self.filtration_modules(x)
        else:
            filtered_v_ = torch.cat([filtration_mod.forward(x)
                                     for filtration_mod in self.filtration_modules], 1)
        filtered_e_, _ = torch.max(torch.stack(
            (filtered_v_[edge_index[0]], filtered_v_[edge_index[1]])), axis=0)

        if batch is None:
            vertex_slices = torch.Tensor((x.shape[0],)).long()
            edge_slices = torch.Tensor((edge_index.shape[0],)).long()
            batch_index = torch.zeros_like(edge_index[0, :]).to(edge_index.device)
        #else:
        
        #vertex_slices = torch.Tensor(batch.__slices__['x']).long()
        #edge_slices = torch.Tensor(batch.__slices__['edge_index']).long()
        #batch_index = batch.batch

        if self.fake:
            return fake_persistence_computation(
                filtered_v_, edge_index, vertex_slices, edge_slices, batch_index)

        vertex_slices = vertex_slices.cpu()
        edge_slices = edge_slices.cpu()

        filtered_v_ = filtered_v_.cpu().transpose(1, 0).contiguous()
        filtered_e_ = filtered_e_.cpu().transpose(1, 0).contiguous()
        edge_index = edge_index.cpu().transpose(1, 0).contiguous()
        
        persistence0_new, persistence1_new = compute_persistence_homology_batched_mt(
            filtered_v_, filtered_e_, edge_index,
            vertex_slices, edge_slices)

        persistence0_new = persistence0_new.to(x.device)
        persistence1_new = persistence1_new.to(x.device)

        if return_filtration:
            return persistence0_new, persistence1_new, filtered_v_
        else:
            return persistence0_new, persistence1_new, None

    def compute_coord_fun(self, persistence, batch, dim1=False):
        """
        Input : persistence [N_points,2]
        Output : coord_fun mean-aggregated [self.num_coord_fun]
        """
        if dim1:
            coord_activation = torch.cat(
                [mod.forward(persistence) for mod in self.coord_fun_modules1], 1)
        else:
            coord_activation = torch.cat(
                [mod.forward(persistence) for mod in self.coord_fun_modules], 1)

        return coord_activation

    def compute_coord_activations(self, persistences, batch, dim1=False):
        """
        Return the coordinate functions activations pooled by graph.
        Output dims : list of length number of filtrations with elements : [N_graphs in batch, number fo coordinate functions]
        """

        coord_activations = [self.compute_coord_fun(
            persistence, batch=batch, dim1=dim1) for persistence in persistences]
        return torch.cat(coord_activations, 1)

    def collapse_dim1(self, activations, mask, slices):
        """
        Takes a flattened tensor of activations along with a mask and collapses it (sum) to have a graph-wise features

        Inputs :
        * activations [N_edges,d]
        * mask [N_edge]
        * slices [N_graphs]
        Output:
        * collapsed activations [N_graphs,d]
        """
        collapsed_activations = []
        if len(slices) == 1:
            slices = torch.cat((slices, torch.LongTensor((0,))))
        for el in range(len(slices)-1):
            activations_el_ = activations[slices[el]:slices[el+1]]
            mask_el = mask[slices[el]:slices[el+1]]
            activations_el = activations_el_[mask_el].sum(axis=0)
            collapsed_activations.append(activations_el)

        return torch.stack(collapsed_activations)

    def forward(self, x, edge_index=None, batch=None, return_filtration=False):
        # Remove the duplicate edges.

        #if batch is not None:
        #    batch = self.remove_duplicate_edges(batch)

        if batch is None:
            edge_slices = torch.Tensor((edge_index.shape[0],)).long()
        else:
            edge_slices = batch.__slices__["edge_index"]

        persistences0, persistences1, filtration = self.compute_persistence(x, edge_index, batch, return_filtration)

        coord_activations = self.compute_coord_activations(
            persistences0, batch)
        if self.dim1:
            persistence1_mask = (persistences1 != 0).any(2).any(0)
            # TODO potential save here by only computing the activation on the masked persistences
            coord_activations1 = self.compute_coord_activations(
                persistences1, batch, dim1=True)
            graph_activations1 = self.collapse_dim1(coord_activations1, persistence1_mask, edge_slices)  # returns a vector for each graph
        else:
            graph_activations1 = None

        if self.residual_and_bn:
            out_activations = self.out(coord_activations)

            if self.dim1 and self.dist_dim1:
                out_activations += self.out1(graph_activations1)[batch]
                graph_activations1 = None
            if self.swap_bn_order:
                out_activations = self.bn(out_activations)
                out_activations = x + F.relu(out_activations)
            else:
                out_activations = self.bn(out_activations)
                out_activations = x + out_activations
        else:
            concat_activations = torch.cat((x, coord_activations), 1)
            out_activations = self.out(concat_activations)
            out_activations = F.relu(out_activations)

        return out_activations, graph_activations1, filtration

    def remove_duplicate_edges(self, batch):

        with torch.no_grad():
            batch = batch.clone()        
            device = batch.x.device
            # Computing the equivalent of batch over edges.
            edge_slices = torch.tensor(batch.__slices__["edge_index"],device= device)
            edge_diff_slices = (edge_slices[1:]-edge_slices[:-1])
            n_batch = len(edge_diff_slices)
            batch_e = torch.repeat_interleave(torch.arange(
                n_batch, device = device), edge_diff_slices)

            correct_idx = batch.edge_index[0] <= batch.edge_index[1]
            #batch_e_idx = batch_e[correct_idx]
            n_edges = scatter(correct_idx.long(), batch_e, reduce = "sum")
           
            batch.edge_index = batch.edge_index[:,correct_idx]
           
            new_slices = torch.cumsum(torch.cat((torch.zeros(1,device=device, dtype=torch.long),n_edges)),0).tolist()

            batch.__slices__["edge_index"] =  new_slices     
            return batch


if __name__ == "__main__":
    import torch_geometric as pyg

    edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = pyg.data.Data(x=x, edge_index=edge_index.t().contiguous())
    data_list = [data, data]
    loader = pyg.data.DataLoader(data_list, batch_size=2)
    batch = next(iter(loader))
    layer = TopologyLayer(1,1, dim1=True)

    out = layer(data.x, data.edge_index)