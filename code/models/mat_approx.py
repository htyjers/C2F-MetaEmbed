"""
The script that implements clustering algorithm.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loading.data_loader import DataLoader, convert_sp_mat_to_sp_tensor, \
    convert_sp_tensor_to_sp_mat
import scipy.sparse as sp
import numpy as np
import os
from timeit import default_timer as timer
from datetime import timedelta
import dgl
import pickle
torch.autograd.set_detect_anomaly(True)
from sklearn.decomposition import SparsePCA
import torch.nn.functional as F
class MatApprox(nn.Module):
    def __init__(self, opt: dict, data_loader: DataLoader):
        super(MatApprox, self).__init__()
        self.opt = opt
        self.latent_dim = opt["latent_dim"]
        self.field_dims = opt["field_dims"]
        self.num_clusters = opt["num_clusters"]
        self.num_clusters_ft = opt["num_clusters_ft"]
        self.num_layers = opt["num_layers"]
        # A: shape N x N
        self.norm_adj_graph = data_loader.norm_adj_graph
        self.num_composition_centroid = opt["num_composition_centroid"]
        self.num_refine = opt["top_k_ft"]
        # centroid embs
        if not opt["use_metis_init"]:
            print("randomly initialize centroid assignment")
            self.centroid_embs = torch.zeros((self.num_clusters, self.latent_dim))
            self.finetune_embs = torch.zeros((self.num_clusters_ft, self.latent_dim))
            # init using normal
            nn.init.normal_(self.centroid_embs)
            # embedding assignment
            self.centroid_assignment = self.random_init_centroid_assignment(
                (sum(self.field_dims), self.num_clusters),
                non_zero_row_count=self.num_composition_centroid).to(opt["device_id"])
            self.finetune_assignment = self.random_init_centroid_assignment(
            (self.num_clusters, self.num_clusters_ft),
            non_zero_row_count=self.num_refine).to(opt["device_id"])
        else:
            # metis subgraph cut
            print("Using metis partitioning for assignment init")
            self.centroid_assignment = self.init_centroid_assignment_using_metis(
                (sum(self.field_dims), self.num_clusters),
                non_zero_row_count=self.num_composition_centroid).to(opt["device_id"])
            self.finetune_assignment = self.random_init_centroid_assignment(
            (self.num_clusters, self.num_clusters_ft),
            non_zero_row_count=self.num_refine).to(opt["device_id"])
            self.centroid_embs = torch.zeros((self.num_clusters, self.latent_dim))
            self.finetune_embs = torch.zeros((self.num_clusters_ft, self.latent_dim))
            # init using normal
            nn.init.normal_(self.centroid_embs)
            nn.init.normal_(self.finetune_embs)
        self.centroid_embs = nn.Parameter(self.centroid_embs)
        self.finetune_embs = nn.Parameter(self.finetune_embs)


        self.assignment_save_path = os.path.join(self.opt["res_prepath"], "assignment")
        self.finetune_assignment_save_path = os.path.join(self.opt["res_prepath"], "finetun_assignment")
        self.centroid_save_path = os.path.join(self.opt["res_prepath"], "centroids")
        self.finetune_save_path = os.path.join(self.opt["res_prepath"], "finetunes")

        os.makedirs(self.assignment_save_path, exist_ok=True)
        os.makedirs(self.finetune_assignment_save_path, exist_ok=True)
        os.makedirs(self.centroid_save_path, exist_ok=True)
        os.makedirs(self.finetune_save_path, exist_ok=True)

        assigment_file_name = os.path.join(self.assignment_save_path, f"init.npz")
        sp.save_npz(assigment_file_name, convert_sp_tensor_to_sp_mat(self.centroid_assignment))

        centroid_file_name = os.path.join(self.centroid_save_path, "init.npz")
        sp.save_npz(centroid_file_name, sp.csr_matrix(self.centroid_embs.data.detach().cpu().numpy()))

        if self.opt["sign_ft"]==1:
            assigment_file_name = os.path.join(self.finetune_assignment_save_path, "init.npz")
            sp.save_npz(assigment_file_name, convert_sp_tensor_to_sp_mat(self.finetune_assignment))
            finetune_file_name = os.path.join(self.finetune_save_path, "init.npz")
            sp.save_npz(finetune_file_name, sp.csr_matrix(self.finetune_embs.data.detach().cpu().numpy()))


        # expanded adj graph: shape (N + c1 + c2) x (N + c1 + c2).
        self.expanded_norm_adj_graph = None
        self.expanded_norm_adj_graph = self.get_expanded_adj_graph(self.centroid_assignment,self.finetune_assignment)

        print('assignment and expanded norm adj graph updated')
        non_zero_percentage, avg_row_non_zero_count = self.get_emb_non_zero_info(
            self.centroid_assignment)
        non_zero_avg_value = self.centroid_assignment.values().mean()
        text = f"Init - centroid assignment stat:\n" \
               f"% of non-zero values {non_zero_percentage * 100:.2f}%, " \
               f"avg non-zero/row = {avg_row_non_zero_count:.2f},\n" \
               f"avg non-zero value = {non_zero_avg_value:.2f}.\n"
        print(text, file=self.opt["performance_fp"], flush=True)
        print(text)

    def init_finetune(self):
        random_indices = np.random.choice(self.num_clusters, size=self.num_clusters_ft, replace=True)
        selected_rows = self.centroid_embs[random_indices]
        n_components = min(self.centroid_embs.shape[1], self.opt['n_components_ft'])
        sparsepca = SparsePCA(n_components, alpha=1)
        sparse_components = sparsepca.fit_transform(selected_rows.detach().numpy())
        if n_components < self.centroid_embs.shape[1]:
            n_components = sparse_components.shape[1]
            padded_rows = []
            for row in sparse_components:
                row_with_padding = np.zeros(self.centroid_embs.shape[1], dtype=row.dtype)
                non_zero_positions = np.sort(np.random.choice(self.centroid_embs.shape[1], n_components, replace=False))
                row_with_padding[non_zero_positions] = row
                padded_rows.append(row_with_padding)
            sparse_components_padded = np.vstack(padded_rows)
            self.finetune_embs.data = torch.from_numpy(sparse_components_padded).float()
        else:
            self.finetune_embs.data = torch.from_numpy(sparse_components).float()

        print("Using metis partitioning for assignment init")
        finetune_assignment = self.init_centroid_assignment_using_metis1(
            (sum(self.opt["field_dims"]), self.opt["num_clusters_ft"]),
            non_zero_row_count=self.opt["num_clusters_ft"]).to(self.opt["device_id"])
        attn_scores = torch.matmul(self.centroid_assignment.to(self.opt["device_id"]).transpose(0, 1),
                                   finetune_assignment)
        attn_weights = attn_scores.to_dense()
        attn_weights = F.softmax(attn_weights, dim=1)
        top_k_vals, top_k_idxes = torch.topk(attn_weights, largest=True, k=self.opt["top_k_ft"])

        assignment2 = torch.zeros(*attn_weights.shape).type_as(attn_weights).scatter(1, top_k_idxes, top_k_vals)

        assignment2 = assignment2.to_sparse_coo()

        self.finetune_assignment = assignment2.coalesce()

        self.expanded_norm_adj_graph = self.get_expanded_adj_graph(self.centroid_assignment, self.finetune_assignment)

    def get_expanded_adj_graph(self, assignment_mat, finetune_assignment_mat) -> torch.sparse.Tensor:
        """
        Compute assignment appended adjacency graph in form
        For Coarse Training Stage
        [A    S1  0]
        [S1.T 0   0]
        [0    0   0]
        For Fine Turning Stage
        [A    S1    0]
        [S1.T  0   S2]
        [0    S2.T  0]
        :return sparse graph in dimension (N + c1 + c2) x (N + c1 + c2)
        """
        device_id = finetune_assignment_mat.get_device() #self.opt["device_id"]#
        c2 = finetune_assignment_mat.shape[1]
        dim = sum(assignment_mat.shape) + c2
        time_start = timer()
        new_graph = sp.dok_matrix((dim, dim), dtype=np.float32).tolil()
        new_graph[:assignment_mat.shape[0], :assignment_mat.shape[0]] = self.norm_adj_graph

        assignment_mat = convert_sp_tensor_to_sp_mat(assignment_mat).tolil()
        finetune_assignment_mat = convert_sp_tensor_to_sp_mat(finetune_assignment_mat).tolil()

        if self.opt["sign_ft"] == 0:
            new_graph[:assignment_mat.shape[0], assignment_mat.shape[0]:assignment_mat.shape[0]+self.num_clusters] = assignment_mat[:,:self.num_clusters]
            new_graph[assignment_mat.shape[0]:assignment_mat.shape[0]+self.num_clusters, :assignment_mat.shape[0]] = assignment_mat[:,:self.num_clusters].T

        ########################################
        else:
            new_graph[:assignment_mat.shape[0], assignment_mat.shape[0]:assignment_mat.shape[0]+self.num_clusters] = assignment_mat[:,:self.num_clusters]
            new_graph[assignment_mat.shape[0]:assignment_mat.shape[0]+self.num_clusters, :assignment_mat.shape[0]] = assignment_mat[:,:self.num_clusters].T
            new_graph[assignment_mat.shape[0]:assignment_mat.shape[0]+self.num_clusters,assignment_mat.shape[0]+self.num_clusters:] = finetune_assignment_mat[:,:]
            new_graph[assignment_mat.shape[0]+self.num_clusters:,assignment_mat.shape[0]:assignment_mat.shape[0]+self.num_clusters] = finetune_assignment_mat[:,:].T
        ########################################

        new_graph = new_graph.todok()
        rowsum = np.array(new_graph.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        new_graph = d_mat.dot(new_graph).dot(d_mat)
        new_graph = convert_sp_mat_to_sp_tensor(new_graph.tocoo()).coalesce().to(device_id)
        time_end = timer()
        print(f"Expanded adj graph updated, time elapsed = {timedelta(seconds=time_end - time_start)}.")
        assert 0 <= new_graph.values().min() and 1 >= new_graph.values().max()
        return new_graph

    import torch
    import dgl
    import numpy as np
    import os
    import pickle

    def init_centroid_assignment_using_metis(self, shape: tuple, non_zero_row_count: int = 2, selected_assignment_weight: float = .5):
        assignment_mat = torch.zeros(shape)
        try:
            with open(os.path.join(self.opt["data_path"], "metis_assignment.pickle"), "rb") as fp:
                true_assignments = pickle.load(fp)
            print("METIS assignment loaded")
        except:
            print("Generating METIS assignment")
            graph = dgl.from_scipy(self.norm_adj_graph, eweight_name="norm_weight")
            true_assignments = dgl.metis_partition_assignment(graph, k=self.num_clusters)
        all_entities = torch.arange(shape[0])
        for idx in range(self.num_clusters):
            selected_entities = all_entities[true_assignments == idx]
            if self.num_composition_centroid == 1:
                assignment_mat[selected_entities, idx] = 1.
                continue
            else:
                assignment_mat[selected_entities, idx] = selected_assignment_weight
                remaining_clusters = list(range(self.num_clusters))
                remaining_clusters.remove(idx)
                for _entity in selected_entities:
                    other_assignments = np.random.choice(remaining_clusters, non_zero_row_count - 1, replace=False)
                    assignment_mat[_entity, other_assignments] = (1 - selected_assignment_weight) / (non_zero_row_count - 1)
        assert torch.allclose(torch.ones(shape[0]).type_as(assignment_mat), assignment_mat.sum(1))

        # Convert the assignment matrix to a sparse COO format
        indices = torch.nonzero(assignment_mat, as_tuple=False).t()
        values = assignment_mat[indices[0], indices[1]]
        assignment_sparse = torch.sparse_coo_tensor(indices, values, size=shape)

        # Coalesce the sparse tensor to remove duplicate entries
        assignment_sparse = assignment_sparse.coalesce()

        return assignment_sparse

    def init_centroid_assignment_using_metis1(self, shape: tuple, non_zero_row_count: int = 2, selected_assignment_weight: float = .5):
        assignment_mat = torch.zeros(shape)
        try:
            with open(os.path.join(self.opt["data_path"], "metis_assignment.pickle"), "rb") as fp:
                true_assignments = pickle.load(fp)
            print("METIS assignment loaded")
        except:
            print("Generating METIS assignment")
            graph = dgl.from_scipy(self.norm_adj_graph, eweight_name="norm_weight")
            true_assignments = dgl.metis_partition_assignment(graph, k=self.num_clusters_ft)
        all_entities = torch.arange(shape[0])
        for idx in range(self.num_clusters_ft):
            selected_entities = all_entities[true_assignments == idx]
            if self.num_composition_centroid == 1:
                assignment_mat[selected_entities, idx] = 1.
                continue
            else:
                assignment_mat[selected_entities, idx] = selected_assignment_weight
                remaining_clusters = list(range(self.num_clusters_ft))
                remaining_clusters.remove(idx)
                for _entity in selected_entities:
                    other_assignments = np.random.choice(remaining_clusters, non_zero_row_count - 1, replace=False)
                    assignment_mat[_entity, other_assignments] = (1 - selected_assignment_weight) / (non_zero_row_count - 1)
        assert torch.allclose(torch.ones(shape[0]).type_as(assignment_mat), assignment_mat.sum(1))

        # Convert the assignment matrix to a sparse COO format
        indices = torch.nonzero(assignment_mat, as_tuple=False).t()
        values = assignment_mat[indices[0], indices[1]]
        assignment_sparse = torch.sparse_coo_tensor(indices, values, size=shape)

        # Coalesce the sparse tensor to remove duplicate entries
        assignment_sparse = assignment_sparse.coalesce()

        return assignment_sparse

    @staticmethod
    def random_init_centroid_assignment(shape: tuple, **kwargs):
        """
        :param row_wise_max_nonzero_percentage/non_zero_row_count

        return a binarized sparse tensor half of the cells to be
        filled as 0
        """
        row_wise_max_nonzero_percentage = kwargs.get("row_wise_max_nonzero_percentage", None)
        non_zero_row_count = kwargs.get("non_zero_row_count", None)
        if not row_wise_max_nonzero_percentage and not non_zero_row_count:
            row_wise_max_nonzero_percentage = .001
            max_centroid_count = int(shape[1] * row_wise_max_nonzero_percentage)
        elif row_wise_max_nonzero_percentage:
            max_centroid_count = int(shape[1] * row_wise_max_nonzero_percentage)
        else:
            max_centroid_count = non_zero_row_count
        print(f"max centroid count = {max_centroid_count}")

        out = torch.zeros(shape)
        for _ in range(shape[0]):
            choice = torch.tensor(np.random.choice(np.arange(shape[1]), max_centroid_count,
                                                   replace=False))
            out[_, choice[0]] = .9
            out[_, choice[1:]] = (1 - .9) / len(choice[1:])
        assert torch.allclose(torch.ones(shape[0]).type_as(out), out.sum(1))
        return out.to_sparse_coo().coalesce()

    def get_full_embs(self):
        """
        Compute the full embedding table
        """
        if self.opt["sign_ft"] == 0:
            return self.centroid_assignment @ self.centroid_embs, self.centroid_embs, torch.zeros_like(self.finetune_embs)
        else:
            sparse_finetune_embs = torch.sign(self.finetune_embs) * torch.nn.functional.relu(torch.abs(self.finetune_embs) - self.opt["threshold_ft"])
            full_finetune_embs = self.finetune_assignment.to(self.opt["device_id"]) @ sparse_finetune_embs.to(self.opt["device_id"])
            centroid_embs = self.opt["gate"] * full_finetune_embs + (1-self.opt["gate"]) * self.centroid_embs
            return self.centroid_assignment @ self.centroid_embs,centroid_embs, sparse_finetune_embs


    def calculate_sparsity(self, finetune_embs, thresholds):
        """
        计算不同阈值下稀疏化嵌入向量的稀疏度百分比。

        参数:
        - finetune_embs (torch.Tensor): 原始嵌入向量
        - thresholds (list of float): 不同的阈值

        返回:
        - dict: 各阈值对应的稀疏度百分比
        """
        sparsity_percentages = {}

        for threshold in thresholds:
            # 稀疏化处理
            sparse_finetune_embs = torch.sign(finetune_embs) * torch.nn.functional.relu(
                torch.abs(finetune_embs) - threshold)

            # 计算非零元素的比例
            nonzero_count = torch.count_nonzero(sparse_finetune_embs).item()
            total_elements = sparse_finetune_embs.numel()
            sparsity_percentage = 100 * (1 - nonzero_count / total_elements)

            # 存储结果
            sparsity_percentages[threshold] = sparsity_percentage

        return sparsity_percentages

    def forward(self, mode: str = "train"):
        """
        Gather new full embedding, then pass in GCN to get GCN_emb.
        Update emb assignment using matrix approximation thereafter.
        """
        assert mode in ["train", "test"]
        if self.opt["sign_ft"] == 1:
            full_embs,centroid_embs, finetune_embs = self.get_full_embs()
            full_embs = torch.tanh(full_embs)
            concat_embs = torch.cat((full_embs, centroid_embs, torch.tanh(finetune_embs)))
        else:
            full_embs,centroid_embs, _ = self.get_full_embs()
            full_embs = torch.tanh(full_embs)
            concat_embs = torch.cat((full_embs, centroid_embs, torch.zeros_like(self.finetune_embs)))

        gcn_embs = [concat_embs]
        for _layer in range(self.num_layers):
            concat_embs = self.expanded_norm_adj_graph @ concat_embs

            if self.opt["sign_ft"] == 1:
                h_full_embs, h_centroid_embs, h_finetune_embs = concat_embs[:len(full_embs)], concat_embs[len(full_embs):len(
                    full_embs) + self.num_clusters], concat_embs[len(full_embs) + self.num_clusters:]
                h_full_embs -= self.centroid_assignment.to(self.opt["device_id"]) @ h_centroid_embs
                h_centroid_embs -= self.finetune_assignment.to(self.opt["device_id"]) @ h_finetune_embs
                h_centroid_embs = self.opt["gate"] * self.finetune_assignment.to(self.opt["device_id"]) @ h_finetune_embs + (1-self.opt["gate"]) * h_centroid_embs
                h_full_embs = h_full_embs + self.centroid_assignment.to(self.opt["device_id"]) @ h_centroid_embs
                concat_embs = torch.cat((h_full_embs, h_centroid_embs, h_finetune_embs))

            gcn_embs.append(concat_embs)
        gcn_embs = torch.stack(gcn_embs, dim=1).mean(dim=1)
        assert not torch.isnan(gcn_embs).any()
        h_full_embs, h_centroid_embs, h_finetune_embs= gcn_embs[:len(full_embs)], gcn_embs[len(full_embs):len(full_embs)+self.num_clusters], gcn_embs[len(full_embs)+self.num_clusters:]

        if mode == "test":
            return h_full_embs
        if self.opt["update_assignment"]:
            with torch.no_grad():
                self.update_assignment(h_full_embs, h_centroid_embs, h_finetune_embs)
        return full_embs, h_full_embs

    def update_assignment(self, h_full_embs, h_centroid_embs, h_finetune_embs):
        # update assignment, and corresponding expanded adj graph

        assignment, unique_count,assignment2,unique_count2 = self.compute_assignment(h_full_embs, h_centroid_embs, h_finetune_embs)

        if self.opt["sign_ft"] == 0:
            self.centroid_assignment = assignment.to_sparse_coo().coalesce()
            self.expanded_norm_adj_graph = self.get_expanded_adj_graph(self.centroid_assignment,self.finetune_assignment)
            print('assignment and expanded norm adj graph updated')
        else:
            self.finetune_assignment = assignment2.to_sparse_coo().coalesce()
            self.expanded_norm_adj_graph = self.get_expanded_adj_graph(self.centroid_assignment,self.finetune_assignment)
            print('assignment and expanded norm adj graph updated')
        #
        non_zero_percentage, avg_row_non_zero_count = self.get_emb_non_zero_info(assignment)
        avg_value, non_zero_avg_value = assignment.mean(), self.centroid_assignment.values().mean()
        text = f"Epoch {self.opt['epoch_idx']}, Batch {self.opt['batch_idx']} - centroid assignment stat:\n" \
               f"% of non-zero values {non_zero_percentage * 100:.2f}%, " \
               f"avg non-zero/row = {avg_row_non_zero_count:.2f},\navg value: {avg_value:.2f}, " \
               f"avg non-zero value = {non_zero_avg_value:.2f},\n" \
               f"#centroids utilized = {unique_count:d}.\n" \
               f"#refine utilized = {unique_count2:d}.\n"
        print(text, file=self.opt["performance_fp"], flush=True)
        print(text)

    def get_centroid_embedding_sparsity(self) -> float:
        """
        Get the sparsity rate of the centroid embedding
        """
        density = self.centroid_embs.count_nonzero() / np.prod(self.centroid_embs.shape)
        return (1 - density).item()

    def get_full_embedding_sparsity(self) -> float:
        """
        Get the sparsity rate of the entire embedding table
        """
        full_emb = self.get_full_embs()
        density = full_emb.count_nonzero() / np.prod(full_emb.shape)
        return (1 - density).item()

    def compute_assignment(self, h_full_embs, h_centroid_embs, h_finetune_embs: torch.Tensor):

        if self.opt["sign_ft"] == 1:
            h_centroid_inv = torch.linalg.pinv(h_centroid_embs)
            assignment = h_full_embs @ h_centroid_inv
            assignment1 = assignment

            h_finetune_inv = torch.linalg.pinv(h_finetune_embs)
            assignment2 = h_full_embs @ h_finetune_inv

            attn_scores = torch.matmul(assignment1.transpose(0, 1), assignment2)
            attn_weights = F.softmax(attn_scores, dim=1)
            top_k_vals, top_k_idxes = torch.topk(attn_weights, largest=True, k=self.opt["top_k_ft"])
            assignment2 = torch.zeros(*attn_weights.shape).type_as(attn_weights).scatter(1, top_k_idxes, top_k_vals)
            unique_count2 = len(top_k_idxes.unique())

            top_k_vals, top_k_idxes = torch.topk(assignment, largest=True, k=self.num_composition_centroid)
            assignment = torch.zeros(*assignment.shape).type_as(assignment).scatter(1, top_k_idxes, top_k_vals)
            unique_count = len(top_k_idxes.unique())
            return assignment, unique_count,assignment2,unique_count2

        else:
            h_centroid_inv = torch.linalg.pinv(h_centroid_embs)
            assignment = h_full_embs @ h_centroid_inv
            top_k_vals, top_k_idxes = torch.topk(assignment, largest=True, k=self.num_composition_centroid)
            assignment = torch.zeros(*assignment.shape).type_as(assignment).scatter(1, top_k_idxes, top_k_vals)
            unique_count = len(top_k_idxes.unique())
            return assignment, unique_count,assignment,unique_count



    @staticmethod
    def get_emb_non_zero_info(emb):
        """
        :return percentage of non-zero elements in the entire tensor,
                avg non-zero element w.r.t. #rows
        """
        if emb.is_sparse:
            if not emb.is_coalesced():
                emb = emb.coalesce()
            non_zero_count = len(emb.values())
        else:
            non_zero_count = emb.count_nonzero()
        non_zero_percentage = non_zero_count / (emb.shape[0] * emb.shape[1])
        avg_row_non_zero_count = non_zero_count / len(emb)
        return non_zero_percentage, avg_row_non_zero_count

    @staticmethod
    def sparse_dropout(x, rate, noise_shape):
        """
        Perform dropout on sparse norm adj graph; used by NGCF
        """
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))
