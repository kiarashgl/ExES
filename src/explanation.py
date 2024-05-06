import heapq
from itertools import combinations
from typing import List, Dict

import networkx as nx
import numpy as np
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from flask import abort
from gensim.models import doc2vec
from shap.plots import colors
from tqdm.auto import tqdm

from src.utils import get_query_emb, get_top_experts, get_rank, get_score, get_kth


class ExES:
    def __init__(self, dataset: str, device: torch.device, model: nn.Module,
                 doc2vec_model: doc2vec.Doc2Vec, graph: nx.Graph, graph_vec: torch_geometric.data.Data,
                 all_skills: List[str], author2id: Dict[str, int], embeddings_dict: Dict, embeddings_path: str) -> None:
        self.device = device
        self.dataset = dataset
        self.model = model
        self.doc2vec_model = doc2vec_model
        self.graph = graph
        self.all_skills = all_skills
        self.author2id = author2id
        self.embeddings_dict = embeddings_dict

        self.graph_x = graph_vec.x.float().to(device)
        self.graph_edge_index = graph_vec.edge_index.to(device)
        self.total_experts = self.graph_x.shape[0]

        # Contains the mapping of expert ids in self.graph to their indices in self.graph_x
        self.mapping = dict(zip(self.graph.nodes(), range(self.graph.number_of_nodes())))
        self.mapping2id = {v: k for k, v in self.mapping.items()}
        self.id2author = {v: k for k, v in author2id.items()}

        self.num_explanations = 10
        self.max_explanation_size = 5
        self.beam_size = 30
        self.num_tokens = 10
        self.tau = 0.1
        self.d_radius = 1

        self.pruning = True
        self.tqdm_disable = True

    def remove_skill_expert(self, query_words: List[str], query_emb: torch.Tensor, expert: int, topk: int) -> List[Dict]:
        if self.pruning:
            neighborhood = nx.ego_graph(self.graph, expert, radius=self.d_radius).nodes
        else:
            neighborhood = self.graph

        neighborhood_skills = set()
        for xp in neighborhood:
            xp_skills = [i for i, skill in enumerate(self.graph_x[self.mapping[xp]]) if skill > 0]
            xp_words = [self.all_skills[i] for i in xp_skills]
            neighborhood_skills = neighborhood_skills.union(set(xp_words))

        query_emb = query_emb.cuda()

        # Find top keywords to include in removal perturbations
        query_filtered_words = [i for i in query_words if i in self.doc2vec_model.wv.key_to_index]
        candidate_skills = self.doc2vec_model.wv.most_similar(positive=query_filtered_words, topn=len(self.all_skills))
        candidate_skills = [self.all_skills.index(i) for i, j in candidate_skills if i in neighborhood_skills]
        if self.pruning:
            candidate_skills = candidate_skills[:self.num_tokens]

        new_graph_x = self.graph_x.clone().cuda()
        experts_emb = self.model(self.graph_x, self.graph_edge_index)
        initial_score = get_score(query_emb, experts_emb, self.mapping[expert])
        best_ranks = []

        # Find individual-skill pairs that are effective in expert's ranking
        for xp in neighborhood:
            xp_skills = [i for i, skill in enumerate(self.graph_x[self.mapping[xp]]) if skill > 0]

            for skill in tqdm(xp_skills, disable=self.tqdm_disable):
                if skill not in candidate_skills:
                    continue
                new_graph_x[self.mapping[xp]][skill] = 0
                new_experts_emb = self.model(new_graph_x, self.graph_edge_index)
                score = get_score(query_emb, new_experts_emb, self.mapping[expert])
                new_graph_x[self.mapping[xp]][skill] = 1
                if score < initial_score:
                    best_ranks.append((score, skill, xp))
        if self.pruning:
            best_ranks = sorted(best_ranks)[:self.beam_size]
        else:
            best_ranks = sorted(best_ranks)

        effective_skill_individual_pairs = [(i[1], i[2]) for i in best_ranks]

        # Expand the selected t features until reaching valid explanations
        explanations = []
        for adding_skill_len in tqdm(range(1, self.max_explanation_size + 1), disable=self.tqdm_disable):
            skills_subsets = combinations(effective_skill_individual_pairs, adding_skill_len)
            for adding_skills in tqdm(skills_subsets, disable=self.tqdm_disable):
                for skill, xp in adding_skills:
                    new_graph_x[self.mapping[xp]][skill] = 0
                experts_emb = self.model(new_graph_x, self.graph_edge_index)

                tops = get_top_experts(query_emb, experts_emb, topk)
                tops_ids = [self.mapping2id[person.item()] for person in tops.indices]
                new_rank = get_rank(query_emb, experts_emb, self.mapping[expert])
                if expert not in tops_ids[:topk]:
                    changes = [(self.all_skills[sk], self.id2author[xp]) for sk, xp in adding_skills]
                    new_rank = get_rank(query_emb, experts_emb, self.mapping[expert])
                    explanations.append({"changes": changes, "tops_ids": tops_ids, "new_graph_x": new_graph_x, "new_rank": new_rank})
                    if len(explanations) >= self.num_explanations:
                        for skill, xp in adding_skills:
                            new_graph_x[self.mapping[xp]][skill] = 1
                        return explanations
                for skill, xp in adding_skills:
                    new_graph_x[self.mapping[xp]][skill] = 1

        if explanations:
            return explanations

        abort(404)

    def add_skill_expert(self, query_words: List[str], query_emb: torch.Tensor, expert: int, topk: int) -> List[Dict]:
        if self.pruning:
            neighborhood = nx.ego_graph(self.graph, expert, radius=self.d_radius).nodes
        else:
            neighborhood = self.graph

        query_emb = query_emb.cuda()
        expert_skills = [i for i, skill in enumerate(self.graph_x[self.mapping[expert]]) if skill > 0]
        candidate_words = list(set(query_words))  # | set(expert_skill_words))
        candidate_words = [i for i in candidate_words if i in self.doc2vec_model.wv.key_to_index]
        candidate_skills2 = self.doc2vec_model.wv.most_similar(positive=candidate_words, topn=self.num_tokens)
        candidate_skills2 = [self.all_skills.index(item[0]) for item in candidate_skills2]
        candidate_skills = list(set(candidate_skills2))

        new_graph_x = self.graph_x.clone().cuda()
        experts_emb = self.model(self.graph_x, self.graph_edge_index)
        initial_score = get_score(query_emb, experts_emb, self.mapping[expert])

        expert_skills_copy = {xp: new_graph_x[self.mapping[xp]].clone() for xp in neighborhood}
        best_ranks = []
        skills_subsets = candidate_skills

        # Find individual-skill pairs that are effective in expert's ranking
        for xp in neighborhood:
            for skill in tqdm(skills_subsets, disable=self.tqdm_disable):
                if new_graph_x[self.mapping[xp]][skill] == 1:
                    continue

                new_graph_x[self.mapping[xp]][skill] = 1
                new_experts_emb = self.model(new_graph_x, self.graph_edge_index)
                score = get_score(query_emb, new_experts_emb, self.mapping[expert])
                if score > initial_score:
                    best_ranks.append((score, skill, xp))
                new_graph_x[self.mapping[xp]][skill] = 0

        if self.pruning:
            best_ranks = sorted(best_ranks, reverse=True)[:self.beam_size]
        else:
            best_ranks = sorted(best_ranks, reverse=True)

        candidate_skills = [(i[1], i[2]) for i in best_ranks]
        explanations = []
        for adding_skill_len in tqdm(range(1, self.max_explanation_size + 1), disable=self.tqdm_disable):
            skills_subsets = combinations(candidate_skills, adding_skill_len)
            for adding_skills in tqdm(skills_subsets, disable=self.tqdm_disable):
                for skill, xp in adding_skills:
                    new_graph_x[self.mapping[xp]][skill] = 1
                experts_emb = self.model(new_graph_x, self.graph_edge_index)

                tops = get_top_experts(query_emb, experts_emb, topk)
                tops_ids = [self.mapping2id[person.item()] for person in tops.indices]
                new_rank = get_rank(query_emb, experts_emb, self.mapping[expert])
                new_score = get_score(query_emb, experts_emb, self.mapping[expert])
                if expert in tops_ids[:topk]:
                    changes = [(self.all_skills[sk], self.id2author[xp]) for sk, xp in adding_skills]
                    new_rank = get_rank(query_emb, experts_emb, self.mapping[expert])
                    explanations.append({"changes": changes, "tops_ids": tops_ids, "new_graph_x": new_graph_x.clone(), "new_rank": new_rank})

                    if len(explanations) >= self.num_explanations:
                        for xp in neighborhood:
                            new_graph_x[self.mapping[xp]] = expert_skills_copy[xp].clone()
                        return explanations
                changed_experts = {xp for _, xp in adding_skills}
                for xp in changed_experts:
                    new_graph_x[self.mapping[xp]] = expert_skills_copy[xp].clone()

        if explanations:
            return explanations

        abort(404)

    def add_skill_query(self, query_words: List[str], expert: int, topk: int, expert_in_topk: bool) -> List[Dict]:
        expert_skills = [i for i, skill in enumerate(self.graph_x[self.mapping[expert]]) if skill > 0]
        query_skills = [self.all_skills.index(skill) for skill in query_words]
        if self.pruning:
            if expert_in_topk:
                expert_skill_words = [self.all_skills[i] for i in expert_skills if
                                      self.all_skills[i] in self.doc2vec_model.wv.key_to_index]
                candidate_skills = self.doc2vec_model.wv.most_similar(positive=query_words, negative=expert_skill_words,
                                                                      topn=self.num_tokens)
                candidate_skills = [item[0] for item in candidate_skills]
            else:
                expert_skill_words = [self.all_skills[i] for i in expert_skills if
                                      self.all_skills[i] in self.doc2vec_model.wv.key_to_index]
                candidate_skills = self.doc2vec_model.wv.most_similar(positive=query_words + expert_skill_words,
                                                                      topn=self.num_tokens)
                candidate_skills = [item[0] for item in candidate_skills]
        else:
            candidate_skills = list(set(self.all_skills) - set(query_words))

        explanations = []
        for adding_skill_len in tqdm(range(1, self.max_explanation_size + 1), disable=self.tqdm_disable):
            skills_subsets = combinations(candidate_skills, adding_skill_len)
            for adding_skills in skills_subsets:
                new_query = query_words + list(adding_skills)
                new_query_emb = get_query_emb(new_query, self.doc2vec_model, self.embeddings_dict)
                new_query_emb = new_query_emb.cuda()

                experts_emb = self.model(self.graph_x, self.graph_edge_index)
                tops = get_top_experts(new_query_emb, experts_emb, topk)
                tops_ids = [self.mapping2id[person.item()] for person in tops.indices]
                if (expert_in_topk and expert not in tops_ids[:topk]) or (
                        not expert_in_topk and expert in tops_ids[:topk]):
                    changes = adding_skills
                    new_rank = get_rank(new_query_emb, experts_emb, self.mapping[expert])
                    explanations.append({"changes": changes, "tops_ids": tops_ids, "new_graph_x": self.graph_x, "new_rank": new_rank})
                    if len(explanations) >= self.num_explanations:
                        return explanations
        if explanations:
            return explanations
        abort(404)

    def remove_edge_expert(self, query_emb: torch.Tensor, expert: int, topk: int) -> List[Dict]:
        query_emb = query_emb.cuda()

        experts_emb = self.model(self.graph_x, self.graph_edge_index)

        kth_score = get_kth(query_emb, experts_emb, topk + 1)
        initial_score = get_score(query_emb, experts_emb, self.mapping[expert])
        initial_rank = get_rank(query_emb, experts_emb, self.mapping[expert])
        removing_link_candidates = []
        neighborhood = nx.ego_graph(self.graph, expert, radius=self.d_radius + 1).edges
        for node1, node2 in tqdm(neighborhood, disable=self.tqdm_disable):
            if node2 > node1:
                node1, node2 = node2, node1
            removing_link_candidates.append((node1, node2))
        explanations = []
        removing_link_candidates = list(set(removing_link_candidates))
        priority_queue = [(0, set())]
        mark = set()
        clone_edge_index = self.graph_edge_index.clone()

        effective_edges = []
        effective2 = []
        print("len candidates", len(removing_link_candidates))
        for cand in tqdm(removing_link_candidates, disable=False):
            node1, node2 = cand
            new_edge_index = self.graph_edge_index
            index_to_drop = (new_edge_index[0] == self.mapping[node1]) & (new_edge_index[1] == self.mapping[node2])
            new_edge_index = new_edge_index[:, ~index_to_drop]

            index_to_drop = (new_edge_index[1] == self.mapping[node2]) & (new_edge_index[0] == self.mapping[node1])
            new_edge_index = new_edge_index[:, ~index_to_drop]

            experts_emb = self.model(self.graph_x, new_edge_index)
            score = get_score(query_emb, experts_emb, self.mapping[expert])
            if score < initial_score:
                if node1 == expert or node2 == expert:
                    effective2.append((score, cand))
                else:
                    effective_edges.append((score, cand))
            if len(effective_edges) + len(effective2) >= self.beam_size:
                break
        if self.pruning:
            effective2.sort()
            effective_edges = effective2 + effective_edges
            effective_edges.sort()
            effective_edges = effective_edges[:self.beam_size]
        else:
            effective_edges = effective_edges + effective2
        effective_edges = [item[1] for item in effective_edges]
        print("effective edges:", len(effective_edges))
        while len(explanations) < self.num_explanations and priority_queue:
            if self.pruning:
                w_open = heapq.nsmallest(self.beam_size, priority_queue)
            priority_queue = []
            for rnk, subset in w_open:
                if len(subset) > self.max_explanation_size:
                    continue
                for cand in tqdm(effective_edges, disable=self.tqdm_disable):
                    if cand in subset:
                        continue
                    new_subset = frozenset(subset | {cand})
                    if new_subset in mark:
                        continue
                    new_edge_index = self.graph_edge_index

                    for node1, node2 in new_subset:
                        index_to_drop = (new_edge_index[0] == self.mapping[node1]) & (
                                    new_edge_index[1] == self.mapping[node2])
                        new_edge_index = new_edge_index[:, ~index_to_drop]

                        index_to_drop = (new_edge_index[1] == self.mapping[node2]) & (
                                    new_edge_index[0] == self.mapping[node1])
                        new_edge_index = new_edge_index[:, ~index_to_drop]
                    experts_emb = self.model(self.graph_x, new_edge_index)
                    score = get_score(query_emb, experts_emb, self.mapping[expert])
                    if score >= initial_score:
                        continue
                    if score < kth_score:
                        tops = get_top_experts(query_emb, experts_emb, topk)
                        tops_ids = [self.mapping2id[person.item()] for person in tops.indices]
                        if expert not in tops_ids[:topk]:
                            cosine_similarities = F.cosine_similarity(query_emb.unsqueeze(0), experts_emb, dim=1)
                            sorted_indices = torch.argsort(cosine_similarities, descending=True).tolist()
                            rank_dict = {}
                            for rank, item in enumerate(sorted_indices):
                                rank_dict[self.mapping2id[item]] = rank + 1
                            new_rank = get_rank(query_emb, experts_emb, self.mapping[expert])
                            new_neighbors = list(new_subset)
                            explanations.append({"changes": new_neighbors, "tops_ids": tops_ids, "new_graph_x": self.graph_x, "new_rank": new_rank, "rank_dict": rank_dict})
                            if len(explanations) >= self.num_explanations:
                                return explanations
                    else:
                        heapq.heappush(priority_queue, (score, new_subset))
                    mark.add(new_subset)
        if explanations:
            return explanations
        abort(404)

    def add_edge_expert(self, query_emb: torch.Tensor, expert: int, topk: int, gae_model: nn.Module) -> List[Dict]:
        neighbors = list(self.graph.neighbors(expert))
        print(neighbors)
        query_emb = query_emb.cuda()

        experts_emb = self.model(self.graph_x, self.graph_edge_index)

        kth_score = get_kth(query_emb, experts_emb, topk)
        initial_score = get_score(query_emb, experts_emb, self.mapping[expert])
        initial_rank = get_rank(query_emb, experts_emb, self.mapping[expert])
        z = gae_model.encode(self.graph_x, self.graph_edge_index)
        predicted_edges = gae_model.decoder.forward_all(z)
        added_link_candidates = []
        if self.pruning:
            neighborhood = nx.ego_graph(self.graph, expert, radius=1).nodes
            for node in tqdm(neighborhood):
                cur_node_candidates = []
                node_neighbors = list(self.graph.neighbors(node))
                for ind, prob in enumerate(predicted_edges[self.mapping[node]]):
                    if self.mapping2id[ind] != node and self.mapping2id[ind] not in node_neighbors:
                        min_node = min(node, self.mapping2id[ind])
                        max_node = max(node, self.mapping2id[ind])
                        cur_node_candidates.append((prob.item(), (min_node, max_node)))

                cur_node_candidates = sorted(cur_node_candidates, reverse=True)
                if node != expert:
                    cur_node_candidates = cur_node_candidates[:self.num_tokens]
                added_link_candidates.extend(cur_node_candidates)
        else:
            all_possible_edges = combinations(self.graph.nodes, 2)
            added_link_candidates = set(all_possible_edges) - set(self.graph.edges)
        explanations = []
        added_link_candidates = list(set(added_link_candidates))
        priority_queue = [(0, set())]
        mark = set()
        clone_edge_index = self.graph_edge_index.clone()

        effective_edges = []
        effective2 = []
        for _, cand in tqdm(added_link_candidates):
            new_edges = []
            node_1, node_2 = cand
            new_edges.append([self.mapping[node_1], self.mapping[node_2]])
            new_edges.append([self.mapping[node_2], self.mapping[node_1]])
            new_edge_index = torch.tensor(new_edges).T.cuda()
            new_edge_index = torch.cat((clone_edge_index, new_edge_index), dim=1)

            experts_emb = self.model(self.graph_x, new_edge_index)
            score = get_score(query_emb, experts_emb, self.mapping[expert])
            if score > initial_score:
                if node_1 == expert or node_2 == expert:
                    effective2.append((score, cand))
                else:
                    effective_edges.append((score, cand))
        effective2.sort(reverse=True)
        effective_edges = effective2[self.num_tokens] + effective_edges
        effective_edges.sort(reverse=True)
        effective_edges = effective_edges[:self.beam_size]
        effective_edges = [item[1] for item in effective_edges]
        while len(explanations) < self.num_explanations and priority_queue:
            if self.pruning:
                w_open = heapq.nsmallest(self.beam_size, priority_queue)
            priority_queue = []
            for rnk, subset in w_open:
                if len(subset) > self.max_explanation_size:
                    continue
                for cand in tqdm(effective_edges):
                    if cand in subset:
                        continue
                    new_subset = frozenset(subset | {cand})
                    if new_subset in mark:
                        continue
                    new_edges = []
                    for adding_edge in new_subset:
                        node_1, node_2 = adding_edge
                        new_edges.append([self.mapping[node_1], self.mapping[node_2]])
                        new_edges.append([self.mapping[node_2], self.mapping[node_1]])
                    new_edge_index = torch.tensor(new_edges).T.cuda()
                    new_edge_index = torch.cat((clone_edge_index, new_edge_index), dim=1)

                    experts_emb = self.model(self.graph_x, new_edge_index)
                    score = get_score(query_emb, experts_emb, self.mapping[expert])
                    if score <= initial_score:
                        continue
                    if score >= kth_score:
                        tops = get_top_experts(query_emb, experts_emb, topk)
                        tops_ids = [self.mapping2id[person.item()] for person in tops.indices]
                        if expert in tops_ids[:topk]:
                            cosine_similarities = F.cosine_similarity(query_emb.unsqueeze(0), experts_emb, dim=1)
                            sorted_indices = torch.argsort(cosine_similarities, descending=True).tolist()
                            rank_dict = {}
                            for rank, item in enumerate(sorted_indices):
                                rank_dict[self.mapping2id[item]] = rank + 1
                            new_rank = get_rank(query_emb, experts_emb, self.mapping[expert])
                            new_neighbors = list(new_subset)
                            explanations.append({"changes": new_neighbors, "tops_ids": tops_ids, "new_graph_x": self.graph_x, "new_rank": new_rank, "rank_dict": rank_dict})
                            if len(explanations) >= self.num_explanations:
                                return explanations
                    else:
                        heapq.heappush(priority_queue, (-score, new_subset))
                    mark.add(new_subset)

        if explanations:
            return explanations
        abort(404)

    def transform_plot(self, plot):
        plot = plot.replace("rgb(255.0, 0.0, 81.08083606031792)", "rgb(28.0, 162.0, 75.0)") \
            .replace("rgb(0.0, 138.56128015770724, 250.76166088685738)", "rgb(255.0, 0.0, 81.08083606031792)") \
            .replace("stroke:rgb(255, 195, 213)", "stroke:rgb(195, 255, 207)") \
            .replace("stroke:rgb(208, 230, 250)", "stroke:#ffe6e4")
        # print(plot)
        return plot

    def skill_shap_values(self, query_words, query_emb, expert, topk):
        masker = shap.maskers.Text(r"\W", "-1 ")
        if self.pruning:
            working_graph = nx.ego_graph(self.graph, expert, radius=1)
        else:
            working_graph = self.graph
        all_skill_graph = []
        for xp in working_graph.nodes:
            skills_indices = self.graph_x[self.mapping[xp]].nonzero(as_tuple=True)[0].tolist()
            expert_skills = [self.all_skills[i].strip() for i in skills_indices]
            all_skill_graph.extend(expert_skills)

        all_skill_graph = list(set(all_skill_graph))

        expert_skills_str = " ".join(list(map(str, all_skill_graph)))
        query_emb = query_emb.cuda()

        experts_emb = self.model(self.graph_x, self.graph_edge_index)
        kth_score = get_kth(query_emb, experts_emb, topk)

        def get_rankings(skills_list):
            result = []
            for skills in skills_list:
                new_skills = skills.split()
                new_skills = [item.strip() for item in new_skills]
                new_graph_x = self.graph_x.clone()

                removed_skills = {sk for sk in all_skill_graph if sk not in new_skills}
                removed_skills_ind = {self.all_skills.index(sk) for sk in removed_skills}

                for neighbor in all_skill_graph.nodes:
                    neighbor_ind = self.mapping[neighbor]
                    for sk in removed_skills_ind:
                        new_graph_x[neighbor_ind][sk] = 0
                experts_emb = self.model(new_graph_x, self.graph_edge_index).cuda()

                score = get_score(query_emb, experts_emb, self.mapping[expert])
                target = 1 if score >= kth_score else 0
                result.append(target)
            return result

        explainer = shap.Explainer(get_rankings, masker, seed=42)
        print(explainer.__class__)

        shap_values = explainer([expert_skills_str])

        plot = shap.plots.text(shap_values[0], display=False)

        plot = self.transform_plot(plot)
        return shap_values, plot

    def query_shap_values(self, query_words, expert, topk):
        masker = shap.maskers.Text(r"\W", "-1 ")
        query_skills_str = " ".join(query_words)
        expert_ind = self.mapping[expert]
        experts_emb = self.model(self.graph_x, self.graph_edge_index).cuda()

        def get_rankings(skills_list):
            result = []
            for query in skills_list:
                new_query = query.split()
                new_query_emb = get_query_emb(new_query, self.doc2vec_model, self.embeddings_dict)
                new_query_emb = new_query_emb.cuda()
                target = 1 if get_rank(new_query_emb, experts_emb, self.mapping[expert]) <= topk else 0

                result.append(target)
            return result

        explainer = shap.Explainer(get_rankings, masker)
        shap_values = explainer([query_skills_str])
        plot = shap.plots.text(shap_values[0], display=False)
        plot = self.transform_plot(plot)
        return shap_values, plot

    def edge_shap_values(self, query_emb, expert, topk):
        query_emb = query_emb.cuda()
        experts_emb = self.model(self.graph_x, self.graph_edge_index)
        kth_score = get_kth(query_emb, experts_emb, topk)

        mark = set()
        sg = []

        def shap_from_node_x(input_node):
            masker = shap.maskers.Text(r"\W", "-1 ")
            sgg = [(input_node, other) for other in self.graph.neighbors(input_node)]

            edges = [f"{e[0]}a{e[1]}" for e in sgg]

            def get_rankings(edges_list):
                result = []
                for neighbors in edges_list:
                    new_neighbors = neighbors.split()
                    removing_edges = []
                    saving_edges = []
                    for e in new_neighbors:
                        if e.strip() == "-1":
                            continue
                        i, j = e.split("a")
                        i = int(i)
                        j = int(j)
                        saving_edges.append((i, j))
                    for edge in sgg:
                        if (edge[0], edge[1]) not in saving_edges and (edge[1], edge[0]) not in saving_edges:
                            removing_edges.append(edge)
                    new_edge_index = self.graph_edge_index
                    for i, j in removing_edges:
                        index_to_drop = (new_edge_index[0] == self.mapping[i]) & (new_edge_index[1] == self.mapping[j])
                        new_edge_index = new_edge_index[:, ~index_to_drop]

                        index_to_drop = (new_edge_index[1] == self.mapping[i]) & (new_edge_index[0] == self.mapping[j])
                        new_edge_index = new_edge_index[:, ~index_to_drop]

                    experts_emb = self.model(self.graph_x, new_edge_index).cuda()

                    score = get_score(query_emb, experts_emb, self.mapping[expert])
                    # print(score)
                    target = 1 if score >= kth_score else 0
                    result.append(target)
                return result

            joined_edges = " ".join(edges)

            explainer = shap.Explainer(get_rankings, masker)
            shap_values = explainer([joined_edges])
            return shap_values

        candidate_nodes = [expert]
        candidate_edges = []
        mark.add(expert)
        distances = nx.shortest_path_length(self.graph, source=expert)
        # print(distances)
        while len(candidate_nodes) > 0:
            cur_node = candidate_nodes[0]
            candidate_nodes = candidate_nodes[1:]
            if distances[cur_node] >= 5:
                continue
            print("CUR NODE:", cur_node, self.id2author[cur_node])
            shap_values = shap_from_node_x(cur_node)
            output_values = shap_values.values.sum(0)  # + shap_values.base_values
            print(output_values)
            output_max = np.max(np.abs(shap_values.values))
            print("OUTPUTMAX", output_max)
            if output_max < 1e-8:
                continue
            for i, name in enumerate(shap_values.data[0]):
                shap_value = output_values[i]
                print("tau", self.tau)
                if abs(shap_value) / abs(output_max) < self.tau:
                    continue
                other_node = int(name.split("a")[1])
                if other_node not in mark:
                    mark.add(other_node)
                    candidate_nodes.append(other_node)
                    candidate_edges.append((cur_node, other_node))

        print(candidate_edges)

        masker = shap.maskers.Text(r"\W", "-1 ")
        sg = candidate_edges
        edges = [f"{e[0]}a{e[1]}" for e in sg]
        joined_edges = " ".join(edges)

        def get_rankings2(edges_list):
            result = []
            for neighbors in edges_list:
                new_neighbors = neighbors.split()
                removing_edges = []
                saving_edges = []
                for e in new_neighbors:
                    if e.strip() == "-1":
                        continue
                    i, j = e.split("a")
                    i = int(i)
                    j = int(j)
                    saving_edges.append((i, j))

                for edge in sg:
                    if (edge[0], edge[1]) not in saving_edges and (edge[1], edge[0]) not in saving_edges:
                        removing_edges.append(edge)
                new_edge_index = self.graph_edge_index
                for i, j in removing_edges:
                    index_to_drop = (new_edge_index[0] == self.mapping[i]) & (new_edge_index[1] == self.mapping[j])
                    new_edge_index = new_edge_index[:, ~index_to_drop]

                    index_to_drop = (new_edge_index[1] == self.mapping[i]) & (new_edge_index[0] == self.mapping[j])
                    new_edge_index = new_edge_index[:, ~index_to_drop]

                experts_emb = self.model(self.graph_x, new_edge_index).cuda()

                score = get_score(query_emb, experts_emb, self.mapping[expert])
                # print(score)
                target = 1 if score >= kth_score else 0
                result.append(target)
            return result

        explainer = shap.Explainer(get_rankings2, masker)
        shap_values = explainer([joined_edges])
        print(shap_values)
        output_values = shap_values.values.sum(0)  # + shap_values.base_values
        output_max = np.max(np.abs(output_values))

        experts_emb = self.model(self.graph_x, self.graph_edge_index)
        cosine_similarities = F.cosine_similarity(query_emb.unsqueeze(0), experts_emb, dim=1)
        sorted_indices = torch.argsort(cosine_similarities, descending=True).tolist()
        rank_dict = {}
        for rank, item in enumerate(sorted_indices):
            rank_dict[self.mapping2id[item]] = rank + 1
        graph_edges = []
        valid_nodes = set()
        all_nodes = set()

        for i, name in enumerate(shap_values.data[0]):
            scaled_value = 0.5 + 0.5 * output_values[i] / (output_max + 1e-8)
            color = colors.red_transparent_blue(scaled_value)
            color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), color[3])
            shap_value = output_values[i]
            if abs(shap_value) < 0.00001:
                continue
            if shap_value < 0:
                color = (255, 0, 81, color[3])
            else:
                color = (33, 223, 99, color[3])
            # print(name, color)
            node_1, node_2 = name.split("a")
            graph_edges.append((name.split("a"), color, round(output_values[i], 3)))
            all_nodes.add(int(node_1))
            all_nodes.add(int(node_2))

        nodes = {f"node{item}": {'name': item, "rank": rank_dict[item]} for item in all_nodes}
        edges = {f"edge{i}": {"source": f"node{src.strip()}", "target": f"node{dst.strip()}", "color": color,
                              "shap_value": shap_value} for i, ((src, dst), color, shap_value) in
                 enumerate(graph_edges)}
        valid_nodes = []
        for item in edges.values():
            valid_nodes.append(item["source"])
            valid_nodes.append(item["target"])
        valid_nodes = set(valid_nodes)
        nodes = {k: v for k, v in nodes.items() if k in valid_nodes}
        plot = shap.plots.text(shap_values[0], display=False)
        return plot, nodes, edges
