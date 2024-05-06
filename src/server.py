from typing import Dict, List
from flask import Flask, request, abort
from flask_cors import CORS, cross_origin

import gc
import os
import sys
import torch
import torch_geometric
import networkx as nx
from tqdm import tqdm
from utils import get_query_emb, get_top_experts, get_rank
from explanation import ExES
import yaml

from dao import load_data, load_models

sys.path.insert(0, os.path.abspath(os.path.join("../")))

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_path = "./exes_config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
exes_models = load_models(config, device)
exes_data = load_data(config, device)


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def get_author(author_id: int, graph_x: torch.Tensor, mapping: Dict, id2author: Dict, all_skills: List) -> Dict:
    result = {"name": id2author[author_id], "id": author_id, "skills": []}
    skills_indices = graph_x[mapping[author_id]].nonzero(as_tuple=True)[0].tolist()
    result["skills"] = [all_skills[ind] for ind in skills_indices]

    return result


@app.route('/search')
@cross_origin()
@torch.inference_mode()
def search():
    query = request.args.get('query')
    dataset = request.args.get('dataset')
    model_num = int(request.args.get('model')) - 1  # To make the model_num 0-based
    query = query.lower()

    # Load corresponding dataset and model
    model = exes_models[dataset]['expert_search_models'][model_num]
    doc2vec_model = exes_models[dataset]['doc2vec']
    graph_x = exes_data[dataset]['graph_x']
    graph_edge_index = exes_data[dataset]['graph_edge_index']
    all_skills = exes_data[dataset]['all_skills']
    id2author = exes_data[dataset]['id2author']
    mapping = exes_data[dataset]['mapping']
    mapping2id = exes_data[dataset]['mapping2id']
    embeddings_dict = exes_models[dataset]['embeddings']
    embeddings_path = config['datasets'][dataset]['models']['doc2vec']['embeddings']

    query = query.split()
    query = list(filter(lambda x: x in all_skills, query))  # Remove keywords that are not in the universe of skills
    if not query:
        return {}

    query.sort()

    q_emb = get_query_emb(query, doc2vec_model, embeddings_dict, save_to_file=True, embeddings_path=embeddings_path).to(device)
    emb = model(graph_x, graph_edge_index)
    top_experts = get_top_experts(q_emb, emb, graph_x.shape[0])

    answers = []
    for ind, expert in enumerate(tqdm(top_experts.indices)):
        author_id = mapping2id[expert.item()]
        author = get_author(author_id, graph_x, mapping, id2author, all_skills)
        author["rank"] = ind
        author["graph"] = ""
        answers.append(author)

    result = {
        "answer": answers,
        "query_words": query,
        "query_embedding": q_emb.tolist()
    }
    clear_cache()
    return result


@app.route('/teamsearch', methods=["POST"])
@cross_origin()
@torch.inference_mode()
def teamsearch():
    data = request.json
    query_emb = torch.Tensor(data["qemb"])
    dataset = data['dataset']
    query_words = data["query"]
    expert = data["expert"]
    model_num = int(request.args.get('model')) - 1  # To make the model_num 0-based

    model = exes_models[dataset]['expert_search_models'][model_num]
    graph = exes_data[dataset]['graph']
    graph_x = exes_data[dataset]['graph_x']
    graph_edge_index = exes_data[dataset]['graph_edge_index']
    all_skills = exes_data[dataset]['all_skills']
    id2author = exes_data[dataset]['id2author']
    mapping = exes_data[dataset]['mapping']

    emb = model(graph_x, graph_edge_index).to(device)
    query_emb = query_emb.to(device)
    team = {}
    for word in query_words:

        layers = nx.bfs_layers(graph, [expert])

        for candidates in layers:
            good_candidates = []
            for candidate in candidates:
                cand = get_author(candidate, graph_x, mapping, id2author, all_skills)
                if word in cand["skills"]:
                    good_candidates.append(cand)
            if len(good_candidates) > 0:
                rank_cand = []
                for candidate in good_candidates:
                    rank = get_rank(query_emb, emb, mapping[candidate['id']])
                    rank_cand.append((rank, candidate))
                rank_cand.sort()
                for ind, item in enumerate(rank_cand):
                    rank_cand[ind][1]["rank"] = rank_cand[ind][0]
                team[word] = {"selected": rank_cand[0][1], "candidates": [x[1] for x in rank_cand]}
                break

    team_members = [item["selected"]["id"] for item in team.values()]
    graph_nodes = {expert}
    main_member_neighborhood = list(nx.ego_graph(graph, expert, radius=1).nodes)
    for n in main_member_neighborhood:
        graph_nodes.add(n)
    for member in team_members:
        if member not in graph_nodes:
            path = nx.shortest_path(graph, source=expert, target=member)
            for node in path:
                graph_nodes.add(node)
    team_graph = nx.subgraph(graph, list(graph_nodes))
    print(team_members)
    nodes = {f"node{item}": {'name': id2author[item], "rank": 2, "mainNode": item == expert or item in team_members,
                             "size": 17 if item == expert or item in team_members else 4} for item in team_graph}
    edges = {f"edge{i}": {"source": f"node{src}", "target": f"node{dst}", "status": "old"} for i, (src, dst) in
             enumerate(team_graph.edges) if src != dst}

    return {
        "team": team,
        "graph": {"nodes": nodes, "edges": edges}
    }


@app.route('/explain/<method>', methods=['POST'])
@cross_origin()
@torch.inference_mode()
def explain(method):
    data = request.json
    query_emb = torch.Tensor(data["qemb"]).to(device)
    dataset = data['dataset']
    query_words = data["query"]
    expert = data["expert"]
    topk = data["topk"]
    num_explanations = data["num_explanations"]
    max_explanation_size = data["max_explanation_size"]
    expert_in_topk = data["expert_in_topk"]
    model_num = int(data["model"]) - 1  # To make the model_num 0-based
    main_team_node = data.get("main_team_node", expert)
    is_team = data.get("isteam", False)
    pruning = data.get("pruning", True)

    model = exes_models[dataset]['expert_search_models'][model_num]
    doc2vec_model = exes_models[dataset]['doc2vec']
    gae_model = exes_models[dataset]['link_prediction']
    graph = exes_data[dataset]['graph']
    graph_vec = exes_data[dataset]['graph_vec']
    all_skills = exes_data[dataset]['all_skills']
    author2id = exes_data[dataset]['author2id']
    id2author = exes_data[dataset]['id2author']
    mapping = exes_data[dataset]['mapping']
    embeddings_dict = exes_models[dataset]['embeddings']
    embeddings_path = config['datasets'][dataset]['models']['doc2vec']['embeddings']

    explainer = ExES(dataset, device, model, doc2vec_model, graph, graph_vec, all_skills, author2id, embeddings_dict, embeddings_path)
    explainer.num_explanations = num_explanations
    explainer.max_explanation_size = max_explanation_size
    explainer.beam_size = config['exes_parameters']['beam_size']
    explainer.tau = config['exes_parameters']['tau']
    explainer.pruning = pruning

    # Saliency Explanations
    if method == "shap":
        shap_values, shap_plot = explainer.skill_shap_values(query_words, query_emb, expert, topk)
        result = {
            "plot": shap_plot
        }
    elif method == "shap_query":
        shap_values, shap_plot = explainer.query_shap_values(query_words, expert, topk)
        result = {
            "plot": shap_plot
        }
    elif method == "shap_edge":
        shap_plot, nodes, edges = explainer.edge_shap_values(query_emb, expert, topk)

        for item in nodes:
            nodes[item]["name"] = id2author[int(item.strip("node"))]
            nodes[item]["size"] = 10
        result = {
            "plot": shap_plot,
            "nodes": nodes,
            "edges": edges
        }
    else:  # Counterfactual Explanations
        if method == "remove_skill_expert":
            explanations = explainer.remove_skill_expert(query_words, query_emb, expert, topk)

        elif method == "add_skill_expert":
            explanations = explainer.add_skill_expert(query_words, query_emb, expert, topk)

        elif method == "add_skill_query":
            explanations = explainer.add_skill_query(query_words, expert, topk, expert_in_topk)

        elif method == "remove_edge_expert":
            explanations = explainer.remove_edge_expert(query_emb, expert, topk)

        elif method == "add_edge_expert":
            explanations = explainer.add_edge_expert(query_emb, expert, topk, gae_model)

        results = []
        for ex in explanations:
            changes = ex.get('changes')
            new_tops = ex.get('tops_ids')
            new_graph_x = ex.get('new_graph_x')
            new_rank = ex.get('new_rank')
            new_tops = [get_author(author, new_graph_x, mapping, id2author, all_skills) for author in new_tops]
            for i, author in enumerate(new_tops):
                author["rank"] = i
            if method == "add_skill_expert" or method == "remove_skill_expert":
                changes = [f"{a} - {b}" for a, b in changes]
            result = {
                "new_tops": new_tops,
                "changes": changes,
                "new_rank": new_rank
            }

            if method == "remove_edge_expert" or method == "add_edge_expert":
                rank_dict = ex.get('rank_dict')
                # Build the displayed graph in the format of v-network-graph (https://dash14.github.io/v-network-graph/getting-started.html)
                new_graph = nx.ego_graph(graph, expert, radius=1)
                edge_list = []

                # Add edges between nodes in explanation and nodes in Expert's neighborhood to the new graph
                for node in new_graph.nodes:
                    if node != expert:
                        edge_list.append((min(expert, node), max(expert, node)))
                    for (source, dest) in changes:
                        if source in graph.neighbors(node):
                            edge_list.append((min(source, node), max(source, node)))
                        if dest in graph.neighbors(node):
                            edge_list.append((min(dest, node), max(dest, node)))

                edge_list = list(set(edge_list))
                if method == "remove_edge_expert":
                    edge_list = [(source, dest) for source, dest in edge_list if
                                 (source, dest) not in changes and (dest, source) not in changes]

                # Create old edges which exist in the new graph as well
                edges = {f"edge{i}": {"source": f"node{source}", "target": f"node{dest}", "status": "old"} for
                         i, (source, dest) in enumerate(edge_list)}

                num_edges = len(edges)
                explanation_nodes = set()

                # Include added/removed edges to the new graph
                for i, (source, dest) in enumerate(changes):
                    explanation_nodes.add(source)
                    explanation_nodes.add(dest)
                    edges[f"edge{i + num_edges}"] = {"source": f"node{source}", "target": f"node{dest}",
                                                     "status": "new" if method != "remove_edge_expert" else "removed"}

                node_list = set([val["source"].strip("node") for val in edges.values()]
                                + [val["target"].strip("node") for val in edges.values()])
                node_list = [int(item) for item in node_list]
                nodes = {f"node{item}": {'name': id2author[item], "rank": rank_dict[item],
                                         "mainNode": item == expert or item in explanation_nodes,
                                         "size": 17 if item == expert or item in explanation_nodes else 4} for item in node_list}

                result["new_graph"] = {"nodes": nodes, "edges": edges}

                # Replace user ids with their names in the explanations
                changes = [f"{id2author[a]} - {id2author[b]}" for a, b in changes]
                result["changes"] = changes

            if is_team:
                print("seearching for new team")
                new_query = result["explanation"] if method == "add_skill_query" else query_words
                new_q_emb = rank_dict.to(device) if method == "add_skill_query" else query_emb
                new_graph = graph.copy()
                if method == "remove_edge_expert":
                    for (i, j) in result["explanation"]:
                        new_graph.remove_edge(i, j)
                elif method == "add_edge_expert":
                    for (i, j) in result["explanation"]:
                        new_graph.add_edge(i, j)
                new_graph_vec = torch_geometric.utils.from_networkx(new_graph)
                new_graph_edge_index = new_graph_vec.edge_index.to(device)
                new_emb = model(new_graph_x, new_graph_edge_index)
                team = {}
                for word in new_query:

                    layers = nx.bfs_layers(new_graph, [main_team_node])

                    for candidates in layers:
                        good_candidates = []
                        for candidate in candidates:
                            cand = get_author(candidate, new_graph_x, mapping, id2author, all_skills)
                            if word in cand["skills"]:
                                good_candidates.append(cand)
                        if len(good_candidates) > 0:
                            rank_cand = []
                            for candidate in good_candidates:
                                rank = get_rank(new_q_emb, new_emb, mapping[candidate['id']])
                                rank_cand.append((rank, candidate))
                            rank_cand.sort()
                            for ind, item in enumerate(rank_cand):
                                rank_cand[ind][1]["rank"] = rank_cand[ind][0]
                            team[word] = {"selected": rank_cand[0][1], "candidates": [x[1] for x in rank_cand]}
                            break

                team_members = [item["selected"]["id"] for item in team.values()]
                print(team_members)
                new_graph_nodes = {main_team_node}
                main_member_neighborhood = list(nx.ego_graph(new_graph, main_team_node, radius=1).nodes)
                for n in main_member_neighborhood:
                    new_graph_nodes.add(n)
                for member in team_members:
                    if member not in new_graph_nodes:
                        path = nx.shortest_path(new_graph, source=main_team_node, target=member)
                        for node in path:
                            new_graph_nodes.add(node)
                new_team_graph = nx.subgraph(graph, list(new_graph_nodes))
                new_nodes = {f"node{item}": {'name': id2author[item], "rank": 2,
                                             "mainNode": item == main_team_node or item in team_members,
                                             "size": 17 if item == main_team_node or item in team_members else 4} for item in new_team_graph}
                new_edges = {f"edge{i}": {"source": f"node{src}", "target": f"node{dst}", "status": "old"} for
                             i, (src, dst) in enumerate(new_team_graph.edges) if src != dst}

                result["new_graph"] = {"nodes": new_nodes, "edges": new_edges}

            results.append(result)

        coeff = -1 if expert_in_topk else 1

        results.sort(key=lambda item: (len(item["changes"]), coeff * item["new_rank"]))
        clear_cache()
        return results

    clear_cache()
    return result


if __name__ == '__main__':
    app.run("0.0.0.0", port=8094)
