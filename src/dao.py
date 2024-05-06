from typing import Dict, List

import importlib
import json
import torch
import pickle
import torch_geometric
from torch_geometric.nn import GAE

def load_models(config: Dict, device: torch.device) -> Dict:
    dataset_configs = config['datasets']
    result = {}
    for dataset in dataset_configs:
        model_dict = dataset_configs[dataset]['models']
        result[dataset] = {
            'expert_search_models': [],
            'link_prediction': None,
            'doc2vec': None
        }
        for model_config in model_dict['expert_search']:
            model_name = model_config['model_name']
            models_module = importlib.import_module("model.models")
            model_class = getattr(models_module, model_name)
            model_parameters = model_config['params']

            model = model_class(*model_parameters)
            model.load_state_dict(torch.load(model_config['path']))
            model.to(device)
            result[dataset]['expert_search_models'].append(model)

        link_prediction_config = model_dict['link_prediction']
        link_prediction_model_name = link_prediction_config['model_name']
        models_module = importlib.import_module("model.models")
        model_class = getattr(models_module, link_prediction_model_name)
        model_parameters = link_prediction_config['params']

        model = GAE(model_class(*model_parameters))
        model.load_state_dict(torch.load(link_prediction_config['path']))
        model.to(device)
        result[dataset]['link_prediction'] = model

        doc2vec_path = model_dict['doc2vec']['path']
        with open(doc2vec_path, 'rb') as f:
            doc2vec_model = pickle.load(f)
        result[dataset]['doc2vec'] = doc2vec_model
        embeddings_path = model_dict['doc2vec']['embeddings']
        with open(embeddings_path) as f:
            embeddings_dict = json.load(f)
        result[dataset]['embeddings'] = embeddings_dict

    return result


def load_data(config: Dict, device: torch.device) -> Dict:
    result = {}
    dataset_configs = config['datasets']

    for dataset in dataset_configs:
        data_dict = dataset_configs[dataset]['data']
        with open(data_dict['graph'], 'rb') as f:
            graph = pickle.load(f)
        with open(data_dict['author2id'], 'rb') as f:
            author2id = pickle.load(f)
        with open(data_dict['all_skills'], 'rb') as f:
            all_skills = pickle.load(f)

        graph_vec = torch_geometric.utils.from_networkx(graph)
        graph_x = graph_vec.x.float().to(device)
        graph_edge_index = graph_vec.edge_index.to(device)
        mapping = dict(zip(graph.nodes(), range(graph.number_of_nodes())))
        mapping2id = {v: k for k, v in mapping.items()}
        id2author = {v: k for k, v in author2id.items()}

        result[dataset] = {
            'graph': graph,
            'author2id': author2id,
            'all_skills': all_skills,
            'graph_vec': graph_vec,
            'graph_x': graph_x,
            'graph_edge_index': graph_edge_index,
            'mapping': mapping,
            'mapping2id': mapping2id,
            'id2author': id2author,
        }
    return result