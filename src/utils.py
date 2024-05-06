import torch
import json
import torch.nn.functional as F


def get_query_emb(words, doc2vec_model, embeddings_dict, save_to_file=False, embeddings_path=None):
    vec = doc2vec_model.infer_vector(words, epochs=500)
    query = " ".join(words)
    if query in embeddings_dict:
        vec = embeddings_dict[query]
    else:
        embeddings_dict[query] = vec.tolist()
        if save_to_file:
            with open(embeddings_path, 'w') as f:
                json.dump(embeddings_dict, f)

    q_emb = torch.Tensor(vec)
    
    return q_emb

def get_top_experts(query_emb, experts_emb, topk=20):
    cosine_similarities = F.cosine_similarity(query_emb.unsqueeze(0), experts_emb, dim=1)
    tops = cosine_similarities.topk(topk)
    return tops

def get_rank(query_emb, experts_emb, expert_mapped):
    cosine_similarities = F.cosine_similarity(query_emb.unsqueeze(0), experts_emb, dim=1)
    sorted_indices = torch.argsort(cosine_similarities, descending=True)

    expert_rank = (sorted_indices == expert_mapped).nonzero(as_tuple=False)[0, 0]
    return expert_rank.item() + 1

def get_score(query_emb, experts_emb, expert_mapped):
    cosine_similarities = F.cosine_similarity(query_emb.unsqueeze(0), experts_emb[expert_mapped], dim=1)
    return cosine_similarities.detach().cpu().item()


def get_kth(query_emb, experts_emb, k):
    cosine_similarities = F.cosine_similarity(query_emb.unsqueeze(0), experts_emb, dim=1)
    sorted_values, _ = torch.sort(cosine_similarities, descending=True)
    return sorted_values[k - 1]