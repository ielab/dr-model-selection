
import os
import pickle
import numpy as np
import random


def read_doc_enc_from_pickle(dataset_name, model_name, log_dir):
    """
    Read the embedding from a pickled file
    :return: [array, array]: document embeddings and associated document ids
    """

    name = "{}_{}.pkl".format(dataset_name, model_name)
    # Example: ./embeddings/nfcorpus/nfcorpus_contriever.pkl
    log_dir = os.path.join(log_dir, dataset_name)

    embeddings, docids = [], []
    with open(os.path.join(log_dir, name), 'rb') as f:
        while 1:
            try:
                emb, ids = pickle.load(f)
                embeddings.append(emb)
                docids.append(ids)
            except EOFError:
                break
    embeddings, docids = np.concatenate(embeddings, axis=0), np.concatenate(docids, axis=0)
    return embeddings, docids


def get_embedding_subset(doc_embeds, subsample_size=1000000):
    if len(doc_embeds) > subsample_size:
        random_indx = random.sample(range(len(doc_embeds)), subsample_size)
    else:
        random_indx = list(range(len(doc_embeds)))
    return doc_embeds[random_indx]


def save_enc_to_pickle(embeddings, docids, dataset_name, model_name, log_dir, batch_num):
    name = "{}_{}.pkl".format(dataset_name, model_name)
    # Example: ./log_dir/nfcorpus/nfcorpus_contriever.pkl
    log_dir = os.path.join(log_dir, dataset_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # If file exists - remove it first
    if os.path.exists(os.path.join(log_dir, name)) and batch_num == 0:
        os.remove(os.path.join(log_dir, name))

    if batch_num == 0:
        writing_type = 'wb'
    else:
        writing_type = 'ab+'

    with open(os.path.join(log_dir, name), writing_type) as f:
        pickle.dump((embeddings, docids), f)
    return True


def save_result_to_pickle(result, result_dir, dataset_name, model_name, experiment_index):
    # Save results
    log_dir = os.path.join(result_dir, dataset_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # bentropy
    name = "{}_{}_{}.pkl".format(experiment_index, dataset_name, model_name)
    path_to_file = os.path.join(log_dir, name)
    with open(path_to_file, 'wb') as f:
        pickle.dump(result, f)
    return True


def load_search_results(log_dir, dataset_name, model_name):
    log_dir = os.path.join(log_dir, dataset_name)
    file_name = "sch_{}_{}.pkl".format(dataset_name, model_name)
    path_to_file = os.path.join(log_dir, file_name)
    with open(path_to_file, 'rb+') as f:
        search_results = pickle.load(f)
    return search_results



