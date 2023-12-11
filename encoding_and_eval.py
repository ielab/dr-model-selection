
from utils import read_and_write
import pickle

from data.dataset_collection import Datasets
from beir.retrieval.evaluation import EvaluateRetrieval
from model.model_zoo import CustomModel, BeirModels
import numpy as np
import faiss
from utils.read_and_write import read_doc_enc_from_pickle
import gc, os
from utils.get_args import get_args


def tokenize_and_save(args, models, model_names,
                      # model_dir,
                      corpus):

    """
    Encodes the corpus and saves the encoding into a file
    :param args: arguments from the input
    :param beir_models: BEIR MODELs class
    :param corpus: corpus
    :return: name where to save the file
    """

    # Sort the documents by its size
    corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
                        reverse=True)
    corpus = [corpus[cid] for cid in corpus_ids]
    itr = range(0, len(corpus), args.corpus_chunk_size)

    for model_name in model_names:
        model = models.load_model(model_name, model_name_or_path=None) # args.model_path)

        #  Encoding
        for batch_num, corpus_start_idx in enumerate(itr):
            corpus_end_idx = min(corpus_start_idx + args.corpus_chunk_size, len(corpus))
            # Returns numpy arrays
            sub_corpus_embeddings = model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                batch_size=32,
                convert_to_tensor=False
            )
            # Save results in a file
            read_and_write.save_enc_to_pickle(sub_corpus_embeddings, corpus_ids[corpus_start_idx:corpus_end_idx],
                                              dataset_name=args.dataset_name,#args.dataset_name+"_train",
                                              model_name=model_name,
                                              log_dir=args.embedding_dir,
                                              batch_num=batch_num)

            print("Saved batch", batch_num, "of", len(itr), "batches")
    return True

def search(query_embeddings, doc_embeddings, top_k=1000, score_function="dot"):
    """
    Extracts top_k documents based on the queries.
    Saves the scores and the ids of the extracted documents in a file

    Implemented with faiss library

    :param query_embeddings:
    :param doc_embeddings:
    :param top_k: How many docs to extract per query
    :param score_function: "dot" or "cos_sim"retrieve_and_eval.py
    :return: Scores and Associated document indices
    """
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    # If normalized - becomes Cosine similarity
    if score_function == "cos_sim":
        faiss.normalize_L2(doc_embeddings)
        faiss.normalize_L2(query_embeddings)

    elif score_function != "dot":
        raise "Unknown score function"

    # Otherwise - Dot product
    index.add(doc_embeddings)
    # To save the index -> faiss.write_index()
    # Search for query embeddings
    scores, indices = index.search(query_embeddings, top_k)
    return scores, indices


def save_search_results(queries, doc_ids, scores, indices,
                        where_to_save="del2.pkl"):
    """
    :return: results in a format of a dictionary,
             suitable for EvaluateRetrieval function from BEIR lib
    """
    query_ids = list(queries.keys())
    assert len(query_ids) == len(scores)
    res = {}
    for i, ind_query in enumerate(query_ids):
        res[ind_query] = {}
        for score, indice in zip(scores[i], indices[i]):
            doc_name = doc_ids[indice]
            res[ind_query][doc_name] = float(score)
    with open(where_to_save, 'wb') as f:
        pickle.dump(res, f)
    return res


def save_eval_results(qrels, results, where_to_save):
    eval_retrieval = EvaluateRetrieval()
    # Can add 1000 here if needed
    eval_results = eval_retrieval.evaluate(qrels=qrels, results=results, k_values=[1, 3, 5, 10, 100, 1000])
    print(eval_results)
    with open(where_to_save, 'wb') as f:
        pickle.dump(eval_results, f)


def run_evaluation(args, models, names):
    # for model_name in model_names:
    for model_name in names:
        # model_name = beir_models.names[5]
        model = models.load_model(model_name, model_name_or_path=None)
        # contriever_model = CustomModel()
        # model = contriever_model.load_model()

        dataset = Datasets(args.dataset_dir)
        if args.dataset_name == "msmarco_train":
            # query_dname = "msmarco"
            query_dname = "msmarco_train"
            split = "dev"
        else:
            query_dname = args.dataset_name
            split = "test"
        _, queries, qrels = dataset.load_dataset(query_dname,  # args.dataset_name,
                                                 load_corpus=False,
                                                 split=split)
        # Get document embeddings
        doc_embeds, doc_ids = read_doc_enc_from_pickle(args.dataset_name, model_name, args.embedding_dir)
        # read_doc_enc_from_pickle(args.dataset_name, model_name, args.embedding_dir)

        # Get query embeddings
        query_list = [queries[qid] for qid in queries]

        query_embeds = model.encode_queries(query_list, batch_size=32, show_progress_bar=True,
                                            convert_to_tensor=False)

        scores, indices = search(query_embeds, doc_embeds, top_k=1000,
                                 score_function=models.score_function[model_name])
        # Save results
        # Example: ./log_dir/nfcorpus/sch_nfcorpus_contriever.pkl
        log_dir = os.path.join(args.log_dir, "search_results", args.dataset_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        name = "sch_{}_{}.pkl".format(args.dataset_name, model_name)
        path_to_file = os.path.join(log_dir, name)
        results = save_search_results(queries, doc_ids, scores, indices, where_to_save=path_to_file)


        log_dir = os.path.join(args.log_dir, "eval_results", args.dataset_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        name = "eval_{}_{}.pkl".format(args.dataset_name, model_name)
        path_to_file = os.path.join(log_dir, name)
        save_eval_results(qrels, results, path_to_file)
        del doc_embeds, doc_ids, queries, qrels
        gc.collect()


def run_encoding(args,  models, names):

    dataset = Datasets(args.dataset_dir)
    split = "test"
    corpus, queries, qrels = dataset.load_dataset(args.dataset_name, split=split,
                                                  load_corpus=True)
    if args.little_corpus:
        corp2_keys = ['MED-10', 'MED-14', 'MED-118', 'MED-301', 'MED-306',
                      'MED-329', 'MED-330', 'MED-332', 'MED-334', 'MED-335',
                  'MED-398', 'MED-557', 'MED-666', 'MED-691', 'MED-692', 'MED-1130']
        corpus_2 = {key: corpus[key] for key in corp2_keys}
        corpus = corpus_2
    tokenize_and_save(args, models, names, corpus)


def run_encoding_or_eval():
    args = get_args()
    for models in [BeirModels(args.model_path), CustomModel(model_dir=args.model_path)]:
        for model_name in models.names:
            args.model_name = model_name
            print("Start with model", model_name)
            if args.task == "encode":
                run_encoding(args, models, [args.model_name])
            elif args.task == "eval":
                run_evaluation(args, models, [args.model_name])
            else:
                raise "Unknown task"


if __name__ == "__main__":
    run_encoding_or_eval()
