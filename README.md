Dense Retriever Model Selection
Code for SIGIR-AP paper "Selecting which Dense Retriever to use for Zero-Shot Search". Pdf can be found under the following link.

All model selection methods are summarised in model_selection_methods.py

In order to reproduce the results, first load all the models and the datasets.

Then, store the embeddings of each corpus in files for speeding up the subsequent computations. The following command will encode the dataset nfcorpus and store the embedding in a file PATH_TO_ENCODINGS/nfcorpus_MODEL_NAME.pkl

python encoding_and_eval.py \
--dataset_name nfcorpus \
--dataset_dir PATH_TO_DATA \
--model_path PATH_TO_MODELS \
--log_dir PATH_TO_LOGS \
--embedding_dir PATH_TO_ENCODINGS \
--task encode
Replace the task to eval for calculating the performance of the models (ngcd, MAP, recall)

Finally, for performing model selection experiments and computing Tau correlation between NDCG@10 and a score of each model selection method, run

python model_selection_methods.py \
--dataset_name nfcorpus \
--dataset_dir PATH_TO_DATA \
--model_path PATH_TO_MODELS \
--log_dir PATH_TO_LOGS \
--embedding_dir PATH_TO_ENCODINGS \
--task query_similarity
Possible model selection strategies: query_similarity, corpus_similarity, extracted_corpus_similarity, binary_entropy, query_alteration. For additional hyperparameters, see utils/get_args.py

If you find this useful for your work, please consider citing:

@inproceedings{10.1145/3624918.3625330,
author = {Khramtsova, Ekaterina and Zhuang, Shengyao and Baktashmotlagh, Mahsa and Wang, Xi and Zuccon, Guido},
title = {Selecting Which Dense Retriever to Use for Zero-Shot Search},
year = {2023},
isbn = {9798400704086},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3624918.3625330},
doi = {10.1145/3624918.3625330},
booktitle = {Proceedings of the Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region},
pages = {223–233},
numpages = {11},
keywords = {Model selection, Zero Shot Model Evaluation, Dense retrievers},
location = {<conf-loc>, <city>Beijing</city>, <country>China</country>, </conf-loc>},
series = {SIGIR-AP '23}
}
