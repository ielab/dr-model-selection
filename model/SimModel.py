
from typing import List, Dict
from model.model_collection import CustomDEModel

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np


class SimLMDEModel(CustomDEModel):
    def __init__(self, model_name_or_path, cache_dir, cuda):
        super().__init__()
        # Re-init this in the constructor
        self.query_encoder = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                       cache_dir=cache_dir).eval()
        if cuda:
            self.query_encoder = self.query_encoder.cuda()

        self.doc_encoder = self.query_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       cache_dir=cache_dir)
        self.config = None
        self.score_function = "dot"
        self.cuda = cuda

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries, batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i + batch_size]
                inputs = self.tokenizer(batch,
                                        max_length=32,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')

                if self.cuda:
                    inputs = {key: value.cuda() for key, value in inputs.items()}
                outputs = self.query_encoder(**inputs, return_dict=True)
                query_embedding = self._l2_normalize(outputs.last_hidden_state[:, 0, :])
                embeddings.append(query_embedding.cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings

    def tokenize_query(self, queries, **kwargs):
        inputs = self.tokenizer(queries,
                                max_length=32,
                                padding=True,
                                truncation=True,
                                add_special_tokens=False,
                                # return_tensors='pt'
                                )["input_ids"]
        return inputs

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus, batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(corpus), batch_size):
                titles = [doc['title'] if 'title' in doc.keys() else '-' for doc in corpus[i:i + batch_size]]
                texts = [doc['text'] for doc in corpus[i:i + batch_size]]
                inputs = self.tokenizer(titles,
                                        text_pair=texts,
                                        max_length=144,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')
                if self.cuda:
                    inputs = {key: value.cuda() for key, value in inputs.items()}
                outputs = self.doc_encoder(**inputs, return_dict=True)
                doc_embedding = self._l2_normalize(outputs.last_hidden_state[:, 0, :])
                embeddings.append(doc_embedding.cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings

    def tokenize_corpus(self, corpus, **kwargs):
        titles = [doc['title'] if 'title' in doc.keys() else '-' for doc in corpus]
        texts = [doc['text'] for doc in corpus]
        inputs = self.tokenizer(titles,
                                text_pair=texts,
                                max_length=350,
                                padding=True,
                                truncation=True,
                                add_special_tokens=False,
                                # return_tensors='pt'
                                )["input_ids"]
        return inputs

    def _l2_normalize(self, x: torch.Tensor):
        return torch.nn.functional.normalize(x, p=2, dim=-1)


class CoCondenserModel(CustomDEModel):
    def __init__(self, name, cache_dir, cuda):
        super().__init__()
        # Re-init this in the constructor

        self.query_encoder = AutoModel.from_pretrained(name,
                                                       cache_dir=cache_dir).eval()
        if cuda:
            self.query_encoder = self.query_encoder.cuda()

        self.doc_encoder = self.query_encoder
        if "co-condenser-marco-retriever" in name:
            self.tokenizer = AutoTokenizer.from_pretrained(name,
                                                           cache_dir=cache_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",  # name,
                                                           cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(name,
                                                       cache_dir=cache_dir)
        self.config = None
        self.score_function = "dot"
        self.cuda = cuda

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries, batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i + batch_size]
                inputs = self.tokenizer(batch,
                                        max_length=32,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')

                if self.cuda:
                    inputs = {key: value.cuda() for key, value in inputs.items()}
                outputs = self.query_encoder(**inputs, return_dict=True)
                query_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(query_embedding.cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings

    def tokenize_query(self, queries, **kwargs):
        inputs = self.tokenizer(queries,
                                max_length=32,
                                padding=True,
                                truncation=True,
                                add_special_tokens=False,
                                #return_tensors='pt',
                                )['input_ids']
        return inputs

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus, batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(corpus), batch_size):
                titles = [doc['title'] if 'title' in doc.keys() else '-' for doc in corpus[i:i + batch_size]]
                texts = [doc['text'] for doc in corpus[i:i + batch_size]]
                input_texts = [f'{title}{self.tokenizer.sep_token}{text}' for title, text in zip(titles, texts)]
                inputs = self.tokenizer(input_texts,
                                        max_length=144,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')
                if self.cuda:
                    inputs = {key: value.cuda() for key, value in inputs.items()}
                outputs = self.doc_encoder(**inputs, return_dict=True)
                doc_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(doc_embedding.cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings

    def tokenize_corpus(self, corpus):
        titles = [doc['title'] if 'title' in doc.keys() else '-' for doc in corpus]
        texts = [doc['text'] for doc in corpus]
        input_texts = [f'{title}{self.tokenizer.sep_token}{text}' for title, text in zip(titles, texts)]
        inputs = self.tokenizer(input_texts,
                                max_length=350,
                                padding=True,
                                truncation=True,
                                add_special_tokens=False,
                                # return_tensors='pt'
                                )['input_ids']
        return inputs