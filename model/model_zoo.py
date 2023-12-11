
import os

from transformers import AutoTokenizer

import transformers
from transformers import AutoConfig
from model.contriever.src.beir_utils import DenseEncoderContrieverModel
from model.SimModel import SimLMDEModel, CoCondenserModel
from model.model_collection import ModelClass

from sentence_transformers import SentenceTransformer
from beir.retrieval import models


class CustomModel(ModelClass):
    def __init__(self, model_dir="/opt/data/IR_models/"):
        super().__init__(model_dir)
        self.names = ["contriever",
                      "simlm-base-msmarco-finetuned",
                      "co-condenser-marco-retriever",
                      #"character-bert-dr"
                      ]
        self.score_function = {"contriever": "dot",
                               "simlm-base-msmarco-finetuned": "dot",
                               "co-condenser-marco-retriever": "dot",
                               #"character-bert-dr": "dot"
                               }
        for name in self.names:
            self.source_datasets[name] = "msmarco_train"

    def load_model(self, name, cuda=True, model_name_or_path=None):
        assert name in self.names
        if name == "contriever":
            if model_name_or_path is None:
                model_name_or_path = "facebook/contriever-msmarco"
            model = DenseEncoderContrieverModel(model_name_or_path=model_name_or_path,
                                                cache_dir=self.model_dir,
                                                max_length=350,
                                                cuda=cuda)

        elif name == "simlm-base-msmarco-finetuned":
            if model_name_or_path is None:
                model_name_or_path = 'intfloat/simlm-base-msmarco-finetuned'
            model = SimLMDEModel(model_name_or_path=model_name_or_path,
                                 cache_dir=self.model_dir,
                                 cuda=cuda)

        elif name == "co-condenser-marco-retriever":
            if model_name_or_path is None:
                model_name_or_path = 'Luyu/co-condenser-marco-retriever'
            model = CoCondenserModel(name=model_name_or_path,
                                     cache_dir=self.model_dir, cuda=cuda)
        elif name == "character-bert-dr":
            model = CoCondenserModel(name=self.model_dir+'/models--arvin--character-bert-dr/',
                                     cache_dir=self.model_dir, cuda=cuda)
        else:
            raise "Unknown model name"
        return model


class BeirModels(ModelClass):
    def __init__(self, model_dir):
        super().__init__(model_dir)
        self.names = [
                # "facebook-dpr-question_encoder-multiset-base",
                 # "facebook-dpr-ctx_encoder-multiset-base",
                 # "nq-distilbert-base-v1",
                 "msmarco-MiniLM-L-6-v3",
                 "msmarco-MiniLM-L-12-v3",
                 "msmarco-distilbert-base-v2",
                 "msmarco-distilbert-base-dot-prod-v3",
                 "msmarco-distilbert-base-v3",
                 "msmarco-roberta-base-ance-firstp",
                 "msmarco-distilbert-base-tas-b",
                 # "BAAI-bge-large-zh-v1.5"
                 ]

        score_function, source_datasets = {}, {}
        for name in self.names:
            score_function[name] = "dot"
            source_datasets[name] = "msmarco_train"
        source_datasets["nq-distilbert-base-v1"] = "nq"

        # ToDo: deal with DPR model
        source_datasets["facebook-dpr-ctx_encoder-multiset-base"] = "MANY"
        names_with_cos = [# "nq-distilbert-base-v1",
                          "msmarco-MiniLM-L-6-v3",
                          "msmarco-MiniLM-L-12-v3",
                          "msmarco-distilbert-base-v3",
                          # "BAAI-bge-large-zh-v1.5",

            ]
        for name in names_with_cos:
            score_function[name] = "cos_sim"
        score_function["dpr"] = "dot"
        self.score_function = score_function
        self.source_datasets = source_datasets

    def download_models(self):
        names = [#"facebook-dpr-question_encoder-multiset-base",
                 #"facebook-dpr-ctx_encoder-multiset-base",
                 #"nq-distilbert-base-v1",
                 "msmarco-MiniLM-L-6-v3",
                 "msmarco-MiniLM-L-12-v3",
                 "msmarco-distilbert-base-v2",
                 "msmarco-distilbert-base-dot-prod-v3",
                 "msmarco-distilbert-base-v3",
                 "msmarco-roberta-base-ance-firstp",
                 "msmarco-distilbert-base-tas-b",
                 #"BAAI-bge-large-zh-v1.5"

                 ]
        for name in names:
            print("Downloading", name)
            SentenceTransformer(model_name_or_path=name,
                                cache_folder=self.model_dir)

    def load_model(self,  model_name, model_name_or_path=None):
        # ("model_name_or_path", model_name_or_path)
        #print("Self names", self.names)
        # if model_name_or_path is None:
        #     model_name = "sentence-transformers_" + model_name
        #     model_dir = os.path.join(self.model_dir, model_name)
        #     model = models.SentenceBERT(model_dir)
        # else:
        #     model_dir = model_name_or_path
        #     model = models.SentenceBERT(model_dir)
        
        model_name = "sentence-transformers_" + model_name
        model_dir = os.path.join(self.model_dir, model_name)
        model = models.SentenceBERT(model_dir)
        model.config = AutoConfig.from_pretrained(model_dir)
        model.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        return model


