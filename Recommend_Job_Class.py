import spacy
import numpy as np
import pandas as pd
import re
from spacy.language import Language
from spacy.tokens import Doc
import h5py

@Language.factory('trf_vectors')
class TrfContextualVectors:
    def __init__(self, nlp: Language, name: str):
        self.name = name
        Doc.set_extension("trf_token_vecs", default=None)

    def __call__(self, doc):
        # inject hooks from this class into the pipeline
        if type(doc) == str:
            doc = self._nlp(doc)

        # pre-calculate all vectors for every token:
        # calculate groups for spacy token boundaries in the trf vectors
        vec_idx_splits = np.cumsum(doc._.trf_data.align.lengths)

        # get transformer vectors and reshape them into one large continous tensor
        trf_vecs = doc._.trf_data.tensors[0].reshape(-1, 768)
        # calculate mapping groups from spacy tokens to transformer vector indices
        vec_idxs = np.split(doc._.trf_data.align.dataXd, vec_idx_splits)

        # take sum of mapped transformer vector indices for spacy vectors
        vecs = np.stack([trf_vecs[idx].sum(0) for idx in vec_idxs[:-1]])
        doc._.trf_token_vecs = vecs

        doc.user_token_hooks["vector"] = self.vector
        doc.user_span_hooks["vector"] = self.span_vector
        doc.user_hooks["vector"] = self.doc_vector
        doc.user_token_hooks["has_vector"] = self.has_vector

        return doc

    def vector(self, token):
        return token.doc._.trf_token_vecs[token.i]
    
    def span_vector(self,span):
        return span.doc._.trf_token_vecs[span.start : span.end].sum(axis=0)
    
    def doc_vector(self,doc):
        return doc.doc._.trf_token_vecs[:].mean(axis=0)
    
    def has_vector(self, token):
        return True
    
class Recommend_Job():
    def __init__(self,user_skill):
        self.nlp = spacy.load("./IT_skill_NER/model-best",exclude="ner") # add
        self.nlp.add_pipe("trf_vectors",after="transformer")
        list_user_skill =  user_skill.split(",")
        list_user_skill = [" ".join(skill.split()) for skill in list_user_skill]
        self.emb_user_skill = np.array([self.nlp(skill+" IT").vector for skill in list_user_skill])
        df = pd.read_csv("./data/JDs_Eval.csv",sep=";")
        self.URL = list(df["URL"])
        self.name_job = list(df["nameJob"])

    def _similarity(self,emb1, emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def _avg_max_similar(self, emb_user_skill, emb_job_eval) -> float:
        average = 0
        for user_emb in emb_user_skill:
            average += np.max([self._similarity(user_emb, emb_job) for emb_job in emb_job_eval])
        return average/len(emb_user_skill)

    def top3_job(self):
        top_job = []
        with h5py.File("embedding_job_eval.h5", "r") as f:
            for i in range(144):
                emb_job_eval = f[f"JD{i}"][()]
                top_job.append((self._avg_max_similar(self.emb_user_skill,emb_job_eval),i))
        top_job = sorted(top_job, key=lambda a: a[0],reverse=True)[:3]
        print(top_job)
        return [(self.URL[index], self.name_job[index]) for _,index in top_job]  
    def result(self):
        strr = ""
        top3 = self.top3_job()  
        for i,(url,name) in enumerate(top3) :
            strr += f"JOB {i+1}: " + name +"\n"+ url +"\n"
        return strr

if __name__ == "__main__":
    list1 = "ML,Deep Learning"
    model = Recommend_Job(list1)
    print(model.result())


