from spacy.language import Language
from spacy.tokens import Doc
import numpy as np
import spacy
import re
import pandas as pd
import h5py
from spacy.lang.en.stop_words import STOP_WORDS
from IT_Skill_NER_Class import IT_SKILL_NER_Class

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
    
class ExtractSkillJDsEval():
    def __init__(self):
        self.nlp = spacy.load("/home/vkuai/AI_Extract_IT_Skill/IT_skill_NER/model-best")
        self.nlp.add_pipe("trf_vectors",after="transformer")
        df = pd.read_csv("/home/vkuai/AI_Extract_IT_Skill/data/JDs_Eval.csv", sep=";")
        self.list_JDs = list(df["nameJob"]+". "+df["JDs"]+".")
        self.model = IT_SKILL_NER_Class()

    def get_skill(self):
        list_skills = []
        for jd in self.list_JDs:
            doc = self.model.nlp(jd)
            skill = self.model._get_hard_skill(doc)
            list_skills.append(skill)
        return self.remove_stop_word(list_skills)
    
    def remove_stop_word(self,list_skills):
        list_skills_new = []
        for skill_each_job in list_skills:
            list_skills_each_job_new = []
            for skill in skill_each_job:
                doc = self.nlp(skill)
                for token in doc:
                    if token.is_stop or token.text in ["/","(",")",",",";","&"]:
                        skill = skill.replace(token.text , " ")
                        skill = " ".join(skill.split())
                skill = skill + " IT"
                list_skills_each_job_new.append(skill)
            list_skills_new.append(list_skills_each_job_new)
        return list_skills_new
    
    def embedding_token(self):
        list_skill = self.get_skill()
        with h5py.File("embedding_job_eval.h5", "w") as f:
            for i,skills_each_job in enumerate(list_skill):
                list_embedding_each_job = []
                for skill in skills_each_job:
                    doc = self.nlp(skill)
                    list_embedding_each_job.append(doc.vector)
                list_embedding_each_job = np.array(list_embedding_each_job)
                print(list_embedding_each_job.shape)
                f.create_dataset(f"JD{i}", data=list_embedding_each_job)
        f.close()

a = ExtractSkillJDsEval()
a.embedding_token() 
