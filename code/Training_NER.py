import pandas as pd
import spacy
import random
from tqdm import tqdm
from IT_Skill_NER_Class import IT_SKILL_NER_Class
from spacy.training import Example
from spacy.tokens import DocBin
from thinc.api import set_gpu_allocator, require_gpu, set_active_gpu

class Training_NER():
    def __init__(self):
        self.using_gpu()
        self.model = IT_SKILL_NER_Class()
        df = pd.read_csv("./data/JobDescription.csv", sep="\t")
        self.list_JDs = list(df["JD"])
        random.shuffle(self.list_JDs)
    def using_gpu(self):
        set_gpu_allocator("pytorch")
        gpu = require_gpu(0)
        set_active_gpu(0)
        print("Using GPU : ",gpu)

    def get_training_data(self,is_train = True):
        if is_train:
            list_JDs = self.list_JDs[:25500]
        else :
            list_JDs = self.list_JDs[25500:]
        
        DATA = list()
        for jd in tqdm(list_JDs):
            try:
                doc = self.model.nlp(jd)
                list_ents = self.model.get_list_ents(doc)
                DATA.append((doc.text, {"entities": list_ents}))
            except:
                print("Bug Data")
        return DATA
    
    def save_data_spacy_file(self,data,is_train = True):
        db = DocBin()
        for text, train_value in tqdm(data):
            try:
                doc = self.model.nlp.make_doc(text)
                example = Example.from_dict(doc, train_value)
                db.add(example.reference)
            except:
                print("BUG ENTS")
        if is_train:
            db.to_disk("./config/training_data/train.spacy") 
        else :
            db.to_disk("./config/training_data/dev.spacy") 
    
if __name__ == "__main__":
    training = Training_NER()
    print("GET TRAIN DATA")
    print("-"*15)
    train_data = training.get_training_data()
    training.save_data_spacy_file(train_data)

    print("GET DEV DATA")
    print("-"*15)
    dev_data = training.get_training_data(is_train=False)
    training.save_data_spacy_file(dev_data,is_train=False)