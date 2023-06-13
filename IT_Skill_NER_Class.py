import spacy
from spacy.tokens.token import Token
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
import re
import os
from tqdm.auto import tqdm
from pathlib import Path

class IT_SKILL_NER_Class():
    def __init__(self,text: str = ""):
        self.text = text
        self.nlp = self._initialize_spacy()
        self.doc = self.nlp(text)

        self._index_ex = ["Senior","Junior","Trainee","Internship","Fresher","Sr","Jr"]
        self._skill_phrase = ["software","algorithm","library", "model", "tool","module","platform","method","equipment","component","testing","engine",
                        "management", "development", "methodology", "certify", "certification","programming","analysis","application","analysis","research",
                        "technology", "technical", "technique", "language", "infrastructure", "design", "system","systems","network","networks","web","code","process"]
        self._word_ing = ["overseeing","developing","transforming","building","implementing","analyzing","executing","writing","deploying","troubleshooting",
                        "managing","configuring","designing","maintaining","evaluating","running","monitoring","solving","resolving","leveraging","standardizing",
                        "operating","establishing","integrating","optimizing","coding","learning","updating","securing","fixing","training","testing",
                        "utilizing","recommending","producing","defining","ng","mining","creating","supporting"]
        self._word_after_remove = ["year","years","experience","knowledge","excellent"]
        self._word_ing_remove = ["including","seeking","using","looking","following"]
   
    def _initialize_spacy(self):
        # data_file_path = os.path.join('..', folder, file)
        # with open(data_file_path, 'r') as f:
        #     data = f.read()
        
        nlp = spacy.load("en_core_web_trf")
        if "entity_ruler" in nlp.pipe_names:
            ruler = nlp.get_pipe("entity_ruler")
        else:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.from_disk("/home/vkuai/AI_Extract_IT_Skill/data/EntityRuler_Patterns.jsonl")
        return nlp
    
    # Func support extract skill
    def _ing_rules(self,doc:Doc, idx: int) -> int:
        """
        Input:  doc: Doc 
                token: Token -> token is the verb in front of the noun phrase, in word_ing list
        Return: Number of tokens that satisfy the verb rule ing suffix after the noun phrase
        """
        t = 1
        if (idx-t) < 0:
            return 0
        while True :
            if (idx-t) > 0 and re.search("ing$",doc[idx-t].text) and doc[idx-t].text not in self._word_ing_remove: 
                t += 1
            elif (idx-t-1) > 0 and doc[idx-t].text in ["and","or",",","/","&","and/or"] and re.search("ing$",doc[idx-t-1].text):
                t += 2
            elif (idx-t-2) > 0 and doc[idx-t].text in ["and","or"] and doc[idx-t-1].text == "," and re.search("ing$",doc[idx-t-2].text):
                t += 3
            else : 
                return t
            
    def _check_token_to_add(self,token: Token, 
                         token_in_noun_phrase: list) -> bool:
        if token.pos_ in ["NOUN","PROPN"] :
            return True
        if token.ent_type_ == "HARD-SKILL" :
            return True
        if token.lemma_ in self._skill_phrase :
            return True
        if token.lower_ in self._word_ing :
            return True
        if token.text in self._index_ex :
            return True
        if token in token_in_noun_phrase:
            return True
        return False

    def _check_token_break(self,token: Token) -> bool:
        if token.pos_ in ["PRON","DET","ADV","CCONJ","NUM"]:
            return True
        elif (not re.search("[a-zA-Z]",token.text)) :
            return True
        elif token.lower_ in ["strong","existing","experience","knowledge","excellent","new","other","that","full","e.g.","good","necessary","and","or","&","year","years"]:
            return True
        return False
            
    def _rule_dash(self,doc:Doc, index : int)-> bool:
        if index>0 and index<len(doc) and doc[index].text == "-":
            if doc.text[doc[index].idx-1] != " " and doc.text[doc[index].idx+1] != " " :
                return True
        return False
    
    def _check_if_roman_numeral(self,numeral:str):
        numeral = {c for c in numeral}
        validRomanNumerals = {c for c in "XVI"}
        return not numeral - validRomanNumerals
    
    def _remove_unnecessary_word_in_chunk(self,doc: Doc, chunk: Span, word_ing: bool = True) -> Span:
        """
        Remove unneccessary words in chunk
        Words have pos ["PRON","DET","ADV","CCONJ","NUM"], without letters and some exception words

        Input:  chunk: Span -> 1 noun phrase in doc.noun_chunks
        Return: Span -> Span removed all unneccessary words
        """
        len_chunk = len(chunk)
        start_char = chunk.start_char
        pre = 0
        while True :
            if len_chunk == 0 :
                break
            elif self._check_token_break(chunk[pre]):
                if chunk[pre].i < (len(doc)-1) and doc.text[start_char + len(chunk[pre].text)] != " ":
                    start_char = start_char + len(chunk[pre].text)
                else:
                    start_char = start_char + len(chunk[pre].text) + 1
                pre += 1
                len_chunk -= 1 
            else:
                break
        span = doc.char_span(start_char,chunk.end_char)
        if word_ing :
            if (len_chunk == 1 and span[0].pos_ not in ["NOUN","PROPN"]) or len_chunk == 0 :
                return None
        else:
            if (len_chunk == 1 and (not re.search("A-Z",span.text))) or len_chunk == 0 :
                return None
        return span
    
    def _list_tokens_in_noun_chunks(self,doc: Doc)-> list:
        """
        Remove unnecessary words in each chunk
        Then, get all tokens in nouns_chunk

        Input:  doc: Doc
        Return: list -> list contain all tokens in nouns_chunk
        """
        token_list = []
        for chunk in doc.noun_chunks:
            span = self._remove_unnecessary_word_in_chunk(doc,chunk)
            if span != None :
                for token in span :
                    token_list.append(token)
        return token_list

    def _remove_element_duplicate(self,list_entities:list) -> list:
        # remove duplicate elements in list
        list_entities = list(set(list_entities))
        list_entities = sorted(list_entities, key=lambda a: a[0])

        # if len(list_ents)  <= 1 then no need to check
        if len(list_entities) > 1:
            list_remove_ent = []
            # iterate each element in array list_ents
            for tuple_ent in list_entities:
                # check each element in the array against all other elements in the array
                for check in list_entities:
                    if check != tuple_ent:
                        if tuple_ent[0] >= check[0] and tuple_ent[1] <= check[1]:
                            list_remove_ent.append(tuple_ent)
                            break
                    
            list_entities = [ent for ent in list_entities if ent not in list_remove_ent]

        if len(list_entities) > 1:
            list_remove_ent = []
            list_add_ent = []
            # iterate each element in array list_ents
            for tuple_ent in list_entities:
                # check each element in the array against all other elements in the array
                for check in list_entities:
                    if check != tuple_ent:
                        if tuple_ent[0] <= check[0] and check[0] <= tuple_ent[1] and tuple_ent[1] <= check[1]:
                            tuple_ent_new = (tuple_ent[0], check[1], "HARD-SKILL")
                            list_add_ent.append(tuple_ent_new)
                            list_remove_ent.append(tuple_ent)
                            list_remove_ent.append(check)

            list_entities = [ent for ent in list_entities if ent not in list_remove_ent]
            for ent in list_add_ent:
                list_entities.append(ent)
        return list_entities
        
    def _add_other_ents(self, doc:Doc, list_entities:list, is_comma_rule:bool) -> list:
        list_label = ["ORG", "LANGUAGE", "GPE","TIME", "MONEY"]
        if is_comma_rule :
            list_label.append("HARD-SKILL")
        for ent in doc.ents:
            if ent.label_ in list_label:
                check = True
                for start_char, end_char, _ in list_entities:
                    for i in range(ent.start_char, ent.end_char):
                        if i >= start_char and i <= end_char:
                            check = False
                            break
                if check:
                    tuplee = (ent.start_char, ent.end_char, ent.label_)
                    list_entities.append(tuplee)
        list_entities = sorted(list_entities, key=lambda a: a[0])
        return list_entities

    def _repeat_pre_token(self,doc:Doc,token:Token,token_noun_chunks:list) -> int:
        i = 0
        if self._rule_dash(doc,index=token.i-1):
            i = 3
        elif self._rule_dash(doc,index=token.i):
            i = 2
        elif re.search("ing$",token.text) \
            and token.pos_ == "VERB" \
            and token.text not in self._word_ing_remove:
            i = self._ing_rules(doc = doc, idx = token.i)
        elif self._check_token_to_add(token,token_noun_chunks) \
            and token.lower_ not in self._word_after_remove:
            i = 1
        return i
    
    def _repeat_after_token(self,doc:Doc,token:Token,token_noun_chunks:list) -> int:
        i = 0
        if self._rule_dash(doc,index=token.i+1):
            i = 3
        elif self._rule_dash(doc,index=token.i):
            i = 2
        elif token.text == "(" \
            and re.search("[A-Z]",doc[token.i + 1].text) \
            and doc[token.i + 1].lower_ not in ["required","preferred"] \
            and doc[token.i + 2].text == ")" :
            i = 3
        elif self._check_if_roman_numeral(token.text):
            i = 1
        elif (token.like_num and (not re.search("[a-zA-Z]",token.text))):
            if (token.i+1)<len(doc) and doc[token.i+1].text == "+" :
                i = 2
            else:
                i = 1
        elif self._check_token_to_add(token,token_noun_chunks)\
            and token.lower_ not in self._word_after_remove:
            i = 1
        return i

    def get_list_ents(self,doc):
        list_entities = []

        # rule _ing (*)
        for chunk in doc.noun_chunks:
            span = None
            if doc[chunk[0].i-1].lower_ in self._word_ing :
                span = self._remove_unnecessary_word_in_chunk(doc,chunk,word_ing=True)
                if span != None :
                    t = self._ing_rules(doc = doc, idx = chunk[0].i-1)
                    span = doc.char_span(doc[chunk[0].i-t].idx, chunk.end_char)
            elif doc[chunk[0].i-1].lower_ in self._word_ing_remove:
                span = self._remove_unnecessary_word_in_chunk(doc,chunk,word_ing=False)
            if span != None :
                tuplee = (span.start_char, span.end_char, "HARD-SKILL")
                list_entities.append(tuplee)
        
        token_noun_chunks = self._list_tokens_in_noun_chunks(doc)
        for token in doc:
            if  token.ent_type_ == "HARD-SKILL" \
                or token.lemma_.lower() in self._skill_phrase \
                or token.text in self._index_ex : 
                pre = 1
                after = 1
                while True:
                    if token.i - pre < 0:
                        break
                    number = self._repeat_pre_token(doc,doc[token.i - pre],token_noun_chunks)
                    if number != 0 and doc[token.i - pre].text != ".":
                        pre += number
                    else:
                        break
                while True:
                    if token.i + after >= len(doc):
                        break
                    number = self._repeat_after_token(doc,doc[token.i + after],token_noun_chunks)
                    if number != 0 and doc[token.i + after].text != ".":
                        after += number
                    else:
                        break
                    
                span = Span(doc, token.i - pre + 1, token.i + after, label="HARD-SKILL")
                if len(span) == 1 and span[0].ent_type_ != "HARD-SKILL":
                    pass
                else:
                    tuplee = (span.start_char, span.end_char, "HARD-SKILL")
                    list_entities.append(tuplee)
                
        list_entities = self._remove_element_duplicate(list_entities)
        list_entities = self._add_other_ents(doc,list_entities,is_comma_rule=False)
        return list_entities
    
    def _append_ents_into_doc(self,doc:Doc,list_ents:list):
        ents_new = []
        for start,end,label in list_ents:
            span = doc.char_span(start,end,label)
            ents_new.append(span)
        if len(ents_new)>0:
            doc.ents = ents_new
        else:
            doc.ents = []
        return doc

    def _comma_rule_token_len_1_2(self,doc:Doc, list_entities:list,count:int, start:int, end:int)-> list:
        is_break = False
        if start == end :
            return list_entities,is_break
        if count == 1 and doc[start].lower_ not in self._word_ing_remove:
            span = Span(doc,start, end, label="HARD-SKILL")
            if re.search("A-Z",span.text):
                tuplee = (span.start_char,span.end_char,"HARD-SKILL")
                list_entities.append(tuplee)
        elif count == 2:
            if doc[start].lower_ not in self._word_ing_remove:
                span = Span(doc,start, end, label="HARD-SKILL")
                tuplee = (span.start_char,span.end_char,"HARD-SKILL")
                list_entities.append(tuplee)
            else :
                if start+1 != end:
                    span = Span(doc,start+1, end, label="HARD-SKILL")
                    tuplee = (span.start_char,span.end_char,"HARD-SKILL")
                    list_entities.append(tuplee)
                is_break = True
        return list_entities,is_break

    def _comma_rule(self,doc:Doc):
        doc = self._append_ents_into_doc(doc,self.get_list_ents(doc))
        word = ["and","or",",","/","&","and/or"]
        token_noun_chunks = self._list_tokens_in_noun_chunks(doc)
        list_entities = [] 
        for ent in doc.ents :
            if ent.label_ == "HARD-SKILL":
                if doc[ent.start-1].text in word:
                    count = 0
                    step_pre = 0
                    step = 0
                    check_exit = False
                    while True :
                        if ent.start-step >= 0 :
                            break
                        if doc[ent.start-step].text in word or count == 8 or doc[ent.start-step].text == ".":
                            if count != 0 :
                                start = ent.start-step+1
                                end = ent.start-step_pre
                                if count <= 2 :
                                    list_entities,is_break = self._comma_rule_token_len_1_2(doc,list_entities,count,start,end)
                                    if is_break:
                                        break
                                else :
                                    i = end - 1
                                    while i >= start :
                                        number = self._repeat_pre_token(doc,doc[i],token_noun_chunks)
                                        if number != 0:
                                            i -= number
                                        else :
                                            if (i+1) != end :
                                                span = Span(doc,(i+1), end, label="HARD-SKILL")
                                                if len(span) == 1 :
                                                    if re.search("A-Z",span.text):
                                                        tuplee = (span.start_char,span.end_char,"HARD-SKILL")
                                                        list_entities.append(tuplee)
                                                else:
                                                    tuplee = (span.start_char,span.end_char,"HARD-SKILL")
                                                    list_entities.append(tuplee)
                                            check_exit = True
                                            break 
                                if count == 8 or doc[ent.start-step].text == "." or check_exit:
                                    break   
                            step_pre = step
                            count = 0
                        else :
                            count += 1
                        step += 1
                        
                if doc[ent.end].text in word:
                    count = 0
                    step_pre = 0
                    step = 0
                    check_exit = False
                    while True :
                        if ent.end+step < len(doc) :
                            break
                        if doc[ent.end+step].text in word or count == 8 or doc[ent.end+step].text == ".":
                            if count != 0 :
                                start = ent.end+step_pre+1
                                end = ent.end+step
                                if count <= 2 :
                                    list_entities,is_break = self._comma_rule_token_len_1_2(doc,list_entities,count,start,end)
                                    if is_break:
                                        break
                                else :
                                    i = start
                                    while i < end :
                                        number = self._repeat_after_token(doc,doc[i],token_noun_chunks)
                                        if number != 0 :
                                            i += number
                                        else:
                                            if i != start:
                                                span = Span(doc,start, i, label="HARD-SKILL")
                                                if len(span) == 1 :
                                                    if re.search("A-Z",span.text):
                                                        tuplee = (span.start_char,span.end_char,"HARD-SKILL")
                                                        list_entities.append(tuplee)
                                                else:
                                                    tuplee = (span.start_char,span.end_char,"HARD-SKILL")
                                                    list_entities.append(tuplee)
                                            check_exit = True
                                            break 
            
                                if count == 8 or doc[ent.end+step].text == "." or check_exit:
                                    break   
                            step_pre = step
                            count = 0
                        else :
                            count += 1
                        step += 1

        list_entities = self._remove_element_duplicate(list_entities)
        list_entities = self._add_other_ents(doc,list_entities,is_comma_rule=True)
        return list_entities

    def _get_hard_skill(self, doc:Doc):
        doc = self.get_doc_final(doc)
        return [ent.text for ent in doc.ents if ent.label_ == "HARD-SKILL"]
    
    def get_doc_final(self,doc:Doc):
       return self._append_ents_into_doc(doc,self._comma_rule(doc))
