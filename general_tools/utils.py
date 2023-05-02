import spacy
from spacy import displacy
# make sure to run: spacy download en_core_web_lg
NER = spacy.load("en_core_web_lg")
categories = {}
i = 0
for label in NER.get_pipe("ner").labels:
    categories[label] = i
    i += 1

# Just return the ents for now (maybe add the types?)
def do_ner(text):
    ner_text = NER(text)
    for word in ner_text.ents:
        print(word.text, word.label_)

do_ner("Sample text NASA London Rome July 3rd some silly words that are nothing")