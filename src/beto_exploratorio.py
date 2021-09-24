# from importlib.machinery import SourceFileLoader
# clases = SourceFileLoader('clases','../utils/clases.py').load_module()
from utils import clases

import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.visual.ner_html import render_ner_html

df = pd.read_csv('dataset/fraude_TEXTO_TODOS.csv', sep='|')

doc = clases.Documento(df['id'][0], df['text'][0])

# make a sentence
sentence = Sentence(doc.texto, use_tokenizer=True)

# load the NER tagger
tagger = SequenceTagger.load('es-ner-large')
​
# run NER over sentence
tagger.predict(sentence)
print(tagger.predict(sentence))
# Print results
pprint(sentence.to_dict(tag_type='ner'))
with open('out.txt', 'w') as f:
    f.write(str(sentence.to_dict()))
​
sentence_dict = sentence.to_dict(tag_type='ner')
print(len(sentence_dict))
entidades = sentence_dict['entities']
print(entidades)
with open('entidades.txt', 'w') as falias:
    falias.write(str(entidades))
for i in range(len(entidades)):
    if entidades[i]['labels'][0].value == 'PER':
        print(entidades[i]['text'])
        print(entidades[i]['labels'][0].value)
​
print(sentence.to_tagged_string())
with open('ner_html.html', 'w') as alias:
    alias.write(render_ner_html(sentence))
