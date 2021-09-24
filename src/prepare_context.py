from utils import clases
from nltk import FreqDist

docs = clases.Documento.read_csv('dataset/fraude_TEXTO_TODOS.csv')

vocab = []
for doc in docs:
    doc.clean_text()
    vocab += list(set(doc.get_vocab()))

vocab = list(set(vocab))

with open('pytorch/vocab.txt', 'r') as f:
    bert_vocab = f.read()

bert_vocab = bert_vocab.split('\n')

vocab = [i for i in vocab if i in bert_vocab]
vocab = sorted(vocab)
with open('dataset/fraude_vocab-bert_vocab.txt', 'w') as f:
    for i in vocab:
        f.write(i+'\n')
