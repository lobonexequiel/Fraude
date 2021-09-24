from numpy import array
import umap.plot
import umap
from plotly import express as px
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import clean
import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification


tokenizer = BertTokenizer.from_pretrained(
    "pytorch/", do_lower_case=False)
model = BertForSequenceClassification.from_pretrained("pytorch/")
"""
e = model.eval()
text = "[CLS] Para solucionar los [MASK] de Chile, el presidente debe [MASK] de inmediato. [SEP]"
masked_indxs = (4,11)

tokens = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
tokens_tensor = torch.tensor([indexed_tokens])

predictions = model(tokens_tensor)[0]
predictions[0,4].shape
idxs = torch.argsort(predictions[0,4], descending=True)
tokenizer.convert_ids_to_tokens(idxs[:15])

for i,midx in enumerate(masked_indxs):
    idxs = torch.argsort(predictions[0,midx], descending=True)
    predicted_token = tokenizer.convert_ids_to_tokens(idxs[:5])
    print('MASK',i,':',predicted_token)


df = pd.read_csv('dataset/fraude_TEXTO_TODOS.csv', sep='|')
t = df.text[0]
t = clean.general(t)
tokens = tokenizer.tokenize(t[:100])
indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
tokens_tensor = torch.tensor([indexed_tokens])
labels = torch.tensor([30])  # .unsqueeze(0)  # Batch size 1

outputs = model(tokens_tensor)  # ,labels=labels)

loss = outputs.loss
logits = outputs.logits
idxs = torch.argsort(logits, descending=True)
logits
[tokenizer.convert_ids_to_tokens(i) for i in idxs]

from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings

sentence = Sentence('El pasto es verde.')

# use only last layers
embeddings = TransformerWordEmbeddings('dccuchile/bert-base-spanish-wwm-uncased')
embeddings.embed(sentence)

tokenizer = AutoTokenizer.from_pretrained(
    'dccuchile/bert-base-spanish-wwm-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    'dccuchile/bert-base-spanish-wwm-uncased')

classes = ["no es parafrase", "es parafrase"]

sequence_0 = 'la compañía hugging face esta situada en nueva york'
sequence_1 = 'las manzanas son malas para la salud'
sequence_2 = 'la compañía hugging face esta situada en manhattan'

# The tokekenizer will automatically add any model specific separators (i.e. <CLS> and <SEP>) and tokens to the sequence, as well as compute the attention masks.
paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")
not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")
paraphrase['input_ids'].shape
paraphrase_classification_logits = model(**paraphrase).logits
not_paraphrase_classification_logits = model(**not_paraphrase).logits

paraphrase_results = torch.softmax(
    paraphrase_classification_logits, dim=1).tolist()[0]
not_paraphrase_results = torch.softmax(
    not_paraphrase_classification_logits, dim=1).tolist()[0]


# Should be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")

# Should not be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")


df = pd.read_csv('../Iunigo/dataset/auto-clean.csv')
embedder = SentenceTransformer('dccuchile/bert-base-spanish-wwm-uncased')

t, r = [], []
df = df.dropna().sample(3000)
for d, res in zip(df['descripcion'].dropna(), df['Motivo de cierre']):
    if len(str(d).split()) > 0 and type(d) == str:
        t.append(str(d))
        r.append(res)

t_vec = embedder.encode(t)
t_vec.shape
pca = PCA(n_components=200)
pca.fit(t_vec)
pca.explained_variance_ratio_.sum()

t_vec100d = pca.transform(t_vec)

u = umap.UMAP(
    n_neighbors=len(set(df['Motivo de cierre'])),
    min_dist=0.0,
)
t_ = u.fit(t_vec100d)
t_vec2d = t_.transform(t_vec100d)


umap.plot.points(t_, labels=array(r))
plt.show()

x = [i[0] for i in t_vec2d]
y = [i[1] for i in t_vec2d]

fig = px.scatter(
    [[i, j] for i, j in zip(x, y)],
    x=0, y=1,
    color=r,

)

fig.show()
fig.write_html('../Iunigo/plots/BERT_base_all-desc_20210610.html')
"""
