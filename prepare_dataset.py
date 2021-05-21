import pandas as pd
from utils import clases
documentos = clases.Documento.read_csv(
    'dataset/fraude_TEXTO_TODOS.csv', sep='|')
documentos[0]

df = pd.read_csv('dataset/fraude_TEXTO_TODOS.csv', sep='|')
df = df.drop(columns='Unnamed: 0')

d = clases.Documento(df['id'], df['text'][4])
d.clean_text()
d.texto[:50]
s = d.get_sentencia()
s.set_grama(5)
s.get_gramas()
