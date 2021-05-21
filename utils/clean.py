import nltk
import re
import spacy

stops_nltk = nltk.corpus.stopwords.words('spanish')
# DESCOMENTAR Y COMPLETAR
# stops_spacy = spacy.
stops = stops_nltk  # + stops_spacy


def general(txt):
    """
        elimina caracteres no deseados
        w = texto tipo string
    """
    txt = txt.translate(str.maketrans(
        'áéíóúýàèìòùÁÉÍÓÚÀÈÌÒÙÝ', 'aeiouyaeiouAEIOUAEIOUY'))
    txt = txt.lower()
    # txt = txt.replace('\r', ' ').replace("\v", ' ').replace(
    #    "\t", ' ').replace("\f", ' ').replace("\a", ' ').replace("\b", ' ')
    txt = re.sub(r'[^\w\s]', ' ', txt)
    txt = re.sub(r'\d+', ' ', txt)
    txt = re.sub(' +', ' ', txt)
    txt = txt.strip()
    return txt


def deleteRepeated(row):
    row = row.split()
    i = 0
    while i < len(row) - 1:
        if row[i] == row[i + 1]:
            del row[i]
        i += 1
    return ' '.join(row)


def remove_stops(texto):
    texto = [
        i for i in texto.split() if i not in stops
    ]
    return ' '.join(texto)
