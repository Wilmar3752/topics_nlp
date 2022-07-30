import os
import re
from nltk.corpus import stopwords
import pandas as pd
#import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pdfminer.high_level import extract_text
from deep_translator import GoogleTranslator


def get_pdf(file):
    pdf = extract_text(file)
    text = re.sub('http\S+', ' ', pdf)
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~\\“\\º]'
    text = re.sub(regex , ' ', text)
    # Eliminación de números
    text = re.sub("\d+", ' ', text)
    # Eliminación de espacios en blanco múltiples
    text = re.sub("\\s+", ' ', text)
    return text
def get_txt(file):
    with open(file) as f:
        texto = f.read()
    return texto

def text_to_eng(pdf):
    traductor = GoogleTranslator(source='es', target='en')
    n=4999 ## solo traduce 5000 caracteres
    split = [traductor.translate(pdf[i:i+n]) for i in range(0, len(pdf), n)]
    return " ".join(split)

def clean_tokenize(texto):
    '''
    Esta función limpia y tokeniza el texto en palabras individuales.
    El orden en el que se va limpiando el texto no es arbitrario.
    El listado de signos de puntuación se ha obtenido de: print(string.punctuation)
    y re.escape(string.punctuation). La funcion tambien elimina stop words y realiza una
    lematizacion de palablas en español
    '''
    new_stop = pd.read_csv('./in/stop_words.csv')
    # lista de stop words en español
    stop_words = stopwords.words('english')
    stop_words.extend(new_stop['palabra'])
    # Cargando nlp para lematizar
    wnl = WordNetLemmatizer()
    # Se convierte todo el texto a minúsculas
    text = texto.lower()
    # Eliminación de páginas web (palabras que empiezan por "http")
    text = re.sub('http\S+', ' ', text)
    # Eliminación de signos de puntuación
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    text = re.sub(regex , ' ', text)
    # Eliminación de números
    text = re.sub("\d+", ' ', text)
    # Eliminación de espacios en blanco múltiples
    text = re.sub("\\s+", ' ', text)
    # Tokenización por palabras individuales
    #text = text.split(sep = ' ')
    text = word_tokenize(text)
    # Eliminación de tokens con una longitud < 2
    text = [wnl.lemmatize(token) for token in text if len(token) > 1 if not str(token) in stop_words]
    
    return(text)


def get_data(path) -> pd.DataFrame:
    dir_list = os.listdir(path)
    salida = []
    for dir in dir_list:
        print("Leyendo documentos en: "+dir)
        print("="*32)
        files_dir = os.listdir(path+dir+"/")
        for file in files_dir:
            if file.endswith(".pdf")==True:
                print("procesando " + file)
                texto = get_pdf(path+dir+"/"+file)
            elif file.endswith(".txt")==True:
                print("procesando " + file)
                texto = get_txt(path+dir+"/"+file)
            
            if dir == "espanol":
                print('Traduciendo...')
                texto = text_to_eng(texto)
            else: texto 
            salida.append([texto,dir,file])
    df = pd.DataFrame(salida,columns = ['document','idioma','file'] )
    df['clean_text'] = df['document'].apply(clean_tokenize)
    return df
