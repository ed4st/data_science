from os import walk
from my_functions import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

def reconocer_texto(query = ""):
    '''
    Reconoce e imprime los 5 textos más parecidos al query.
    '''
    #creando un corpus base:
    mypath = "all_files/"
    _, _, filenames = next(walk(mypath))
    
    archivos= [open(mypath+filename, "r", encoding="latin-1") for filename in filenames]
    corpus = [archivo.read() for archivo in archivos]
    corpus.append(query) #añadimos el texto de consulta al vocabulario
    
    #preprocesamos los textos en conjunto
    preprocesador = preprocesaTexto(idioma='es', _tokeniza=False, _muestraCambios=False, _quitarAcentos=True, _quitarNumeros=True,
                                _remueveStop=True, _stemming=False, _lematiza=True, _removerPuntuacion=True)
    corpus_prep = [preprocesador.preprocesa(txt) for txt in corpus]
    
    # creamos la bolsa de palabras (BOW) y vectorizamos
    vectorizer = CountVectorizer(lowercase=True, ngram_range= (1,1), binary=False, max_features=10000)
    X = vectorizer.fit_transform(corpus_prep)
    bow = X.toarray()
    bow_df = pd.DataFrame(bow,columns=vectorizer.get_feature_names())
    
    #seleccionamos las palabras que se repiten por lo menos 5 veces, con el fin de reducir dimensionalidad:
    frecuencias_por_palabra = bow_df.sum(axis=0)
    palabras_relevantes = list(frecuencias_por_palabra[frecuencias_por_palabra>5].index)
    
    #redefinimos la bolsa de palabras con la dimensión reducida:
    # redefinimos la BOW
    bow = bow_df[palabras_relevantes]
    
    #calculamos las disimilaridades con la distancia coseno:
    cos_sim = cosine_similarity(bow, bow)
    sim_df = pd.DataFrame(cos_sim)

    #tomamos los 5 textos que mayor similaridad tienen con el query en la entrada 401
    similaridades_query = np.array(sim_df.iloc[400])
    
    top6 = similaridades_query.argsort()[-6:][::-1] #top 6 textos más parecidos
    top5 = top6[1:] #tomamos los índices de los textos más parecidos quitando al mismo texto
    
    #imprimimos los textos
    print(f"------------Texto original-------------\n {query}")
    i = 1
    for rank_ix in top5:
        print(f" --------------top {i} texto más parecido al query-------------\n{corpus[rank_ix]}")
        i += 1

#probamos el método:
if __name__ == '__main__':
    query_ejemplo = "Los saltos con breves reflexiones, de lectura ágil y sin problemas de comprensión de una historia que se cuenta hacia atrás y en pequeñas dosis. Al libro le sobran páginas, divagaciones que no aportan demasiado, sin embargo me gustó bastante"
    reconocer_texto(query_ejemplo)
    
    #Ahora probemos con un review de la mejor banda de rock de la historia:
    query_led_zeppelin = "El compendio del clima musical creado por Led Zeppelin se encuentra en dos de los temas más destacados del álbum: el hard rock con elementos psicodélicos de “Dazed And Confused”, canción original de Jake Holmes; y la infravalorada “Your Time Is Gonna Come”, elegante pieza pop-rock con un excelente trabajo en los teclados de John Paul Jones y un contagioso estribillo."
    reconocer_texto(query_led_zeppelin)