from cgi import print_environ
from textwrap import indent
from read_files import *
from utils import Utils
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim
import numpy as np
import warnings
import pyLDAvis
import pyLDAvis.gensim_models


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

def run(company,min_topics,max_topics):
    print("="*32)
    print("Empresa analizada: "+company)
    print("="*32)

    data = get_data("./in/" + company +"/")
    new_stop = pd.read_csv('./in/stop_words.csv')
    stop_words = stopwords.words('english')
    stop_words.extend(new_stop['palabra'])

    print("="*32)
    print("Exportando nube de palabras ")
    print("="*32)


    wordcloud = utils.general_cloud(data,stop_words)
    utils.plot_cloud(wordcloud)
    plt.savefig('./out/' + company +  '/_wordcloud.png')
    plt.clf()
    print("="*32)
    print("Creando diccionarios")
    print("="*32)

    dictionary = gensim.corpora.Dictionary(data['clean_text'])
    print('Antes de filtrar')
    print(dictionary)
    dictionary.filter_extremes(no_below=5,no_above=0.8)
    print('Despues de filtrar')
    print(dictionary)

    print("="*32)
    print("Creando Corpus")
    print("="*32)

    corpus = [dictionary.doc2bow(text) for text in data['clean_text']]

    print("="*32)
    print("Encontrando el mejor modelo ")
    print("="*32)

    Resultados = utils.optimize_model(utils,data,corpus,dictionary,min_topics,max_topics)
    optim = Resultados[Resultados['Coherence'] == Resultados['Coherence'].max()]

 

    print("="*32)
    print("Estimando modelo con los parámetros encontrados ..")
    print("="*32)

    k = optim['Topics'].values[0]
    a = optim['Alpha'].values[0]
    b = optim['Beta'].values[0]

    Resultados2=Resultados[(Resultados["Alpha"]==a) & (Resultados["Beta"]==b)]
    coherence_values=Resultados2["Coherence"]
    x = Resultados2["Topics"]
    print(x)
    print(coherence_values)
    utils.plot_line(x,coherence_values)
    plt.savefig('./out/' + company +  '/_coherencia.png')
    plt.clf()

    Perplejidad_values=Resultados2["Perplejidad"]
    x = Resultados2["Topics"]
    utils.plot_line(x,Perplejidad_values)
    plt.savefig('./out/' + company +  '/_perplejidad.png')
    plt.clf()

    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=k, 
                                            random_state=100,
                                            chunksize=100,
                                            passes=10,
                                            alpha=a,
                                            eta=b)

    print("="*32)
    print("Exportando LDAvis ")
    print("="*32)

    p = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(p, './out/'+company+'/lda.html')

    print("="*32)
    print("Características de los tópicos ")
    print("="*32)
    topicos = lda_model.print_topics(num_words=5, num_topics=k)
    for topico in topicos:
        print(topico)
    my_dict = {'Topic_' + str(i): [token for token, score in lda_model.show_topic(i, topn=100)] for i in range(0, lda_model.num_topics)}
    
    print("="*32)
    print("Exportando .csv con palabras por topicos")
    print("="*32)

    topics = pd.DataFrame.from_dict(my_dict)
    topics.to_csv("wordsxtopic.csv")
    
    for i in range(0, k):
        wordcloud = WordCloud(width= 3000, height = 2000, random_state=1, 
                background_color='salmon', colormap='Pastel1', 
                collocations=False, stopwords = stop_words).fit_words(dict(lda_model.show_topic(i, 30)))
        utils.plot_cloud(wordcloud)
        plt.savefig('./out/' + company +  '/' + 'topico_' + str(i) +'_wordcloud.png')

    print("="*32)
    print("Exportando .csv con topicos por documento ")
    print("="*32)

    df_topic_sents_keywords = utils.format_topics_sentences(lda_model,corpus,data["file"])
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Show
    df_dominant_topic.to_csv('./out/'+ company +'/' +'topicoxsxdoc.csv')

    print("="*32)
    print("Exportando .csv con documentos mas representativos por topico ")
    print("="*32)
    sent_topics_sorteddf_mallet = pd.DataFrame()

    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                                grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                                axis=0)
    # Reset Index    
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

    # Show
    sent_topics_sorteddf_mallet.to_csv('./out/'+ company +'/' +'documentos_representativos.csv')

    # Número de documentos para cada tópico
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
    # Percentage of Documents for Each Topic
    topic_contribution = round(topic_counts/topic_counts.sum(), 4)
    print("="*32)
    print("Distribución de los tópicos en los documentos ")
    print("="*32)
    print(topic_contribution)

if __name__ == '__main__':
    utils = Utils()
    companys = os.listdir("./in")

    
    print("="*32)
    print("como desea hacer el analisis: ")
    print("Ingrese 1. Para una empresa en especifico")
    print("Ingrese 2. Para todas las empresas")
    type_run = int(input())
    print("="*32)

        # Topics range
    min_topics = input("Ingrese el numero minimo de topicos a evaluar: ")
    max_topics = input("Ingrese el numero maximo de topocos a evaluar: ")
    if type_run == 1:
        company = input("ingrese el nombre de la empresa: ")
        run(company,min_topics,max_topics)
    elif type_run == 2:    
        for company in companys:
            if company.endswith(".csv") == False:
                run(company)
    else:
        print("Ingrese un valor valido")
        os._exit(os.EX_OK) 

