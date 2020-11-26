# Social Networks Project.
# Ganesh , Shobit , Vanshal

#Libraries Used______________________________
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import math as math
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans
import time
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [14,14]
#_________________________________________________

# READING THE DATA_________________________________
def READ_DATA():
    video = pd.read_csv('netflix_titles.csv')

    # Converting the date_added into format using pandas "to_datetime" function.
    video["date_added"] = pd.to_datetime(video['date_added'])
    video['year'] = video['date_added'].dt.year
    video['month'] = video['date_added'].dt.month
    video['day'] = video['date_added'].dt.day
    # Loading up individual attributes about each movie / show.
    video['directors'] = video['director'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
    video['categories'] = video['listed_in'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
    video['actors'] = video['cast'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
    video['countries'] = video['country'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
    return video

video = READ_DATA()

#________________________________________________

# K_MEANS CLUSTERING_____________________________

start = time.time()
tags = video['description']
vector = TfidfVectorizer(max_df=0.4,   min_df=1, stop_words='english', lowercase=True, use_idf=True, norm=u'l2', smooth_idf=True)
tfidf = vector.fit_transform(tags)

# Clustering  Kmeans
k = 200
kmeans = MiniBatchKMeans(n_clusters=k)
kmeans.fit(tfidf)  # term frequency–inverse document frequency
c = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vector.get_feature_names()
request_transform = vector.transform(video['description'])
video['cluster'] = kmeans.predict(request_transform)
video['cluster'].value_counts().head()

# _________________________________________________________

#Similarity_checker________________________________________

def find_similar(tfidf_matrix, index, top_n=7):
    cosine_similarities = linear_kernel( tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [index for index in related_docs_indices][0:top_n]

# _________________________________________________________

#__________________________________________________________
G = nx.Graph(label="MOVIE")
start = time.time()
for i, rows in video.iterrows():
    
    G.add_node(rows['title'], key=rows['show_id'], label="MOVIE",mtype=rows['type'], rating=rows['rating'])
    for element in rows['actors']:
        G.add_node(element, label="PERSON")
        G.add_edge(rows['title'], element, label="ACTED_IN")
    for element in rows['countries']:
        G.add_node(element, label="COUNTRY")
        G.add_edge(rows['title'], element, label="COUNTRY_IN")
    for element in rows['categories']:
        G.add_node(element, label="CATEGORY")
        G.add_edge(rows['title'], element, label="CATEGORY_IN")
    for element in rows['directors']:
        G.add_node(element, label="PERSON")
        G.add_edge(rows['title'], element, label="DIRECTED_IN")
    indices = find_similar(tfidf, i, top_n=7) #description similary
    snode = "Sim("+rows['title'][:8].strip()+")"
    G.add_node(snode, label="SIMILAR")
    G.add_edge(rows['title'], snode, label="SIMILARITY")
    for element in indices:
        G.add_edge(snode, video['title'].loc[element], label="SIMILARITY") 

print(" finish -- {} seconds --".format(time.time() - start))

# ------------------------------------------------------------------------------------------------------------

def get_all_adj_nodes(list_in):
    sub_graph = set()
    for m in list_in:
        sub_graph.add(m)
        for e in G.neighbors(m):
            sub_graph.add(e)
    return list(sub_graph)


def draw_graph(sg):
    S = G.subgraph(sg)
    colors = []
    for e in S.nodes():
        if G.nodes[e]['label'] == "MOVIE":
            colors.append('blue')
        elif G.nodes[e]['label'] == "PERSON":
            colors.append('red')
        elif G.nodes[e]['label'] == "CATEGORY":
            colors.append('green')
        elif G.nodes[e]['label'] == "COUNTRY":
            colors.append('yellow')
        elif G.nodes[e]['label'] == "SIMILAR":
            colors.append('orange')
        elif G.nodes[e]['label'] == "CLUSTER":
            colors.append('orange')

    nx.draw(S, with_labels=True, font_weight='bold', node_color=colors)
    plt.show()

# _________________________________________________________________________________________________________


def get_recommendation(root):
    intersection_nodes = {}
    for e in G.neighbors(root):
        for e2 in G.neighbors(e):
            if e2 == root:
                continue
            if G.nodes[e2]['label'] == "MOVIE":
                commons = intersection_nodes.get(e2)
                if commons == None:
                    intersection_nodes.update({e2: [e]})
                else:
                    commons.append(e)
                    intersection_nodes.update({e2: commons})
    movies = []
    weight = []
    
    # Adamic measure
    for key, values in intersection_nodes.items():
        w = 0.0
        for e in values:
            w = w+1/math.log(G.degree(e))
        movies.append(key)
        weight.append(w)

    result = pd.Series(data=np.array(weight), index=movies)
    result.sort_values(inplace=True, ascending=False)

    return result

# __________________________________________________________________________________________________________


def show(str):
    movie = get_recommendation(str)
    print("="*70+"\n Recommendation Shows for : " + str + "\n"+"="*70)
    print(movie.head(15))
    print("="*70)
nx.write_graphml(G , "kapa.gml")