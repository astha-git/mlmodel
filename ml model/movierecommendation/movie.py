import streamlit as st
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
import difflib
from sklearn.metrics.pairwise import cosine_similarity
st.title("Movie Reccomendation System")
moviesdata=pd.read_csv('movies.csv')
vectorizer=CountVectorizer()
moname = moviesdata['title'].values
newmoviesdata=moviesdata[["genres","keywords","original_title","title","tagline","cast","director","index"]]
combinedfeatures = newmoviesdata['genres']+' '+newmoviesdata['keywords']+' '+newmoviesdata['tagline']+' '+newmoviesdata['cast']+' '+newmoviesdata['director']+' '+newmoviesdata['original_title']+' '+newmoviesdata['title']
feature=vectorizer.fit_transform(combinedfeatures.values.astype('U'))
similarity= cosine_similarity(feature)
def Reccomendationsystem(movie):
    recomendedmovie=[]
    list_of_all_titles = newmoviesdata['title'].tolist()
    find_close_match = difflib.get_close_matches(movie, list_of_all_titles)
    close_match = find_close_match[0]
    index = moviesdata[newmoviesdata.title == close_match]['index'].values[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[0:10]:
      recomendedmovie.append(newmoviesdata.iloc[i[0]].title)
    return recomendedmovie
selectedmovie = st.selectbox("What next you want :?",moname)
if st.button("Recommend Next Movie"):
    recomendedmovie = Reccomendationsystem(selectedmovie)
    st.text(recomendedmovie[1])
    st.text(recomendedmovie[2])
    st.text(recomendedmovie[3])
    st.text(recomendedmovie[4])
    st.text(recomendedmovie[5])
    st.text(recomendedmovie[6])
    st.text(recomendedmovie[7])
    st.text(recomendedmovie[8])
    st.text(recomendedmovie[9])

    
 