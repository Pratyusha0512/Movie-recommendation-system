#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")
movies.head()


# In[2]:


credits.head()


# In[3]:


movies=movies.merge(credits,on='title')
movies.info()


# In[4]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.info()


# In[5]:


movies.isnull().sum()


# In[6]:


movies=movies.dropna()


# In[7]:


movies.duplicated().sum()


# In[8]:


movies.iloc[0].genres


# In[9]:


import ast
def convert(x):
    l=[]
    for i in ast.literal_eval(x):
        l.append(i['name'])
    return l
        


# In[10]:


movies['genres']=movies['genres'].apply(convert)


# In[11]:


movies['keywords']=movies['keywords'].apply(convert)


# In[12]:


movies.head()


# In[13]:


movies['cast'][0]


# In[14]:


def convert2(x):
    l2=[]
    count=0
    for i in ast.literal_eval(x):
        if(count!=5):
            l2.append(i['name'])
            count+=1
        else:
            break
    return l2


# In[15]:


movies['cast']=movies['cast'].apply(convert2)


# In[16]:


movies['cast']


# In[17]:


movies['crew'][0]


# In[18]:


def convert3(x):
    l3=[]
    for i in ast.literal_eval(x):
        if(i['job']=='Director'):
            l3.append(i['name'])
    return l3


# In[19]:


movies['crew']=movies['crew'].apply(convert3)


# In[20]:


movies


# In[21]:


#ds['overview']=ds['overview'].apply(lambda a:a.split())
movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[22]:


movies


# In[23]:


movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])


# In[24]:


movies


# In[25]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
movies['tags']


# In[26]:


new_df=movies[['movie_id','title','tags']]
new_df


# In[27]:


new_df['tags']


# In[28]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[29]:


import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[30]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[31]:


new_df['tags'].apply(stem)


# In[32]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[33]:


vectors=cv.fit_transform(new_df['tags']).toarray()
vectors


# In[34]:


vectors[0]


# In[35]:


cv.get_feature_names_out()


# In[36]:


from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)
similarity[0]


# In[57]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title) 


# In[58]:


recommend('Tangled')


# In[ ]:




