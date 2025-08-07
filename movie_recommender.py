#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import json
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
try:
    path = r"C:\Users\SEYED\AI\project\movie\data_set\movies_metadata.csv"
    movie_dataset = pd.read_csv(path , low_memory=False)
    movie_dataset = movie_dataset[["title" , "genres" , "overview" , "tagline"]]
    movie_dataset = movie_dataset.dropna(subset = "title").reset_index(drop=True)
    movie_dataset["title"] = movie_dataset["title"].apply(lambda x : x.lower())
    movie_dataset["genres"] = movie_dataset["genres"].apply(lambda x :" ".join([i["name"] for i in ast.literal_eval(x)]) if isinstance(x , str) else "")
    movie_dataset["tagline"] = movie_dataset["tagline"].fillna("")
    movie_dataset["overview"] = movie_dataset["overview"].fillna("")
    movie_dataset["featurs"] = movie_dataset.apply(lambda row :f'{row["genres"]} {row["overview"]} {row["tagline"]}' , axis = 1 )
    movie_dataset = movie_dataset.head(5000)
    tfidfvectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidfvectorizer.fit_transform(movie_dataset["featurs"])
    cos_sim = cosine_similarity(tfidf_matrix , tfidf_matrix)
    indices = pd.Series(movie_dataset.index , index = movie_dataset["title"]).drop_duplicates()
except FileNotFoundError:
    st.error("خطا: فایل movies_metadata.csv پیدا نشد. لطفاً مطمئن شوید که فایل در پوشه صحیح قرار دارد.")
    st.stop()
except MemoryError:
    st.error("خطا: حافظه کافی برای پردازش دیتاست وجود ندارد. لطفاً از یک زیرمجموعه کوچکتر استفاده کنید.")
    st.stop()


def get_recommendatin(title , cos_sim = cos_sim):
    title = title.lower()
    try :
        idx = indices[title]
        sim_score = list (enumerate(cos_sim[idx]))
        sim_score = sorted(sim_score , key = lambda x : x[1] , reverse=True)
        sim_score = sim_score[1:6]
        movie_indices = [i[0]for i in sim_score]
        return movie_dataset["title"].iloc[movie_indices].tolist()
    except KeyError:
        return f"{title} در لیست فیلم ها موجود نیست!"
st.title("چت‌بات پیشنهاد فیلم 🤖")
st.markdown("فیلم موردعلاقه ات رو به انگلیسی وارد کن ، اگه بتونم بهت 5 تا فیلم مشابه با امتیاز و موضوع نزدیک بهش بهت پیشنهاد میدم😍")
movie_title = st.text_input("اسم فیلم رو وارد کن:","Toy Story")
if st.button("شروع کنید🕵️"):
    with st.spinner("دارم میگردم..."):
        recommendations = get_recommendatin(movie_title)
    if isinstance (recommendations , list):
        st.success("اینارم ببینی خوبه 📹:")
        for movie in recommendations:
            st.write(f"▪️ {movie}")
    else:
        st.warning(recommendations)

