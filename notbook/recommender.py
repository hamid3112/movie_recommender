#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import json
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    path = r"C:\Users\SEYED\AI\project\movie\data_set\movies_metadata.csv"
    movie_dataset = pd.read_csv(path , low_memory=False)
    movie_dataset = movie_dataset.head(5000)
    movie_dataset["title"] = movie_dataset["title"].apply(lambda x : x.lower())
    movie_dataset = movie_dataset[["title" , "genres" , "overview" , "tagline"]]
    movie_dataset = movie_dataset.dropna(subset = "title").reset_index(drop=True)
    print("داده ها با موفقیت بارگذاری شدند و آماده پردازش هستند")
except FileNotFoundError:
    print("خطا: فایل movies_metadata.csv پیدا نشد. لطفاً مطمئن شوید که فایل در پوشه صحیح قرار دارد.")
    exit()
def convert_json_to_string(obj):
        if isinstance(obj , str):
            try:
               return " ".join([i["name"] for i in ast.literal_eval(obj)])
            except (ValueError , SyntaxError):
                return ""
        return ""
movie_dataset["genres"] = movie_dataset["genres"].apply(convert_json_to_string)
movie_dataset["tagline"] = movie_dataset["tagline"].fillna("")
movie_dataset["overview"] = movie_dataset["overview"].fillna("")
def create_soup(row):
    return f'{row["genres"]} {row["overview"]} {row["tagline"]}'
movie_dataset["featurs"]= movie_dataset.apply(create_soup , axis = 1)
print("\nداده های بعد از پیش پردازش آماده هستند")
#vectorize
tfidfvectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidfvectorizer.fit_transform(movie_dataset["featurs"])
print("\nتبدیل متن به بردار با موفقیت انجام شد")
print("ابعاد ماتریس TF-DF :" , tfidf_matrix.shape)
#similarity
cos_sim = cosine_similarity(tfidf_matrix , tfidf_matrix)
print("\nمحاسبه ماتریس شباهت با موفقیت انجام شد")
#ساخت تابع توصیه گر
indices = pd.Series(movie_dataset.index , index = movie_dataset["title"]).drop_duplicates()
def get_recommendatin(title , cos_sim = cos_sim):
    title = title.lower()
    try :
        idx = indices[title]
        sim_score = list (enumerate(cos_sim[idx]))
        sim_score = sorted(sim_score , key = lambda x : x[1] , reverse=True)
        sim_score = sim_score[1:11]
        movie_indices = [i[0]for i in sim_score]
        return movie_dataset["title"].iloc[movie_indices].tolist()
    except KeyError:
        return [f"{title} در لیست فیلم ها موجود نیست!"]
#exm : toy story
print("فیلم های مشابه با toy story :")
for i,j in zip(get_recommendatin("toy story"), range(len(get_recommendatin("toy story")))):
    print(j+1,"-",i)

