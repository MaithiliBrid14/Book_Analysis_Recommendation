# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 21:59:55 2020

@author: Admin
"""

import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st


import os
import string
import datetime

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from PIL import Image
from plotly.subplots import make_subplots

import re
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from wordcloud import WordCloud,STOPWORDS



DATA_URL=("new_project_books2.csv")
DATA_URL1=("collaborative.csv")

@st.cache(persist=True)
def load_data(nrows):
    data=pd.read_csv(DATA_URL)     
    return data

@st.cache(persist=True)
def load_data1(nrows):
    data=pd.read_csv(DATA_URL1)     
    return data

books=load_data(10000)
original_data=books


users=load_data1(10000)
original_data1=users
        

content_data = books[['original_title','authors','average_rating', 'image_url']]
content_data = content_data.astype(str)

content_data['content'] = content_data['original_title'] + ' ' + content_data['authors'] + ' ' + content_data['average_rating']+ ' ' + content_data['image_url']

content_data = content_data.reset_index()
indices = pd.Series(content_data.index, index=content_data['original_title'])

#removing stopwords
tfidf = TfidfVectorizer(stop_words='english')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(content_data['authors'])

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(content_data['content'])

cosine_sim_content = cosine_similarity(count_matrix, count_matrix)

def get_recommendations(title, cosine_sim=cosine_sim_content):
    
    idx = indices[title]

    # Get the pairwsie similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim_content[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    return list(content_data['image_url'].iloc[book_indices])

        
#logo=Image.open("final_logo.jpg")
#st.image(logo, use_column_width=True)

st.sidebar.title("BOOK ANALYSIS AND RECOMMENDATION")
st.sidebar.header("")
st.sidebar.header("EDA")

select=st.sidebar.radio(
            'Choose', ['Book Distribution', 'Books', 'Authors','Author Performance','Users', 'Content based recommendation'])
    

if select=="Book Distribution":
    
    st.subheader("Distribution of Books across ratings, langugaes and over the years")
    from plotly.subplots import make_subplots
    
    cnt_srs = books["Ratings_Dist"].value_counts()
    cnt=cnt_srs.sort_index()
    
    cnt_srs1 = books["language_code"].value_counts()[:5]
    
    
    df_year=books[books['original_publication_year']>=1950]
    cnt_year = df_year["original_publication_year"].value_counts()
    cnt_y=cnt_year.sort_index()
    
    
    fig1 = make_subplots(rows=2, cols=2,specs=[[{}, {}],[{"colspan": 2}, None]],subplot_titles=("Rating Distribution","Language Distribution", "Books Published Over Years"))
    
    fig1.add_trace(
        go.Line(x=cnt.index, y=cnt.values),
        row=1, col=1
    )
    
    fig1.add_trace(
        go.Line(x=cnt_srs1.index, y=cnt_srs1.values),
        row=1, col=2
    )
    
    fig1.add_trace(go.Line(x=cnt_y.index, y=cnt_y.values,marker={'color': cnt_y.values, 
        'colorscale': 'speed'}), row=2,col=1 
        
    )
    
    
    fig1.update_layout(
     height=600, width=900,
     showlegend=False,
              #template="plotly_dark",
              font=dict(
                 family="Courier New, monospace",
                   size=18,
                 color="black"
                   )
              )
    
    st.plotly_chart(fig1)

    
elif select=="Books":
    select1=st.selectbox('Books',['10 Most Popular Books','10 Least Popular Books','10 Highest Rated Books','10 Least Rated Books'])
    Images_array=[]
    captions=[]
    if select1=="10 Most Popular Books":
        most_popular=books.sort_values("ratings_count", ascending=False)[:10]
        for img in most_popular['image_url']:
            #Images=Image.open(img)
            Images_array.append(img)
            #captions.append(cap)
        st.image(Images_array, width=120 )
        
    elif select1=="10 Least Popular Books":
        least_popular=books.sort_values("ratings_count", ascending=True)[:10]
        for img in least_popular['image_url']:
            #Images=Image.open(img)
            Images_array.append(img)
            #captions.append(cap)
        st.image(Images_array, width=120 )
        
    elif select1=="10 Highest Rated Books":
        high_rated_book = books.groupby('image_url')['average_rating'].mean().reset_index().sort_values('average_rating', ascending = False).head(10).set_index('image_url')
        high_rated_book.columns = ['count']
        high_rated_book = high_rated_book.sort_values('count')
        for img in high_rated_book.index:
            #Images=Image.open(img)
            Images_array.append(img)
            #captions.append(cap)
        st.image(Images_array, width=120 )
        
    elif select1=="10 Least Rated Books":
        low_rated_book = books.groupby('image_url')['average_rating'].mean().reset_index().sort_values('average_rating', ascending = True).head(10).set_index('image_url')
        low_rated_book.columns = ['count']
        low_rated_book = low_rated_book.sort_values('count')
        for img in low_rated_book.index:
            #Images=Image.open(img)
            Images_array.append(img)
            #captions.append(cap)
        st.image(Images_array, width=120 )
      

elif select=="Authors":
    select3=st.selectbox('Authors',['10 Authors With Most Books','10 Highest Rated Authors'])
    
    if select3=='10 Authors With Most Books':
        most_books = books.groupby('authors')['original_title'].count().reset_index().sort_values('original_title', ascending=False).head(10).set_index('authors')
        
        fig3= go.Figure(data=go.Scatter(
            x=most_books.index,
            y=most_books['original_title'],
            mode='markers',
            marker=dict(size=[100,90,80,70,60,50,40,30,20,10],
                color=[0, 1, 2, 3,4,5,6,7,8,9])
            ))

        fig3.update_layout(
            width=800,
            height=600,
            title="Top 10 Authors with most books",
            xaxis_title="",
        yaxis_title="No. of Books",
        font=dict(
        family="Courier New, monospace",
        size=20,
        color="Black"
        )
        )
        
        
        st.plotly_chart(fig3)
        
    else:
        high_rated_author = books[books['average_rating']>=4]
        high_rated_author = high_rated_author.groupby('authors')['average_rating'].mean().reset_index().sort_values('average_rating', ascending = False).head(10).set_index('authors')
        
        high_rated_author.columns = ['count']
        high_rated_author = high_rated_author.sort_values('count')
        
        fig3=go.Figure(data=go.Scatter(
            x=high_rated_author.index,
            y=high_rated_author['count'],
           
            mode='markers',
            marker=dict(size=[10,20,30,40,50,60,70,80,90,100],
                color=[0, 1, 2, 3,4,5,6,7,8,9])
            ))

        fig3.update_layout(
            width=800,
            height=500,
            title="Top 10 Highest Rated Authors",
            xaxis_title="",
        yaxis_title="No. of Books",
        font=dict(
        family="Courier New, monospace",
        size=18,
        color="Black"
        )
        )
        
        st.plotly_chart(fig3)
        
        
elif select=="Author Performance":
    select4=st.selectbox("Author's overall Performance", books['authors'].unique())
    df_auth=books[books['authors']==select4]
    df_auth_sort=df_auth.sort_values('original_publication_year', ascending=False)
    cnt=df_auth_sort.groupby('original_publication_year')['average_rating'].mean()
    cnt=cnt.reset_index()
    auth_sk=books[books['authors']==select4]
    auth_popular=auth_sk.sort_values('ratings_count', ascending=False)[:5]
        
    auth_high=auth_sk.sort_values('average_rating', ascending=False)[:5]
        
        #df_sk_years=auth_sk.original_publication_year.value_counts()[0:10].reset_index().rename(columns={'index':'original_publication_year','original_publication_year':'count'})
        
    auth_y=auth_sk['original_publication_year'].value_counts()
    auth_year=auth_y.sort_index()
        
    fig4 = make_subplots(rows=2, cols=2,specs=[[{}, {}],[{"colspan": 2}, None]],subplot_titles=("Most Reviewed Books","Highest Rated Books","Performance over the years"))
        
    fig4.add_trace(go.Bar(x=auth_popular['original_title'], y=auth_popular['ratings_count'],marker={'color': auth_popular['ratings_count'], 
        'colorscale': 'Viridis'}), row=1,col=1)
        
    fig4.add_trace(go.Bar(x=auth_high['original_title'], y=auth_high['average_rating'],marker={'color': auth_high['average_rating'], 
        'colorscale': 'Viridis'}), row=1,col=2)
        
    fig4.add_trace(go.Line(x=cnt.original_publication_year, y=cnt.average_rating,marker={'color': cnt.values, 
        'colorscale': 'speed'}), row=2,col=1 
            
    )
        
        
    fig4.update_layout(
       # height=550, width=900,
        #template="plotly_dark",
        showlegend=False,
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="black"
            )
        ) 
    
    fig4.update_xaxes(
        showticklabels=False
        )
    st.plotly_chart(fig4)


elif select=="Users":
    st.header("Users Overview")
    users_city=users["city"].value_counts()[0:5]  
    users_state=users["state"].value_counts()[0:5]   
    users_country=users["country"].value_counts()[0:5]
    cnt_age = users["Age_dist"].value_counts()
    cnt_a=cnt_age.sort_index()
    labels=users.Age_dist.unique()
    labels_c=users.city.unique()
    labels_s=users.state.unique()
    labels_cnt=users.country.unique()
    specs = [[{'type':'domain'}, {'type':'domain'}], [ {'type':'domain'},{'type':'domain'}]]

    select_user=st.selectbox('select',['Age Distribution','Top User Locations'])
    
    if(select_user=='Age Distribution'):
        fig5 = go.Figure(data=[go.Pie(labels=labels, values=cnt_a, hole=.3, textinfo='label+percent')])
    
    else:
        select_loc=st.radio('choose',['City','State','Country'])
        if select_loc=='City':
            fig5 = go.Figure(data=[go.Pie(labels=labels_c, values=users_city, hole=.3, textinfo='label+percent')])
        
        elif select_loc=='State':
            fig5 = go.Figure(data=[go.Pie(labels=labels_s, values=users_state, hole=.3, textinfo='label+percent')])  
    
        else:
            fig5 = go.Figure(data=[go.Pie(labels=labels_cnt, values=users_country, hole=.3)])
    
   
    
    fig5.update_traces(hoverinfo='label+percent',textposition='inside')
    fig5.update_layout(
        showlegend=True,
        height=500, width=800,
        uniformtext_minsize=12, uniformtext_mode='hide',
        #template="plotly_dark",
           font=dict(
              # family="Courier New, monospace",
              #size=20,
              color="black"
               )
           )
    
    fig5.update_xaxes(
        showticklabels=False
        )
       
    st.plotly_chart(fig5)
      

else: #select=='Content based recommendation': 
    select5=st.selectbox("Enter Books of your choice", books['original_title'].unique())
    Images_array=[]
    st.header("Content Based Recommendation")
    book_rec=get_recommendations(select5, cosine_sim_content)
    
    for book in book_rec:
        Images_array.append(book)
        
    st.image(Images_array, width=120)
    







       



