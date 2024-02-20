# 🗞️ News Classifier
#Building an Automated News Classification System with NLP Techniques
#----------------------------------------------------------------------------------------------------------
#Import necessary libraries
#----------------------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import re
import emoji
import spacy
#----------------------------------------------------------------------------------------------------------
#Word2Vec model
from gensim.models import Word2Vec
w2v_model=Word2Vec.load(r"C:\Users\banup\Desktop\N\Trained models\word2vec_model.bin")
#------------------------------------------------
import pickle
#SVM classifier
with open(r"C:\Users\banup\Desktop\N\Trained models\svm_model.pkl", "rb") as f:
    svm_classifier_model = pickle.load(f)
#--------------------------------------------------------------------------------------------------
#Streamlit environment
#Page Layout
st.set_page_config (
                    page_title="News Classifier by Banuprakash",
                    page_icon= "🗞️",  
                    layout="wide", 
                    initial_sidebar_state="expanded",  
                   )
#-----------------------------------------------------------------------------------------------------------
#Title 
st.info("# 🗞️ :orange[News Classifier]")
st.markdown("###### *************************************************************************************************************************************************************************************************************")
#---------------------------------------------------------------------------------------------------------
col1,col2=st.columns(2)
with col1:
    st.markdown("#### Enter the News Content Below ⤵️")
    text=st.text_input("")                    
    predict_button=st.button(":orange[Predict Category]")
    if predict_button and text:
                nlp=spacy.load("en_core_web_sm") #Using Small English Model
                #-----------------------------------------------------------------------------------------
                processed_text=[]
                #------------------------------------------------------------------------------------------
                #Remove Extra spaces
                text = re.sub(r'\s+', ' ', text).strip()
                #-------------------------------------------------------------------------------------------
                #Removing digits
                text=re.sub(r'\d', '', text)
                #-------------------------------------------------------------------------------------------
                #Removing URL
                pattern_1 = r'https?://\S+|www\.\S+'
                #-------------------------------------------------------------------------------------------
                text= re.sub(pattern_1, '', text)
                #Removing the Emojis
                text = emoji.demojize(text)
                #-------------------------------------------------------------------------------------------
                text = re.sub(r'\w+@', '', text)
                data=nlp(text)
                for token in data:
                        #Removing stopwords and punctuations
                        if token.is_stop or token.is_punct or token.like_num or token.text==" " or token.text=="|" or len(token.text)==1:
                            continue
                        else:
                            
                            #Lemmatization
                            processed_text.append(token.lemma_.lower()) 
                #-----------------------------------------------------------------------------------------
                #Words converted to vectors
                def vectorize(words_list):
                    words_vecs = [w2v_model.wv[word] for word in words_list if word in w2v_model.wv]
                    if len(words_vecs) == 0:
                        return np.zeros(100)
                    words_vecs = np.array(words_vecs)
                    return words_vecs.mean(axis=0)
                #Creating a new column to store vectors
                content_vectors=vectorize(processed_text)
                #Reshaping the vector size
                content_vectors_reshaped = content_vectors.reshape(1, -1)
                predicted_value=svm_classifier_model.predict(content_vectors_reshaped)
                predicted_value=predicted_value[0]
                def category(value):
                    if value==1:
                        return"💼 Business"
                    if value==2:
                        return "👨🏻‍🎓 Education"
                    if value==3:
                        return "🏆 Sports"
                    if value==4:
                        return "📟 Technology"
                    if value==5:
                        return "🎬 Entertainment"
                news_category=category(predicted_value)
                print(f'The news content belongs to the category :"{news_category}"')
                st.markdown(f'### The news content belongs to ":green[{news_category}]" category.')
                # st.markdown("## :green[News belongs to:] {} Lakhs".format(news_category))
    st.markdown("### ")
    st.markdown("### ")
    st.markdown("### ")
    st.markdown("### ")
    st.markdown("### ")
    st.text("-created by banuprakash vellingiri ❤︎ ")               
    st.text("Note : This GUI is just for demonstration, predictions may differ.")
    st.text("Thank you 👍")
with col2:  
     image=Image.open(r"C:\Users\banup\Desktop\N\news_image.png")
     st.image(image)
#----------------------------------------------------------------------------------------------------------
