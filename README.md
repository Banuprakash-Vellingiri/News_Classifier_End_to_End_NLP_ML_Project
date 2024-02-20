# 🗞️ **News Classifier** (END to END PROJECT)
-***Building an Automated New Classification System with NLP Techniques***

### 🎯 Aim :
 The aim of this project is to ***build a  classification model*** which should efficiently classify different category of news according to thier content. .It is an ***End to End project*** where the pipeline starts from ***scraping data*** and ends in ***building a model***.

### 🐾 Steps involved :

-  Web scraping
-  Data storage
-  Data cleaning
-  Natural Language Processing (NLP)
-  Model training
-  Model evaluation
-  Model deployment

### 🛠️ Tools Required :
      - Python
      - Selenium (web scraping)
      - MYSQL database
      - Pandas
      - NumPy
      - Matplotlib and Seaborn (visualization)
      - gensim (text representation model)
      - Sci-kit learn (machine learning)
      - Streamlit (GUI)
## 👨‍💻 Approach :
### 💎 Web Scraping :
 The datasets required for this project are scraped from news magazine website, ***"[The Times Of India](https://timesofindia.indiatimes.com/)"***.News content from ***"5"*** different categories such as,

- Business 
- Education
- Sports
- Technology
- Entertainment

are scraped by an automated web scraping tool called ***Selenium***. **Headlines,Description,Content** and **Category** of news are scraped.
- ## Each category around ***"2000"*** unique contents are extracted.
### 📂 Data Storage:
   The scraped datas are stored in  ***MYSQL database*** for easy retrieval.
### 🧹 Data cleaning:
  The stored data is pulled out from MYSQL database and loaded in dataframe and the necessary data wrangling operations are carried out.
### 🗣️ Natural Language Processing (NLP):
The cleaned dataset undergoes various NLP tasks such as 
- Tokenization
- Stopwords removal
- Punctuations removal
- Lemmatization.
### 🔢 Text Representation:
The NLP processed contents are converted into ***numerical vectors*** with the help of text representation technique ***"Word2Vec"***.This ***Word Embeddings*** efficiently captures ***Semantic Relationships*** between the words in a corpus and converts into numerical vectors, which considerably reduces dimensions.
### 🤖 Machine Learning Operations :
 - ### Clustering:
     - By using unsupervised ***"k-means clustering"*** algorithm, the news contents in the datasets are tried to cluster among different news categories.
     - After clustering , tried to manually name the labels of each clusters. 
 - ### Classification :
     - By using ***Support Vector Machine (SVM) Classifier*** algorithm, a classification model is built to predict the target (news category).The model is ***hyper tuned*** with a suitable parameters.The tarined model is evaluated with different evaluation metrics.With a good accuracy ,the model is tried to predict category of a new unseen news content.
### 🌐Streamlit :
 - A ***web application (GUI)*** was built by using streamlit to intake news content as input from the user and to display the predicted category.

### 📰 Documentation :

- [Python](https://docs.python.org/3/)
- [MYSQL](https://dev.mysql.com/doc/)
- [pandas](https://pandas.pydata.org/docs/)
- [Numpy](https://numpy.org/doc/)
- [Matplotlib](https://matplotlib.org/stable/index.html)
- [Seaborn](https://seaborn.pydata.org/)
- [Sci-kit learn](https://scikit-learn.org/stable/)
- [Gensim](https://radimrehurek.com/gensim/auto_examples/index.html)
- [Streamlit](https://docs.streamlit.io/)
