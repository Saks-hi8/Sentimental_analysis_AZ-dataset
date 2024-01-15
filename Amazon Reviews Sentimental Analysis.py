#!/usr/bin/env python
# coding: utf-8

# In[7]:


## Import the libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk


# > # Read in data

# In[8]:


## Loading data

df = pd.read_csv('/Users/sakshichavan/Downloads/Reviews.csv')
print(df.shape)
df = df.head(500)
print(df.shape)


# In[9]:


df.head()


# > Exploratory Data Analysis

# In[25]:


ax = df['Score'].value_counts().sort_index() .plot(kind = 'bar',
    title = 'Reviews by stars')
ax.set_xlabel('Review Stars')
plt.show()


# In[26]:


example = df['Text'][50]
print(example)


# In[30]:


nltk.download('punkt')
tokens = nltk.word_tokenize(example)
tokens[:10]


# In[34]:


nltk.download('averaged_perceptron_tagger')
tagged = nltk.pos_tag(tokens)
tagged[:10]


# In[38]:


nltk.download('maxent_ne_chunker')
nltk.download('words')
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# > # - Evaluating the ratio of Postive - Negative word count

# In[42]:


### VADER Sentiment scoring

nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()


# In[43]:


sia.polarity_scores('I am so happy!')


# In[44]:


sia.polarity_scores('This is the worst thing')


# In[47]:


## Run the polarity score on the entire Dataset

res = {}
for i, row in tqdm(df.iterrows(),total = len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)


# In[55]:


Vaders = pd.DataFrame(res).T
Vaders = Vaders.reset_index().rename(columns={'index': 'Id'})
Vaders = Vaders.merge(df, how='left')


# In[57]:


Vaders.head()


# In[59]:


ax = sns.barplot(data = Vaders, x = 'Score', y = 'compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()


# In[68]:


fig, axs = plt.subplots(1, 3, figsize = (15, 5))
sns.barplot(data = Vaders, x = 'Score', y = 'pos', ax = axs[0])
sns.barplot(data = Vaders, x = 'Score', y = 'neu', ax = axs[1])
sns.barplot(data = Vaders, x = 'Score', y = 'neg', ax = axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




