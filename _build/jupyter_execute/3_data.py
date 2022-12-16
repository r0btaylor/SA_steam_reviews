#!/usr/bin/env python
# coding: utf-8

# # The Data

# ```{figure} https://cdn.akamai.steamstatic.com/steam/apps/1938090/header.jpg?t=1668017465
# ---
# align: center
# ---
# ```
# 
# Review data for the title ['Call of Duty: Modern Warfare 2'](https://store.steampowered.com/app/1938090/Call_of_Duty_Modern_Warfare_II/) published by Activision were collected. 
# 
# At the time of access (2022-12-11), this title held a 'Mixed' review score based on 142,374 user reviews.
# 
# Reviews were scraped from the Steam store using the `steamreviews` API for Python {cite}`wok_2018`.

# In[1]:


# api access

import steamreviews

# set parameters
request_params = dict()
request_params['language'] = 'english'
request_params['purchase_type'] = 'all'
app_id = 1938090

# store results as dictionary
review_dict, query_count = steamreviews.download_reviews_for_app_id(app_id,chosen_request_params=request_params)


# All available English language reviews were scraped.     
# 
# Review text is extracted and all observations without text are dropped. This forms an initial sample of 115,952 observations.
# 
# The resulting data frame is stored as a .csv for use in subsequent stages of the project.

# In[2]:


import pandas as pd

review_id = [x for x in review_dict['reviews']]
review_text = [review_dict['reviews'][x]['review'] for x in review_id]

df = pd.DataFrame({'review_text':review_text})

# Keep reviews with >=1 word
df = df.drop(df[df['review_text'].str.split().str.len()<1].index)

df.to_csv('data/processed_review_data.csv',index=False)

df

