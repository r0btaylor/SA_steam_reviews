#!/usr/bin/env python
# coding: utf-8

# # Steam User Reviews: Aspect Based Sentiment Analysis
# ## Extracting insight for game developers
# 
# <p align="center"> <img src="https://pngimages.in/uploads/png/Steam_Png_Image.png"  width="150" height="150"> </p>
#     
# ### Introduction
# 
# In a [previous project](https://r0btaylor.github.io/NLP_steam_reviews), I attempted to identify salient aspects of game design by applying NLP and machine learning to user reviews on the Steam webstore.
# 
# In the previous methodology, review text was used to predict an overall review classification. Weighted noun tokens were then extracted in an attempt to identify prominent aspects of the game associated with each classification of review. 
# 
# That method was fundamentally flawed as it was impossible to distinguish between varied sentiments within each review classification.
# 
# In the present project, an alternative methodology will be employed. Aspect based sentiment analysis will be performed on a sentence by sentence basis. This should enable specific aspects of game design that users describe as particularly positive or negative to be identified.
# 
# The results will hopefully offer a means for game developers to easily identify prominent negative/positive aspects of game design that could be used to inform future development patches.
# 
# 
