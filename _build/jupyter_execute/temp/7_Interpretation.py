#!/usr/bin/env python
# coding: utf-8

# # Interpretation
# 
# ### Load Data

# In[1]:


import pandas as pd

df = pd.read_csv('data/tag_df_final.csv')


# Viewing the top 10 game aspects tagged with a non-neutral sentiment polarity reveals that 'game' has by far the highest frequency.
# 
# Given the lack of specific information this provides, it will be useful to preclude this token from subsequent charts.

# In[2]:


(df[df['sentiment']!=0]
     .groupby('aspect',as_index=False)['description']
     .count().rename(columns={'description':'Count'})
     .sort_values(by='Count',ascending=False)
     .head(10)).reset_index(drop=True)


# ## Aspect Frequency
# 
# Charting the most frequent aspects tagged with a non-neutral score, it is clear that players felt more strongly aboput the camapaign than other aspects of the game.
# 
# Campaign is the aspect of the game most frequently referred to in a positive context and the second most frquently cited negative aspect. However, the number of postivie references far outstrip all other aspects of game design in either sentiment polarity.
# 
# Notably, many aspects of game design appear to be quite divisive for the playerbase with several featuring in both sentiment lists.

# In[3]:


#%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

sns.catplot(data = (df[(df['sentiment']>0) & (df['aspect']!='game')]
                    .groupby('aspect')[['sentiment']]
                    .count()
                    .sort_values(by='sentiment',ascending=False)
                    .reset_index()
                    .head(20)),
            y='aspect',
            x='sentiment',
            kind='bar',
            palette = ['#88D8B0'],
            height = 6,
           aspect = 1.5)
plt.title('Positive Game Aspects',fontsize=14)
plt.tick_params(labelsize=12)
plt.ylabel('Descriptor',fontsize=12)
plt.xlabel('Count',fontsize=12)
plt.xticks(list(range(0,1401,100)))
plt.tight_layout()
plt.show();

sns.catplot(data = (df[(df['sentiment']<0) & (df['aspect']!='game')]
                    .groupby('aspect')[['sentiment']]
                    .count()
                    .sort_values(by='sentiment',ascending=False)
                    .reset_index()
                    .head(20)),
            y='aspect',
            x='sentiment',
            kind='bar',
            palette = ['#FF6F69'],
            height = 6,
           aspect = 1.5)
plt.title('Negative Game Aspects',fontsize=14)
plt.tick_params(labelsize=12)
plt.ylabel('Descriptor',fontsize=12)
plt.xlabel('Count',fontsize=12)
plt.xticks(list(range(0,1401,100)))
plt.tight_layout()
plt.show();


# In[4]:


import numpy as np
import holoviews as hv
from holoviews import opts, dim
import colorcet as cc
hv.extension('bokeh')
hv.output(size=350)

# create list of aspects
aspect_list = list((df[(df['sentiment']!=0) & (df['aspect']!='game')]
                    .groupby('aspect')[['sentiment']]
                    .count()
                    .reset_index()
             ).loc[:,'aspect'])

# create df of aspects & descriptions with connection count
links = (df[(df['sentiment']!=0) & (df['description']!='other') & (df['aspect']!='game') & (df['aspect'].isin(aspect_list))]
     .groupby(by=['aspect','description'],as_index=False)['sentiment']
     .count().rename(columns={'sentiment':'Count'})
)
# restrict to connections >20
links = links[links['Count'] > 25]

# specify node names
nodes = list(set(links['aspect'].tolist() + links['description'].tolist()))
nodes = hv.Dataset(pd.DataFrame(nodes, columns = ['Token']))

# create chord diagram
chord = hv.Chord((links, nodes)).select(value=(5, None))
chord.opts(
    opts.Chord(labels = 'Token', label_text_font_size='12pt', 
               node_color='aspect', node_cmap=cc.cm.glasbey_light, node_size=10, 
               edge_color='aspect', edge_cmap=cc.cm.glasbey_light, 
               edge_alpha=0.9, edge_line_width=1)
)


# label_data = chord.nodes.data
# label_data['rotation'] = np.arctan((label_data.y / label_data.x))
# 
# label_data['y'] = label_data['y'].apply(lambda x: x * 1.1)
# label_data['x'] = label_data['x'].apply(lambda x: x * 1.1)
# 
# labels = hv.Labels(label_data)
# labels.opts(
#     opts.Labels(cmap='magma', text_font_size='10pt',padding=0.08, angle= dim('rotation') * 1260/22 ))
# chord * labels

# In[6]:


# create df counting all links between aspect and sentiment
counts_df = (df[(df['sentiment']!=0) & (df['aspect']!='game')]
    .groupby(by=['aspect','description'])[['sentiment']]
    .count()
    .rename(columns={'sentiment':'Count'})
    .sort_values(by ='Count', ascending =False)
    .reset_index()
)

# restrict to counts >10
counts_df = counts_df[counts_df['Count']>=15]

# create adjacency matrices for only tokens in counts_df
df_adj = df[df['aspect'].isin(set(df.aspect).intersection(set(counts_df.aspect))) & df['description'].isin(set(df.description).intersection(set(counts_df.description)))]
# adj1 (x,y)
adj1 = pd.crosstab(df_adj.description,(df_adj.aspect))
idx = adj1.columns.union(adj1.index)
adj1 = adj1.reindex(index=idx,columns=idx)
# adj2 (y,x)
adj2 = pd.crosstab(df_adj.aspect,df_adj.description)
idx = adj2.columns.union(adj2.index)
adj2 = adj2.reindex(index=idx,columns=idx)
# merge to replace make symetrical
adj = adj1.fillna(adj2)

links = adj.to_numpy()
nodes = list(adj.columns)

#%matplotlib
import mne
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle

start, end = 45, 135
first_half = (np.linspace(start, end, len(nodes)//2) +90).astype(int)[::+1] %360
second_half = (np.linspace(start, end, len(nodes)//2) -90).astype(int)[::-1] %360
node_angles = np.array(list(first_half) + list(second_half))

fig, ax =  plt.subplots(figsize=(20, 20), facecolor='black',subplot_kw=dict(polar=True))
plot_connectivity_circle(links, nodes, interactive= True, ax=ax)
fig.tight_layout()


# In[11]:


#list(set(counts_df.aspect))
counts_df = (df[(df['sentiment']!=0) & (df['aspect']!='game')]
    .groupby(by=['aspect','description'])[['sentiment']]
    .count()
    .rename(columns={'sentiment':'Count'})
    .sort_values(by ='Count', ascending =False)
    .reset_index()
)
counts_df[counts_df['Count']>15].head(45)

