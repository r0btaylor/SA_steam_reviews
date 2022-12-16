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
# Charting the most frequent aspects tagged with a non-neutral score, it is clear how divisive many aspects of the game design are for the player base.
# 
# Several aspects feature in the top 20 of both sentiment lists which is indicative of the 'mixed' review score that game currently has on the Steam store.
# 
# While the campaign features promintently in both lists, the more than 1600 positive references far outstrip the number of negative references made in user reviews.
# 
# Similarly, the over 350 negative references to the game's multiplayer aspect, is outweighed by almost 800 positive references.
# 
# The remainder of the overlapping aspects of design seem far more divisive and demonstrate more even splits in player opinion.

# In[3]:


#%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
pos = (df[(df['sentiment']>0) & (df['aspect']!='game')]
                    .groupby('aspect')[['sentiment']]
                    .count()
                    .sort_values(by='sentiment',ascending=False)
                    .reset_index()
                    .head(20))
                    
sns.catplot(data = pos,
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
plt.xticks(list(range(0,1701,100)))
plt.tight_layout()
plt.show();

neg = (df[(df['sentiment']<0) & (df['aspect']!='game')]
                    .groupby('aspect')[['sentiment']]
                    .count()
                    .sort_values(by='sentiment',ascending=False)
                    .reset_index()
                    .head(20))

sns.catplot(data = neg,
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
plt.xticks(list(range(0,1701,100)))
plt.tight_layout()
plt.show();


# In[20]:


import numpy as np
import holoviews as hv
from holoviews import opts, dim

hv.extension('bokeh')
hv.output(size=300)

# create list of aspects from top pos and neg lists
aspect_list = set(pos['aspect'].to_list() + neg['aspect'].to_list())

# Create df of top aspects across pos and neg sentiments
top_df = df[(df['aspect'].isin(aspect_list))]

desc_counts = (top_df
     .groupby(by=['description'])[['sentiment']]
     .count()
     .rename(columns={'sentiment':'Count'})
     .sort_values(by ='Count', ascending =False)
     .reset_index())

top_df = top_df = df[(df['aspect'].isin(aspect_list)) & (df['description'].isin(desc_counts[desc_counts['Count']>149]['description']))]

links = (top_df
     .groupby(by=['aspect','description'],as_index=False)[['sentiment']]
     .count()
     .rename(columns={'sentiment':'Count'}))
         
         

# specify node names
#nodes = list(set(links['aspect'].tolist() + links['description'].tolist()))
nodes = list(set(links['aspect'].tolist()))
nodes.extend(set(links['description'].tolist()))
nodes = hv.Dataset(pd.DataFrame(nodes, columns = ['Token']))

def rotate_label(plot, element):
    white_space = "              "
    angles = plot.handles['text_1_source'].data['angle']
    characters = np.array(plot.handles['text_1_source'].data['text'])
    plot.handles['text_1_source'].data['text'] = np.array([x + white_space if x in characters[np.where((angles < -1.5707963267949) | (angles > 1.5707963267949))] else x for x in plot.handles['text_1_source'].data['text']])
    plot.handles['text_1_source'].data['text'] = np.array([white_space + x if x in characters[np.where((angles > -1.5707963267949) | (angles < 1.5707963267949))] else x for x in plot.handles['text_1_source'].data['text']])
    angles[np.where((angles < -1.5707963267949) | (angles > 1.5707963267949))] += 3.1415926535898
    plot.handles['text_1_glyph'].text_align = "center"

# create chord diagram
chord = hv.Chord((links, nodes)).select(Count=(25, None))
chord.opts(
    opts.Chord(title='Reltionships Between Aspects and Tokens', labels = 'Token', label_text_font_size='12pt', 
               node_color='Token', node_cmap=['#c1c1c1','#adadad'],node_size=10,
               edge_color='aspect', edge_cmap=['#FF6F69','#88D8B0','#ffcc5c'],
               hooks=[rotate_label], edge_alpha=0.8, edge_line_width=1)
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

# In[ ]:


# create df counting all links between aspect and sentiment
counts_df = (df[(df['sentiment']!=0) & (df['aspect']!='game') & (df['aspect'].isin(aspect_list))]
    .groupby(by=['aspect','description'])[['sentiment']]
    .count()
    .rename(columns={'sentiment':'Count'})
    .sort_values(by ='Count', ascending =False)
    .reset_index()
)

# restrict to counts >20
counts_df = counts_df[counts_df['Count']>=20]

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
# merge to replace & make symetrical
adj = adj1.fillna(adj2)

links = adj.to_numpy()
nodes = list(adj.columns)

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


# In[ ]:


# Create df of top aspects across pos and neg sentiments
top_df = df[(df['aspect'].isin(pos['aspect'])) | (df['aspect'].isin(neg['aspect']))]
# groupby to create count of descripto/sentiment links
top_df =(top_df
     .groupby(by=['aspect','description'])[['sentiment']]
     .count()
     .rename(columns={'sentiment':'Count'})
     .sort_values(by ='Count', ascending =False)
     .reset_index())
# restrict to links greater than 20 occurrences 
top_df = top_df[top_df['Count']>=20]

# create adjacency matrices for only tokens in top_df
df_adj = df[df['aspect'].isin(set(df.aspect).intersection(set(top_df.aspect))) & df['description'].isin(set(df.description).intersection(set(top_df.description)))]
# adj1 (x,y)
adj1 = pd.crosstab(df_adj.description,(df_adj.aspect))
idx = adj1.columns.union(adj1.index)
adj1 = adj1.reindex(index=idx,columns=idx)
# adj2 (y,x)
adj2 = pd.crosstab(df_adj.aspect,df_adj.description)
idx = adj2.columns.union(adj2.index)
adj2 = adj2.reindex(index=idx,columns=idx)
# merge to replace & make symetrical
adj = adj1.fillna(adj2)

# define array of links
links = adj.to_numpy()

#define names of nodes
nodes = list(adj.columns)

label_names = set(top_df['aspect'].to_list() + top_df['description'].to_list())
lh_labels = list(set(top_df['aspect']))
rh_labels = list(set(top_df['description']))

node_order = list()
node_order.extend(lh_labels[::-1])  # reverse the order
node_order.extend(rh_labels)

node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) / 2])


# In[ ]:


node_angles = circular_layout(label_names, node_order, start_pos=270,
                              group_boundaries=[0, len(lh_labels)])

fig, axes = plot_connectivity_circle(links, label_names, 
    node_angles=node_angles)


# In[ ]:


node_angles = circular_layout(label_names, node_order, start_pos=0,
                              group_boundaries=[0, len(lh_labels)])

fig, axes = plot_connectivity_circle(links, label_names, 
    node_angles=node_angles)


# In[ ]:


# Create df of top aspects across pos and neg sentiments
top_df = df[(df['aspect'].isin(pos['aspect'])) | (df['aspect'].isin(neg['aspect']))]

top_df = (top_df
     .groupby(by=['description'])[['sentiment']]
     .count()
     .rename(columns={'sentiment':'Count'})
     .sort_values(by ='Count', ascending =False)
     .reset_index())

top_df = df[(df['aspect'].isin(aspect_list)) & (df['description'].isin(top_df[top_df['Count']>99]['description']))]

# create adjacency matrices for only tokens in top_df
df_adj = df[df['aspect'].isin(set(df.aspect).intersection(set(top_df.aspect))) & df['description'].isin(set(df.description).intersection(set(top_df.description)))]
# adj1 (x,y)
adj1 = pd.crosstab(df_adj.description,(df_adj.aspect))
idx = adj1.columns.union(adj1.index)
adj1 = adj1.reindex(index=idx,columns=idx)
# adj2 (y,x)
adj2 = pd.crosstab(df_adj.aspect,df_adj.description)
idx = adj2.columns.union(adj2.index)
adj2 = adj2.reindex(index=idx,columns=idx)
# merge to replace & make symetrical
adj = adj1.fillna(adj2)

# define array of links
links = adj.to_numpy()

#define names of nodes
nodes = list(adj.columns)

label_names = set(top_df['aspect'].to_list() + top_df['description'].to_list())
lh_labels = list(set(top_df['aspect']))
rh_labels = list(set(top_df['description']))

node_order = list()
node_order.extend(lh_labels[::-1])  # reverse the order
node_order.extend(rh_labels)

node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) / 2])

node_angles = circular_layout(label_names, node_order, start_pos=270,
                              group_boundaries=[0, len(lh_labels)])

fig, axes = plot_connectivity_circle(links, label_names, 
    node_angles=node_angles)

