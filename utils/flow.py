
# coding: utf-8

# In[3]:


import os
import numpy as np
import shutil


# In[6]:


os.path.stat


# In[2]:



tvl1_flow_u = r'tvl1_flow/u'
tvl1_flow_v = r'tvl1_flow/v'
jpeg = r'hmdb51'


# In[3]:


title = os.listdir(jpeg)


# In[7]:


m = {}
for i in title:
    t_path = os.path.join(jpeg,i)
    for j in os.listdir(t_path):
        m[j[:-4]] = i


# In[8]:


m


# In[10]:


for i in os.listdir(tvl1_flow_u):
    try:    
        title = m[i]
    except:
        continue

    if os.path.exists(os.path.join(tvl1_flow_u,title)) is False:
        os.makedirs(os.path.join(tvl1_flow_u,title))
    shutil.move(os.path.join(tvl1_flow_u,i),os.path.join(tvl1_flow_u,title))


# In[11]:


for i in os.listdir(tvl1_flow_v):
    try:    
        title = m[i]
    except:
        continue
    if os.path.exists(os.path.join(tvl1_flow_v,title)) is False:
        os.makedirs(os.path.join(tvl1_flow_v,title))
    shutil.move(os.path.join(tvl1_flow_v,i),os.path.join(tvl1_flow_v,title))

