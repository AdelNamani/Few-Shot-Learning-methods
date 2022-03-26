import torch
import time
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def getProjection (state,data,num_ep,num_class,dev):
  X=np.empty([0, 1600])
  Y=[]
  step=0
  for sample in data :
      samp={}
      if step<num_ep:
        with torch.no_grad():
          sample['xs'] = sample['xs'].to(dev)
          sample['xq'] = sample['xq'].to(dev)
          samp["xs"]= sample['xs'] [:num_class]
          samp["xq"] = sample['xq'] [:num_class]
          samp["class"] = sample['class'] [:num_class]

          emb=state.model.getEmbedding(samp)
          emb=emb.reshape((-1,1600))

          
          emb=emb.cpu().detach().numpy()
         
          X=np.concatenate((X, emb), axis=0)
          Y=Y+np.array([[x]*5 for x in samp["class"]]).reshape(-1).tolist()



          step+=1
      else :
        break
  return X,Y


def vizualize_tsne (test_X,test_Y):
  df_subset={}
  time_start = time.time()
  tsne = TSNE(n_components=2,verbose=1, perplexity=40, n_iter=300)
  tsne_results = tsne.fit_transform(test_X,test_Y)
  print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
  df_subset['tsne-2d-one'] = tsne_results[:,0]
  df_subset['tsne-2d-two'] = tsne_results[:,1]
  df_subset['y']= test_Y
  plt.figure(figsize=(16,10))
  sns.scatterplot(
      x="tsne-2d-one", y="tsne-2d-two",
      hue="y",
      palette=sns.color_palette("hls", len(np.unique(np.array(test_Y)))),
      data=df_subset,
      legend="full",
      alpha=0.3
  )

def get_Sub_Set(X,Y,num_class):
  labels = np.random.choice((np.unique(np.array(Y))),num_class, replace=False)
  res_X=np.empty([0, 1600])
  res_Y=[]
  for i in range (len(Y)):
    if Y[i] in labels :
      emb=X[i].reshape((1,-1))
      res_X=np.concatenate((res_X,emb),axis=0)
      res_Y.append(Y[i])
  return res_X, res_Y