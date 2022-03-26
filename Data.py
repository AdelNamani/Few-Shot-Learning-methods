from tqdm import tqdm
import random
import torchvision
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchnet.transform import compose
import pickle
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian


"""# MINI IMAGE NET """
def extract_episodes(n_way,n_support,n_query,n_episode,datax,datay):
  labels=list(datay.keys())
  res=[]
  for i in range(n_episode):
    K = np.random.choice(labels,n_way, replace=False)
    all_xs=torch.empty((n_way,n_support,datax[0].shape[0],datax[0].shape[1],datax[0].shape[2]))
    all_xq=torch.empty((n_way,n_query,datax[0].shape[0],datax[0].shape[1],datax[0].shape[2]))
    i=0
    for k in K :
      sample_cls = np.random.permutation(datax[datay[k]])[:(n_support+n_query)]
      all_xs[i]=torch.from_numpy(np.array(sample_cls[:n_support])).float()
      all_xq[i]=torch.from_numpy(np.array(sample_cls[n_support:])).float()
      i+=1      
    res.append({
        'class':list(K),
        'xs': all_xs,
        'xq': all_xq
      })
     
  return torch.utils.data.DataLoader(res, batch_size=1, shuffle=True,collate_fn=lambda d:d[0])

def createDatasets(mode,path,hyperparams):
  with open(path +'/mini-imagenet-cache-'+mode+'.pkl', 'rb') as f:
    data_train = pickle.load(f)

  
  datax_train=data_train['image_data']
  datay_train=data_train['class_dict']
  data_train=None
  way=0
  shot=0
  query=0
  episodes=0
  if mode =='train' :
    way=hyperparams['train_way']
    shot=hyperparams['train_shot']
    query=hyperparams['train_query']
    episodes=hyperparams['train_episodes']
  else :
    way=hyperparams['test_way']
    shot=hyperparams['test_way']
    query=hyperparams['test_way']
    episodes=hyperparams['test_episodes']


  featurestransform = [
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
  featurestransform = compose(featurestransform)

  datax_train= list(map(lambda x: featurestransform(Image.fromarray(x)), datax_train))
  datax_train = torch.stack(datax_train, 0)
  return extract_episodes(way,shot,query,episodes,datax_train,datay_train)


def loadDataMiniIMageNet(path,mode,hyperparams):
  return {
      'train':createDatasets(mode[0],path,hyperparams) ,
      'test':createDatasets(mode[1],path,hyperparams)  ,
      'val': createDatasets(mode[2],path,hyperparams) 
  }


"""# CIFAR100"""


def getImageByClassCifar(data,K):
  res={}
  for im in data :
    if(im[1] in K):
      try :
        r=im[0]
        res[im[1]] = np.append(res[im[1]],r.reshape((1,3,32,32)),axis=0)
      except KeyError :
        r=im[0]
        res[im[1]] = r.reshape((1,3,32,32))
  return res

def extract_episodes_cifar100(n_way,n_support,n_query,n_episode,datax,labels):
  res=[]
  for i in range(n_episode):
    K = np.random.choice(labels,n_way, replace=False)
    all_xs=torch.empty((n_way,n_support,datax[0][0].shape[0],datax[0][0].shape[1],datax[0][0].shape[2]))
    all_xq=torch.empty((n_way,n_query,datax[0][0].shape[0],datax[0][0].shape[1],datax[0][0].shape[2]))
    i=0
    listImages=getImageByClassCifar(datax,K)
    for k in K :
      sample_cls = np.random.permutation(listImages[k])[:(n_support+n_query)]
      all_xs[i]=torch.from_numpy(np.array(sample_cls[:n_support])).float()
      all_xq[i]=torch.from_numpy(np.array(sample_cls[n_support:])).float()
      i+=1      
    res.append({
        'class':list(K),
        'xs': all_xs,
        'xq': all_xq
      })
     
  return torch.utils.data.DataLoader(res, batch_size=1, shuffle=True,collate_fn=lambda d:d[0])

def createDatasetsCifar(data,mode,hyperparams,labels):
  way=0
  shot=0
  query=0
  episodes=0
  if mode =='train' :
    way=hyperparams['train_way']
    shot=hyperparams['train_shot']
    query=hyperparams['train_query']
    episodes=hyperparams['train_episodes']
  else :
    way=hyperparams['test_way']
    shot=hyperparams['test_way']
    query=hyperparams['test_way']
    episodes=hyperparams['test_episodes']

  return extract_episodes_cifar100(way,shot,query,episodes,data,labels)

def loadDataCifar(hyperparams):
  stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
  train_transform = tt.Compose([
      tt.RandomHorizontalFlip(),
      tt.RandomCrop(32,padding=4,padding_mode="reflect"),
      tt.ToTensor(),
      tt.Normalize(*stats)
  ])

  test_transform = tt.Compose([
      tt.ToTensor(),
      tt.Normalize(*stats)
  ])
  train_data = CIFAR100(download=True,root="./data",transform=train_transform)
  test_data = CIFAR100(root="./data",train=False,transform=test_transform)
  all_data = train_data + test_data

  TRAIN_RATIO=0.64
  TEST_RATIO=0.2


  train_labels = np.random.choice(np.arange(100),int(100*TRAIN_RATIO), replace=False)
  rest=[x for x in np.arange(100) if x not in train_labels]
  test_labels = np.random.choice(rest,int(100*TEST_RATIO), replace=False)
  val_labels = [x for x in np.arange(100) if x not in train_labels and x not in test_labels]
  train_data = [x for x in all_data if x[1] in train_labels]
  test_data = [x for x in all_data if x[1] in test_labels]
  val_data = [x for x in all_data if x[1] in val_labels]


  
  return {
      'train':createDatasetsCifar(train_data,'train',hyperparams,train_labels) ,
      'test':createDatasetsCifar(test_data,'test',hyperparams,test_labels)  ,
      'val': createDatasetsCifar(val_data,'val',hyperparams,val_labels) 
  }


"""# OMNIGLOT"""



def data_augmentation (dataset):
  final_train_data = []
  for i in tqdm(range(len(dataset))):
      final_train_data.append(dataset[i])
      final_train_data.append(((rotate(dataset[i][0], angle=45, mode = 'wrap')),dataset[i][1]))
      final_train_data.append(((np.fliplr(dataset[i][0])),dataset[i][1]))
      final_train_data.append((np.flipud((dataset[i][0])),dataset[i][1]))
      final_train_data.append((random_noise(dataset[i][0],var=0.2**2),dataset[i][1]))

  return final_train_data

def getImageByClassOmniglot(data,K):
  res={}
  for im in data :
    if(im[1] in K):
      try :
        r=im[0]
        res[im[1]] = np.append(res[im[1]],r.reshape((1,1,105,105)),axis=0)
      except KeyError :
        r=im[0]
        res[im[1]] = r.reshape((1,1,105,105))
  return res

def extract_episodes_Omniglot(n_way,n_support,n_query,n_episode,datax,listlabels):
  labels=np.array(listlabels)
  res=[]
  for i in range(n_episode):
    K = np.random.choice(labels,n_way, replace=False)
    
    all_xs=torch.empty((n_way,n_support,datax[0][0].shape[0],datax[0][0].shape[1],datax[0][0].shape[2]))
    
    all_xq=torch.empty((n_way,n_query,datax[0][0].shape[0],datax[0][0].shape[1],datax[0][0].shape[2]))
    i=0
    listImages=getImageByClassOmniglot(datax,K)
    for k in K :
      sample_cls = np.random.permutation(listImages[k])[:(n_support+n_query)]
      all_xs[i]=torch.from_numpy(np.array(sample_cls[:n_support])).float()
      all_xq[i]=torch.from_numpy(np.array(sample_cls[n_support:])).float()
      i+=1      
    res.append({
        'class':list(K),
        'xs': all_xs,
        'xq': all_xq
      })
     
  return torch.utils.data.DataLoader(res, batch_size=1, shuffle=True,collate_fn=lambda d:d[0])

def createDatasetsOmniglot(data,mode,hyperparams,listlabels):
  way=0
  shot=0
  query=0
  episodes=0
  if mode =='train' :
    way=hyperparams['train_way']
    shot=hyperparams['train_shot']
    query=hyperparams['train_query']
    episodes=hyperparams['train_episodes']
  else :
    way=hyperparams['test_way']
    shot=hyperparams['test_way']
    query=hyperparams['test_way']
    episodes=hyperparams['test_episodes']

  return extract_episodes_Omniglot(way,shot,query,episodes,data,listlabels)


def loadDataOmniglot(hyperparams):
  
  all_data = torchvision.datasets.Omniglot(
    root="./data", download=True, transform=torchvision.transforms.ToTensor())
  all_data=list(all_data)

  NUM_LABELS = all_data[-1][1]+1
  TRAIN_RATIO=0.64
  TEST_RATIO=0.2
  

  train_labels = np.random.choice(np.arange(NUM_LABELS),int(NUM_LABELS*TRAIN_RATIO), replace=False)
  rest=[x for x in np.arange(NUM_LABELS) if x not in train_labels]
  test_labels = np.random.choice(rest,int(NUM_LABELS*TEST_RATIO), replace=False)
  val_labels = [x for x in np.arange(NUM_LABELS) if x not in train_labels and x not in test_labels]
  train_data = [x for x in all_data if x[1] in train_labels]
  test_data = [x for x in all_data if x[1] in test_labels]
  val_data = [x for x in all_data if x[1] in val_labels]

  train_data =data_augmentation(train_data)
  test_data =data_augmentation(test_data)
  val_data =data_augmentation(val_data)

  return {
      'train':createDatasetsOmniglot(train_data,'train',hyperparams,train_labels) ,
      'test':createDatasetsOmniglot(test_data,'test',hyperparams,test_labels)  ,
      'val': createDatasetsOmniglot(val_data,'val',hyperparams,val_labels) 
  }


def loadTheData(numData,hyperparams,mode=None,path=None):
  if (numData == 0 and (path==None or mode == None)) :
    print("data not load, please insert a correct path or a correct mode !")
    raise (ValueError)
  else :
    if (numData==0):
      return loadDataMiniIMageNet(path,mode,hyperparams)
    elif (numData==1) :
      return loadDataCifar(hyperparams)
    elif (numData==2) :
      return  loadDataOmniglot(hyperparams)
    else :
      print("data not load please insert a correct numData")
      print("\t 0 for MINIIMAGENET")
      print("\t 1 for CIFAR100")
      print("\t 2 for OMNIGLOT")
      raise (ValueError)
