import torch
from torch.nn import functional as F
from torch import nn

class BigPrototypeParams(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        # As mentioned in the paper, <the prototype center is re-initialized at each episode
        # as the mean vector of the support embeddings.> The only optimizable parameter of 
        # a prototype is radius
        self.radius = nn.ParameterDict({})

    def center_init(self, support_embeddings):
        center = torch.mean(support_embeddings, 0) 
        if torch.cuda.is_available():
            center = center.cuda()
        return center
    
    def radius_init(self, center, support_embeddings):
        distances = (torch.pow(support_embeddings - center, 2)).sum(1)
        mean_distances = torch.mean(distances).detach()
        radius = mean_distances
        if torch.cuda.is_available():
            radius = radius.cuda()
        return nn.Parameter(radius)

    def forward(self, _class, support_embeddings):
        center = self.center_init(support_embeddings)
        # If the radius of the class is already calculcated during 
        # a previous episode, we use it directly as it's getting
        # optimized with backprop
        if str(_class) not in self.radius:
            radius = self.radius_init(center, support_embeddings)
            self.radius[str(_class)] = radius
        return center, self.radius[str(_class)]



class BigProtonet(nn.Module):
    def __init__(self, encoder):
        super(BigProtonet, self).__init__()
        self.encoder = encoder
        self.proto_param = BigPrototypeParams()
      
    # Return centers and radiuses for all classes.
    def getParams(self, support_embed, classes):
        proto_centers = torch.zeros([support_embed.shape[0],support_embed.shape[2]])
        proto_rad = torch.zeros(support_embed.shape[0])

        for j, label in enumerate(classes):
            support_embeddings = support_embed[j]
            c, r = self.proto_param(label, support_embeddings)
            proto_centers[j] = c
            proto_rad[j] = r

        return proto_centers, proto_rad

    def getEmbedding(self,sample):
      xs = sample['xs'] # support
      xq = sample['xq'] # query

      n_class = xs.size(0)
      n_support = xs.size(1)
      n_query = xq.size(1)

      all_points = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                      xq.view(n_class * n_query, *xq.size()[2:])], 0)

      encoded_points = self.encoder.forward(all_points)
      return encoded_points[:n_class*n_support].view(n_class, n_support, encoded_points.size(-1))
    
    def loss(self, sample, device , eval=False):
        # Extracting support and query sets
        support_points = sample['xs'] 
        query_points = sample['xq'] 
        
        # Sizes
        classes_nb = support_points.size(0)
        supports_nb = support_points.size(1)
        querys_nb = query_points.size(1)

        # We concatenate both sets as we are going to pass them through the encoder
        all_points = torch.cat([support_points.view(classes_nb * supports_nb, *support_points.size()[2:]),
                       query_points.view(classes_nb * querys_nb, *query_points.size()[2:])], 0)

        # We encode the two sets
        encoded_points = self.encoder.forward(all_points)

        # We constuct back our classes X support_points but this time they are in the embedding dim
        points_embedding = encoded_points[:classes_nb*supports_nb].view(classes_nb, supports_nb, encoded_points.size(-1))

        # As mentionned in the paper: 
        # <There are two strategies to handle the unseen classes at the episodic evaluation 
        # stage ... . Alternatively, we could directly estimate the big prototype like vanilla 
        # prototypical network. ... we use the latter strategy in experiments.>
        # In the evaluation stage we estimate prototypes and we do not use BigPrototypeParams
        if eval:
            center = points_embedding.mean(1) # (classes_nb, z_dim)
            expanded_proto_centers = center.unsqueeze(1).expand(points_embedding.size())
            radius = torch.pow(points_embedding - expanded_proto_centers, 2).sum(-1).mean(-1) / 2
    
        else:
            center, radius = self.getParams(points_embedding, sample['class'])

        center = center.to(device)
        radius = radius.to(device)

        # Distances between query points and hyperspheres 
        dists = torch.cdist(encoded_points[classes_nb*supports_nb:], center)**2 - radius

        # Our targets: true classes for query points
        query_actual_classes = torch.empty(classes_nb,querys_nb , dtype=torch.long ).to(device)
        for i in range(classes_nb):          
          query_actual_classes[i] = torch.tensor([i]*querys_nb)
        query_actual_classes = query_actual_classes[:, :, None]

        # Predictions: nearest hypersphere.
        predictions_log = F.log_softmax(-dists, dim=1).view(classes_nb, querys_nb, -1)

        # Calculating the loss
        tmp = torch.zeros([classes_nb,querys_nb,1])
        for i in range(classes_nb) :
         for j in  range(querys_nb):
           tmp[i][j][0] = predictions_log[i][j][i]
        
        tmp = tmp.reshape([classes_nb,querys_nb]) #get rid of the dimension = 1
        tmp = tmp.reshape(classes_nb*querys_nb)
        loss_val = -tmp.view(-1).mean()
        
        # Predicted class
        _, y_hat = predictions_log.max(2)
        acc_val = torch.eq(y_hat, query_actual_classes.squeeze()).float().mean()

        return loss_val, {'loss': loss_val.item(), 'acc': acc_val.item()}