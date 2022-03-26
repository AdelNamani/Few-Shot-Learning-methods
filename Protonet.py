import torch
from torch.nn import functional as F
from torch import nn

class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        
        self.encoder = encoder

    def loss(self, sample, dev):
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

        prototypes = points_embedding.mean(1) #calculer les prototypes
        
        # Distances between query points and the centers 
        dists = torch.cdist(encoded_points[classes_nb*supports_nb:], prototypes)**2  #calculer les distances vers chaque protoypes

        query_actual_classes = torch.empty(classes_nb,querys_nb , dtype=torch.long ).to(dev)
        for i in range(classes_nb):          
          query_actual_classes[i] = torch.tensor([i]*querys_nb)
        query_actual_classes = query_actual_classes[:, :, None]

        predictions_log = F.log_softmax(-dists, dim=1).view(classes_nb, querys_nb, -1)

        # la distance entre un élément de l'ensemble de query et le prototype correspondant à sa classe réelle.
        tmp = torch.zeros([classes_nb,querys_nb,1])
        for i in range(classes_nb) :
         for j in  range(querys_nb):
           tmp[i][j][0] = predictions_log[i][j][i]
        loss_val = -tmp.squeeze().view(-1).mean()
        _, y_hat = predictions_log.max(2)
        acc_val = torch.eq(y_hat, query_actual_classes.squeeze()).float().mean()

        return loss_val, {'loss': loss_val.item(), 'acc': acc_val.item()}