import torch
import math
from torch import optim
from pathlib import Path
import BigProtonet
import Protonet
import Encoders
from torch.utils.tensorboard import SummaryWriter
import datetime

class State:
    def __init__(self, model, optim, proto_optimizer, scheduler=None):
        self.model = model
        self.optimizer = optim
        self.proto_optimizer = proto_optimizer
        self.scheduler = scheduler
        self.epoch = 0

def init_train(save_path, hyperparameters):
    save_path = Path(save_path)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if save_path.is_file():
        with save_path.open('rb') as fp:
            state = torch.load(fp, map_location=torch.device(dev))
    else:
        encoder = Encoders.Encoder(hyperparameters)
        if hyperparameters['bigProto'] :
            model = BigProtonet.BigProtonet(encoder)
            parameters, parameters_for_proto = model.encoder.parameters(), model.proto_param.parameters()

            # The prototype parameters optmizer is initialized with no parameters as the model has seen 
            # no class before and its parameters (radiuses) are related to classes.
            proto_optimizer = optim.Adam([{'params': parameters_for_proto}], lr=hyperparameters['lr_proto'])
        else:
            model = Protonet.Protonet(encoder)
            parameters = model.encoder.parameters()
            proto_optimizer = None

        if hyperparameters['encoder']=='convnet' : optimizer = optim.SGD(parameters , lr=hyperparameters['lr']) 
        else : optimizer = optim.SGD(parameters , lr=hyperparameters['lr'] ,momentum = 0.9) 
        model.to(dev)            
        state = State(model, optimizer, proto_optimizer)

    experiment_name = 'BigPrototypes' if hyperparameters['bigProto'] else 'Prototype'
    experiment_name += '-' + str(hyperparameters['test_way']) + 'ways-'+ str(hyperparameters['test_shot']) +'shots-'+hyperparameters['encoder']
    train_writer = SummaryWriter("runs/train"+ experiment_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    val_writer = SummaryWriter("runs/test"+ experiment_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return state, save_path, dev, train_writer, val_writer

def train(state, train_loader, val_loader, max_epochs, dev, save_path, patience, train_writer, val_writer , hyperparameters):
    # Early stopping implementation
    best_val_loss = math.inf
    patience_count = patience
    # Training loop
    for epoch in range(state.epoch, max_epochs):
        # Supervision metrics
        loss, acc = 0, 0
        loss_v, acc_v = 0, 0
        # iterating over previously sampled episodes
        for sample in train_loader:
            state.optimizer.zero_grad()
            if hyperparameters['bigProto'] :
                state.proto_optimizer.zero_grad()

            # support and query set to device
            sample['xs'] = sample['xs'].to(dev)
            sample['xq'] = sample['xq'].to(dev)
            
            # Forward and loss calculation
            j, output = state.model.loss(sample,dev)
            
            loss += output['loss']
            acc += output['acc']

            j.backward()
            state.optimizer.step()

            if hyperparameters['bigProto'] :
                # We add the new parameters of the model (radii of new classes)
                # to the prototype optimizer so they can get optimized on backward
                new_radii = []
                for p in state.model.proto_param.parameters():
                    new_radii.append(p)
                state.proto_optimizer.param_groups[0]['params'] = new_radii
                state.proto_optimizer.step()
        # Validation
        for sample in val_loader:
            with torch.no_grad():
                sample['xs'] = sample['xs'].to(dev)
                sample['xq'] = sample['xq'].to(dev)
                if hyperparameters['bigProto'] :
                  j, output = state.model.loss(sample, dev,eval=True)
                else:
                  j, output = state.model.loss(sample,dev)
                loss_v += output['loss']
                acc_v += output['acc']
            
        loss /= len(train_loader)
        acc /= len(train_loader)
        loss_v /= len(val_loader)
        acc_v /= len(val_loader)

        # Early stopping checking
        if loss_v < best_val_loss:
            patience_count = patience
            best_val_loss = loss_v
            with save_path.open("wb") as fp:
                torch.save(state, fp)
        else:
            patience_count -= 1
            if patience_count == 0:
                break

        print("\n\nEpoch:", epoch)
        print('Loss: ', loss, ' Acc: ', acc)
        print('Loss_v: ', loss_v, ' Acc_v: ', acc_v)
        print('Best loss: ', best_val_loss, ' Patience count: ', patience_count)

        val_writer.add_scalar("bigProto"+'/loss', loss_v, epoch)
        train_writer.add_scalar("bigProto"+'/loss', loss, epoch)

        
        state.epoch = epoch + 1
    print('Training ended after ', epoch, ' epochs.\n Best loss value: ', best_val_loss)

def confidence_interval(z, accuracy, n):
    interval = z * math.sqrt( (accuracy * (1 - accuracy)) / n)
    return interval

def test(test_loader, state, device, hyperparameters):
    acc = 0
    for sample in test_loader:
        with torch.no_grad():
            sample['xs'] = sample['xs'].to(device)
            sample['xq'] = sample['xq'].to(device)
            if hyperparameters['bigProto'] :
                j, output = state.model.loss(sample,device, eval=True)
            else:
                j, output = state.model.loss(sample)
            acc += output['acc']
    acc /= len(test_loader)
    ci = confidence_interval(1.96, acc, len(test_loader) * hyperparameters['test_way'] * hyperparameters['test_query'])
    print('Accuracy on test set: ', acc, '+/-', ci,".  95% confidence interval")
    return acc, ci