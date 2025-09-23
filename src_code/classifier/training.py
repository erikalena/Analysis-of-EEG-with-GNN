import os
import numpy as np
import time
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch_geometric.loader import DataLoader
from utils.utils import logger


LEARNING_RATE = 0.001
STEP_SIZE = 7
GAMMA = 0.1

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.warmup_epochs = 20

    def early_stop(self, validation_loss, epoch):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience and epoch > self.warmup_epochs:
                return True
        return False


def save_checkpoint(model: torchvision.models, optimizer: torch.optim, save_path: str, epoch: int):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)


def train(model: torchvision.models, optimizer: torch.optim, criterion: torch.nn, dataloaders: DataLoader, num_epochs: int=10,
            folder: str = None, load_checkpoint: bool = False, checkpoint_path: str = None, device: str ="cpu"):
    """
    Function for model training. It trains the model for a number of epochs 
    and saves the best model based on the validation accuracy.
    Model can be loaded from a checkpoint.
    """
    # make directory to save results
    if not os.path.isdir(folder):
        os.mkdir(folder)
        
    since = time.time()
    start_epoch = 0    

    # load checkpoint if needed
    if load_checkpoint:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # save on file checkpoint file name from which we are loading
        # and the epoch from which we are starting
        with open(os.path.join(folder, 'results.txt'), 'a') as f:
            f.write(f'Loading checkpoint {checkpoint_path} at epoch {start_epoch - 1}\n')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
        
    # move optimizer state to device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    criterion = criterion.to(device)

    # move model parameters to device
    model = model.to(device)
    model.device = device
    logger.info(f'Running on {device}')
    
    # Create a temporary directory to save training checkpoints
    best_model = os.path.join(folder, 'best_model_params.pt')

    torch.save(model.state_dict(), best_model)
    best_acc = 0.0
    logger.info(f'Training on {len(dataloaders["train"])} samples and validating on {len(dataloaders["val"])} samples')

    early_stopper = EarlyStopper(patience=10, min_delta=0.001)
    with open(os.path.join(folder, 'results.txt'), 'a') as f:
        # write header
        f.write(f'epoch,train_loss,train_acc,val_loss,val_acc\n')
        
    for epoch in range(num_epochs):
        logger.info(f'\nEpoch {epoch}/{num_epochs - 1}\n')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            train_loss = 0.0
            size = 0
            
            for data in dataloaders[phase]:
                data = data.to(model.device)
                size += data.num_graphs
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(data.x, data.edge_index, data.batch) 
                    loss = criterion(outputs, data.y)  

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    train_loss += loss.item() * data.num_graphs
                
            if phase == 'train':
                scheduler.step()

            # statistics
            epoch_acc = test(model, dataloaders[phase], folder=None)
            epoch_loss = train_loss / size
            if epoch%5 == 0:
               logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model)
            
            # save checkpoint
            save_checkpoint(model, optimizer, os.path.join(folder, 'checkpoint.pt'), epoch)
            
            # print results to a file
            with open(os.path.join(folder, 'results.txt'), 'a') as f:
                if phase == 'train':
                    f.write(f'{epoch},{epoch_loss},{epoch_acc:.4f},')
                else:
                    f.write(f'{epoch_loss},{epoch_acc:.4f}\n')
            
            if phase == 'val':
                # check if we need to early stop
                if early_stopper.early_stop(epoch_loss, epoch):
                    with open(os.path.join(folder, 'results.txt'), 'a') as f:
                        f.write('\nEarly stopping\n')
                    break
        
    time_elapsed = time.time() - since
    with open(os.path.join(folder, 'results.txt'), 'a') as f:
        f.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model))

    return model

def test(model: torchvision.models, loader: DataLoader, folder: str = None):
    """
    Test the model on the dataloader provided
    """
    # get the device of the model
    device = model.device

    model.eval() 
    correct = 0
    for data in loader: 
        data = data.to(model.device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # get the class with highest probability
        correct += int((pred == data.y).sum()) 
        
    test_acc = correct / len(loader.dataset)
    # print results to a file
    if folder is not None:
        with open(os.path.join(folder, 'results.txt'), 'a') as f:
            f.write(f'Test Acc: {test_acc:.4f}\n')
         
    return test_acc


#################################
# train function for EEGCN model
#################################

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_eegcn(model: torchvision.models, dataloaders: DataLoader, optim: torch.optim, criterion: torch.nn, epochs: int, device: str ="cpu", folder: str = None):
    """
    EEGCN model training
    Input:
    - model: model instance to be trained
    - dataloaders: list for train, valid and test dataloaders
    - optim: chosen optimzer to adjust network's parameters
    - criterion: loss function to be used as evaluation
    - epochs: number of epochs required for training
    - device: on which device to move model and data before training
    - folder: path where to save results
    """
    model = model.to(device)
    model.device = device
    criterion = criterion.to(device)
    train_loader, val_loader, test_loader = dataloaders["train"], dataloaders["val"], dataloaders["test"]
    threshold = 0.4
    early_stopper = EarlyStopper(patience=10, min_delta=0.001)

    with open(os.path.join(folder, 'results.txt'), 'a') as f:
        # write header
        f.write(f'epoch,train_loss,train_acc,val_loss,val_acc\n')
        
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_acc, train_loss, val_acc, val_loss = 0., 0., 0., 0.
        model.train()

        for data in train_loader:
            data = data.to(model.device)
            inputs = (data.x.float(), data.edge_index, None, data.batch)
            labels_y = data.y.to(model.device)

            # zero the parameter gradients
            optim.zero_grad()
            
            out_y, grads = model(*inputs)
            pred = out_y.argmax(dim=1)  # Use the class with highest probability.
            train_acc += int((pred == labels_y).sum())
            loss_aux = grads.square().mean()
            loss = criterion(out_y, labels_y) + 1e-3*loss_aux 
            loss.backward()
            optim.step()
            train_loss += loss.item()
        

        # before measuring validation acc
        model.eval()
        if val_loader is not None and len(val_loader) > 0:
            #bn_train_eval(model) #if epoch < 20 else model.eval()
            for data in val_loader:
                data = data.to(model.device)
                inputs = (data.x.float(), data.edge_index, None, data.batch)
                labels_y = data.y.to(model.device)
                out_y = model(*inputs)
                pred = out_y.argmax(dim=1)  
                val_acc += int((pred == labels_y).sum())
                loss = criterion(out_y, labels_y) 
                val_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader) if val_loader is not None and len(val_loader) > 0 else 0
        train_acc = train_acc / len(train_loader.dataset)
        val_acc = val_acc / len(val_loader.dataset) if val_loader is not None and len(val_loader) > 0 else 0
        
        if epoch %5 == 0:
           logger.info(f"Epoch: {epoch}, train accuracy: {train_acc:.2f}, val accuracy: {val_acc:.2f}")
           logger.info(f"Train loss: {train_loss:.2f}, lr: {optim.param_groups[0]['lr']}")
        with open(os.path.join(folder, 'results.txt'), 'a') as f:
            f.write(f'{epoch},{train_loss},{train_acc:.4f},')
            f.write(f'{val_loss},{val_acc:.4f}\n')
        
        if val_loader:   
            # check if we need to early stop
            if early_stopper.early_stop(val_loss, epoch):
                with open(os.path.join(folder, 'results.txt'), 'a') as f:
                    f.write('Early stopping\n')
                break
    
    if test_loader:
        logger.info("Testing the model...")
        model.eval()
        test_acc = test_eegcn(model, test_loader)
        item = next(iter(test_loader))
        logger.info(item[0].id)
        with open(os.path.join(folder, 'results.txt'), 'a') as f:
                f.write(f'Test Acc: {test_acc:.4f}\n')
        logger.info(f"Test accuracy: {test_acc:.2f}")

    # save model parameters in folder
    torch.save(model.state_dict(), os.path.join(folder, 'model.pt'))
    
    return test_acc
    
def apply_along_axis(function, x, axis: int = 0):
     return torch.stack([
         function(x_i) for x_i in torch.unbind(x, dim=axis)
     ], dim=axis).to(x.device)
        
def test_eegcn(model, loader):
    
    model.eval() # set evaluation mode for the model
    correct = 0

    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset
            data = data.to(model.device)
            inputs = (data.x.float(), data.edge_index, None, data.batch)
            labels = data.y.to(model.device)
            out_y = model(*inputs)
            pred = out_y.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == labels).sum())  # Check against ground-truth labels.
       
    acc = correct / len(loader.dataset)  # Derive ratio of correct predictions.

    return acc 
