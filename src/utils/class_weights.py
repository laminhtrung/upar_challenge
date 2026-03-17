import torch
from tqdm import tqdm

def compute_class_weights(train_loader, num_classes, device = 'cpu'):
    class_frequencies = torch.zeros(num_classes, dtype = torch.float32, device = device)
    for _,labels in tqdm( train_loader,desc=" Computing class weight", leave= False):
        labels = labels.to(device)
        class_frequencies += labels.sum(dim = 0)
    
    max_frequence = class_frequencies.max()
    class_weights = torch.log(max_frequence/ (class_frequencies + 1e-6))
    
    return class_weights    
        
    