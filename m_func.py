import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from torch import nn
from torch import optim
import numpy as np
from PIL import Image
import seaborn as sns
import argparse


#Load the data

def loader():
    ''' retrieve and transform files so that they can be used in the network'''
    data_dir = './ImageClassifier/flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transforms for the training, validation and test sets
    valid_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
    ])

    train_transforms = transforms.Compose([
    transforms.RandomRotation(25),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
    ])
    
    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    # Define dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 40, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 40, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 40, shuffle = True)
    
    return trainloader, validloader, testloader, train_data

def loader_p(device = 'cuda', arquitecture = 'densenet161', dropout = 0.2, lr = 0.001):
    ma = {'vgg19': 25088,
          'densenet161': 2208}
    
    if arquitecture == 'vgg19':
        model = models.vgg19(pretrained = True)
    if arquitecture == 'densenet161':
        model = models.densenet161(pretrained = True)
    else:
        print('function only accepts vgg19 or densenet161')
    
    #Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False
    
    imgClass = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(ma.get(arquitecture), 1000)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(dropout)),
    ('fc2', nn.Linear(1000, 1000)),
    ('relu2', nn.ReLU()),
    ('dropout2', nn.Dropout(dropout)),
    ('fc3', nn.Linear(1000, 102)),
    ('output', nn.LogSoftmax(dim = 1))
    
    ]))
    
    model.classifier = imgClass
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)
    
    model.to(device)
    
    return model, criterion, optimizer
    
    
# creating architecture
def train_network(epochs = 1, dropout = 0.2, learning_rate = 0.001, data = None, eval_data = None, path = 'mycheckpoint1.pth', mapping_data = None, device = 'cuda', arquitecture = 'densenet161', fc1 = 1000, fc2 = 500):
    
    ma = {'vgg19': 25088,
          'densenet161': 2208}
    
    if arquitecture == 'vgg19':
        model = models.vgg19(pretrained = True)
    if arquitecture == 'densenet161':
        model = models.densenet161(pretrained = True)
    else:
        print('function only accepts vgg19 or densenet161')
    #Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False
    
    imgClass = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(ma.get(arquitecture), fc1)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(dropout)),
    ('fc2', nn.Linear(fc1, fc2)),
    ('relu2', nn.ReLU()),
    ('dropout2', nn.Dropout(dropout)),
    ('fc3', nn.Linear(fc2, 102)),
    ('output', nn.LogSoftmax(dim = 1))
    
    ]))
    
    model.classifier = imgClass
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    model.to(device);

    steps = 0
    print_every = 5
    training_loss = 0
    epochs = 1
    
    for epoch in range(epochs):
        for image, label in data:
            steps += 1
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()

            logout = model.forward(image)
            loss = criterion(logout, label)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for image, label in eval_data:
                        image, label = image.to(device), label.to(device)
                        logout = model.forward(image)
                        batch_loss = criterion(logout, label)

                        valid_loss += batch_loss.item()

                        #accuracy
                        prob = torch.exp(logout)
                        top_p, top_class = prob.topk(1, dim = 1)
                        equality = top_class == label.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                print(f'Epoch: {epoch+1}/{epochs}..  '
                      f'Training Loss: {training_loss/print_every:.4f}..  '
                      f'Validation Loss: {valid_loss/len(eval_data):.4f}.. '
                      f'Validation Accuracy: {accuracy/len(eval_data):.4f}')
            
            training_loss = 0
            model.train() 
            
    model.class_to_idx = mapping_data.class_to_idx
    torch.save({'epoch':epochs,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'class_to_idx': model.class_to_idx}, path)

def load_checkpoint(PATH = None, dropout = 0.2, device = 'cuda'):
    checkpoint = torch.load(PATH)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']    
   
    model, criterion, optimizer = loader_p(arquitecture = 'vgg19')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model.to('cuda')


def ratio_check(image):
    size = 256, 256
    left, top, right, bottom = 0, 0, 224, 224
    im = Image.open(image)
    width, height = im.size
    if width < height:
        ratio = float(height) / float(width)
        new_height = ratio * size[0]
        new_height = int(new_height)
        new_width = 256
        
    else:
        ratio = float(width) / float(height)
        new_width = ratio * size[0]
        new_width = int(new_width)
        new_height = 256
        
    return new_width, new_height, ratio

def process_image(image, width = None, height = None):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    size = int(width), int(height)
    left, top, right, bottom = 0, 0, 224, 224
    im = Image.open(image)
    pil_image = im.resize(size)
    pil_image = pil_image.crop((left, top, right, bottom))
    np_image = np.array(pil_image)
    np_image = np_image/255
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image-means)/std
    np_image = np_image.transpose((2,0,1))
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if title != None:
        plt.title(title)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(model = None, image_path = None, topk=5, mapping = None, width = None, height = None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    with torch.no_grad():
        width, height,_ = ratio_check(image_path)
        ima = process_image(image_path, width = width, height = height)
        img = torch.FloatTensor(ima)
        img = img.to('cuda')
        img = img.unsqueeze_(0)
        logps = model.forward(img)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(5, dim=1)
        top_p = top_p.cpu()
        top_class = top_class.cpu()
        top_p = top_p.detach().numpy().tolist()[0]
        top_class = top_class.detach().numpy().tolist()[0]
        idx_to_class = {str(value):int(key) for key, value in model.class_to_idx.items()}
        top_labels = [idx_to_class[str(lab)] for lab in top_class]
        flowers = [mapping[str(idx_to_class[str(lab)])] for lab in top_class]
        
    return top_p, top_labels, flowers

def display_prediction(image_path = None, model = None, mapping = None, prob = None, classes = None):
    flower_name = image_path.split('/')[6]
    title = mapping[flower_name]
    print('\n\nPREDICTION:' \
          f'\n\nOriginal: {title}' \
          f'\n\nTop Flower prediction: {classes[0]} ..  ' \
          f'Top Flower Probability: {prob[0]}' \
          f'\n\n2nd best Flower prediction: {classes[1]} ..  ' \
          f'2nd best Flower prediction: {prob[1]}')
          
  
    
    
   

                           
                      
      


    
    
    