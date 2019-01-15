#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Imports here
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models


# In[4]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[5]:


# TODO: Define your transforms for the training, validation, and testing sets
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
# TODO: Load the datasets with ImageFolder
# image_datasets = 
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size = 40, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size = 40, shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 40, shuffle = True)


# In[6]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# In[7]:


model = models.vgg19(pretrained = True)
model


# In[8]:


# Build and train your network
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
from torch import nn
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

imgClass = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, 1000)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(0.2)),
    ('fc2', nn.Linear(1000, 1000)),
    ('relu2', nn.ReLU()),
    ('dropout2', nn.Dropout(0.2)),
    ('fc3', nn.Linear(1000, 102)),
    ('output', nn.LogSoftmax(dim = 1))
    
]))

model.classifier = imgClass
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)

model.to(device);


# In[8]:


epochs = 6
steps = 0
training_loss = 0
print_every = 5


for epoch in range(epochs):
    for image, label in trainloader:
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
                for image, label in validloader:
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
                  f'Validation Loss: {valid_loss/len(validloader):.4f}.. '
                  f'Validation Accuracy: {accuracy/len(validloader):.4f}')
            
            training_loss = 0
            model.train() 
        


# In[17]:


# Do validation on the test set
epochs = 5
steps = 0
training_loss = 0
print_every = 5


for epoch in range(epochs):
    for image, label in trainloader:
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
                for image, label in testloader:
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
                  f'Test Loss: {valid_loss/len(testloader):.4f}.. '
                  f'Test Accuracy: {accuracy/len(testloader):.4f}')
            
            training_loss = 0
            model.train() 


# In[9]:


print("My model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())


# In[14]:


# Save the checkpoint 
model.class_to_idx = train_data.class_to_idx

torch.save({'epoch':epochs,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'class_to_idx': model.class_to_idx,
            'arch': 'vgg19'}, 'chkpoint.pth')


# In[9]:


state_dict = torch.load('chkpoint.pth')

print(state_dict.keys())


# In[10]:


# Write a function that loads a checkpoint and rebuilds the model
from collections import OrderedDict
from torch import nn
from torch import optim


def load_checkpoint(path_to_file):
    checkpoint = torch.load(path_to_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']    
    model.class_to_idx = checkpoint['class_to_idx']
    
    cpt = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, 1000)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(0.2)),
    ('fc2', nn.Linear(1000, 1000)),
    ('relu2', nn.ReLU()),
    ('dropout2', nn.Dropout(0.2)),
    ('fc3', nn.Linear(1000, 102)),
    ('output', nn.LogSoftmax(dim = 1))]))
    
    model.classifier = cpt
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.to('cuda')

# model = TheModelClass(*args, **kwargs)
# optimizer = TheOptimizerClass(*args, **kwargs)
# 
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']


# In[13]:





# In[11]:


model = load_checkpoint('chkpoint.pth')
print(model)


# In[12]:


import numpy as np
from PIL import Image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
       
    # transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # TODO: Process a PIL image for use in a PyTorch model
    size = 256, 256
    size2 = 224,224
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

    


# In[14]:


img = ('flowers/test/1/image_06743.jpg')
img = process_image(img)
print(img.shape)


# In[15]:




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


# In[16]:


image = 'flowers/test/1/image_06743.jpg'
imshow(process_image(image))


# In[17]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    with torch.no_grad():
        ima = process_image(image_path)
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
        flowers = [cat_to_name[str(idx_to_class[str(lab)])] for lab in top_class]
        
    return top_p, top_class, flowers


# In[18]:


image1 = 'flowers/test/1/image_06743.jpg'


# In[19]:




a,b,c = predict(image1, model)


print(b)
print(a)
print(cat_to_name[str(98)])
print(c[0])


# In[20]:




# ima = process_image(image)
# ima = torch.FloatTensor(ima)
# ima = ima.to('cuda')
# print(ima)



prob, classes, flowers= predict(image1, model)
imshow(process_image(image1))
print(prob)
print(classes) 
print(flowers)


# In[23]:


# Display an image along with the top 5 classes

import seaborn as sns
def display_prediction(image_path, model):
    plt.figure(figsize = (6,8))
    ax = plt.subplot(2,1,1)
    plt.axis('off')
    # Set up title
    flower_name = image_path.split('/')[2]
    title_ = cat_to_name[flower_name]
    # Plot flower
    img = process_image(image_path)
    imshow(img, ax, title = title_);
    # Make prediction
    ps, classes, flowers = predict(image_path, model) 
    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=ps, y=flowers, palette='rocket');
    plt.show()


# In[24]:


display_prediction(image1, model)


# In[75]:


cat_to_name

