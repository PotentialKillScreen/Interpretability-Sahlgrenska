import torch.nn as nn
from torchsummary import summary
from PIL import Image
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2


#The two immages that were tried

#img = np.load('C:/Users/AxelAhlqvist/Documents/Interpretability/data_vis2/positive/tensor_RF.npy')
img = np.load('C:/Users/AxelAhlqvist/Documents/Interpretability/data_vis2/positive/tensor_RF_2.npy')


class SimpleCNN2(nn.Module):
    def __init__(self, scalar_scale=1.0):
        super(SimpleCNN2, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 37 * 37, 128)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(128 + 1, 1)  # Output size will be 1 (scalar output)
        self.sigmoid = nn.Sigmoid()

        self.scalar_scale = scalar_scale

        #New code
        
        self.gradients = None
    def activations_hook(self,grad):
        self.gradients = grad
    
    def forward_with_hook(self, x, scalar_input):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        h = x.register_hook(self.activations_hook)

        x = self.maxpool3(x)
        x = x.view(x.size()[0], -1)

        x = self.fc1(x)
        x = self.relu4(x)

        scalar_input = torch.tensor([scalar_input], dtype=torch.float32, device=x.device)  # Convert scalar to tensor and match device
        scalar_input = scalar_input.unsqueeze(1).expand(x.shape[0], 1)
        scaled_scalar = self.scalar_scale * scalar_input

        x = torch.cat((x, scaled_scalar), dim=1)

        x = self.fc2(x)
        x = self.sigmoid(x)

        return x
    

    def forward_to_last_conv(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        return x

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):

        return self.features_conv(x)
    

    def forward(self, x, scalar_input):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = x.view(x.size()[0], -1) # Flatten the tensor
        x = self.fc1(x)
        x = self.relu4(x)
       
        # Scale the scalar input and concatenate it with the output of the previous layer
        scalar_input = torch.tensor([scalar_input], dtype=torch.float32, device=x.device)  # Convert scalar to tensor and match device
        scalar_input = scalar_input.unsqueeze(1).expand(x.shape[0], 1)
        scaled_scalar = self.scalar_scale * scalar_input
       
        x = torch.cat((x, scaled_scalar), dim=1)

        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

img = torch.tensor(img, dtype =torch.float)
img = torch.reshape(img, (1,3,300,300))

print("img.shape: ", img.shape)

model = SimpleCNN2()
model.load_state_dict(torch.load('C:/Users/AxelAhlqvist/Documents/Interpretability/model_weights.pth'))
model.eval()
pred = model.forward_with_hook(img, torch.tensor([10.0], dtype=torch.float))

print("pred: ", pred)




#backpropagation ---------------------------------------

pred.backward()
gradients = model.get_activations_gradient()
print("gradients: ", gradients)
pooled_gradients = torch.mean(gradients, dim = [0, 2,3])
print("pooled_gradients: ", pooled_gradients)

activations = model.forward_to_last_conv(img).detach()


#heatmap-----------------------------------------

for i in range(64):
     activations[:,i,:,:] *= pooled_gradients[i]
heatmap = torch.mean(activations,dim = 1).squeeze()

heatmap = np.maximum(heatmap,0)

heatmap /= torch.max(heatmap)

print("heatmap: ", heatmap)


heatmap = cv2.resize(np.array(heatmap), (300, 300))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
cv2.imshow("heatmap",heatmap)
cv2.waitKey(0)

cv2.imwrite('heatmap.png', heatmap)




