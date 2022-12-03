#intalling
#pip install-q -U pennylane datasets matplotlib tqdm torchvision torch gupload pillow
#pip install -q -U Pennylane Datasets Matplotlib tqdm TorchVision Gupload Pillow
#importing 
import pennylane as qml
from pennylane import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from pennylane.templates import AngleEmbedding
import torchvision
import torch
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
from matplotlib import pyplot as plt
from pennylane.optimize import NesterovMomentumOptimizer
#parameters 
num_qubits = 4
wires = range(num_qubits)
num_layers = 10
batch_size = 50
epochs = 200
num_classes = 4
dev = qml.device('default.qubit', wires=num_qubits)
keep_labels = [1, 4, 7, 9]
learners = [3]#,5,7,9,11,]

#data_setup - converting to torch and dividing into train and test data sets
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist', 
                                                          download=True, 
                                                          train=True,
                                                          
                                                          transform=transforms.Compose([
                                                              torchvision.transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,)),
                                                              transforms.Lambda(torch.flatten),
                                                          ])), 
                                           batch_size=10, 
                                           shuffle=True)
images = []
labels = []
for batch in train_loader:
  images.append(batch[0][0].numpy())
  labels.append(batch[1][0])
  

#print(images[:5])
#print(labels[:5])

test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist', 
                                                          download=True, 
                                                          train=False,
                                                          
                                                          transform=transforms.Compose([
                                                              torchvision.transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,)),
                                                              transforms.Lambda(torch.flatten) 
                                                          ])), 
                                           batch_size=100, 
                                           shuffle=True)

def to_onehot(label):
  if label == 1:
    return [1, 0, 0, 0]
  if label == 4:
    return [0, 1, 0, 0]
  if label == 7:
    return [0, 0, 1, 0]
  if label == 9:
    return [0, 0, 0, 1]
train_data = []
train_labels = []
test_data = []
test_labels = []
for (data, labels) in train_loader:
  for x, y in zip(data, labels):
    if y in keep_labels:
      train_data.append(x.numpy())
      train_labels.append(to_onehot(y.numpy()))
for (data, labels) in test_loader:
  for x, y in zip(data, labels):
    if y in keep_labels:
      test_data.append(x.numpy())
      test_labels.append(to_onehot(y.numpy()))
train_data = train_data[:150]
train_labels = train_labels[:150]
test_data = test_data[:150]
test_labels = test_labels[:150]


#PCA
def filter_out_data(images, labels, labels_to_keep=[1,2,3,4]):
  filtered_images = []
  filtered_labels = []
  for image, label in zip(images, labels):
    if int(label) in labels_to_keep:
      filtered_images.append(image)
      filtered_labels.append(label)
  return np.array(filtered_images, dtype=object), filtered_labels



filtered_images, filtered_labels = filter_out_data(images[:50], labels[:50])
#print(images[0])
x = [i.shape for i in filtered_images]
#print(x)
pca = PCA(16)
images = pca.fit_transform(images)



def scale_data(data, scale=[0, 1], dtype=np.float32):
    min_data, max_data = [float(np.min(data)), float(np.max(data))]
    min_scale, max_scale = [float(scale[0]), float(scale[1])]
    data = ((max_scale - min_scale) * (data - min_data) / (max_data - min_data)) + min_scale
    return data.astype(dtype)
    
def perform_pca(keep_digits=list(range(16)), pca_dims=16 ):
    pcas = []
    pca = PCA(pca_dims)
    pca.fit(x_train)
    approximation = pca.inverse_transform(pca.transform(x_train))


#defining_the_model
class Quilt2:
    dev = qml.device("default.qubit", wires)
    def __init__(self, name, weights=None):
      self.name = name
      self.weights = weights if weights is not None else 0.01 * np.random.randn(num_qubits, num_layers, 3)
      #self.opt = qml.MottenOd
      self.opt = qml.GradientDescentOptimizer(stepsize=0.4)
      self.losses = []
    def train(self, data, labels):
      for _ in range(epochs):
        train_indecies= np.random.randint(0, len(data), (batch_size,))
        x_train_batch, y_train_batch = [data[im] for im in train_indecies], [labels[im] for im in train_indecies]
        self.weights = self.opt.step(lambda v: self.cost(v, x_train_batch, y_train_batch), self.weights)
        np.save(f'weights_{self.name}', self.weights)
   
    def learn(rot, image): 
      #qml.Hadamard(wires=[0,1])
      qml.CNOT(wires=[0,1])
      if rot == "x,y": 
        qml.RX(image[0], wires=[0])
        qml.RY(image[1], wires=[1])
      elif rot == "x,z":
        qml.RX(image[0], wires=[0])
        qml.RZ(image[1], wires=[1])
      elif  rot == "y,x": 
        qml.RY(image[0], wires=[0])
        qml.RX(image[1], wires=[1])
      elif rot == "y,z":
        qml.RY(image[0], wires=[0])
        qml.RZ(image[1], wires=[1])
      elif rot == "z,x":
        qml.RZ(image[0], wires=[0])
        qml.RX(image[1], wires=[1])
      elif rot == "z,y":
        qml.RZ(image[0], wires=[0])
        qml.RY(image[1], wires=[1])
    

    @qml.qnode(dev)
    def circuit(self, weights, features=None):
        image = features=features.astype('float64')
        #top = [i[:2] for i in batch_size]
        for i in range(0,len(features),2): 
          AngleEmbedding(features=np.reshape(features[i:i+2], [-1]).astype('float64'), wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in wires]

        #learn("x,y", image)

    def eval(self, images, labels):
      bias = {0: 1, 1: 1, 2: 1, 3:1}
      guesses = []
      correct = 0
      for image, label in zip(images, labels):
        out = np.argmax(self.circuit(self, self.weights, image))
        guesses.append(out)
        bias[out] +=1
        if out == np.argmax(label):
          correct+=1
      bias_vector = [len(images)/bias[0], len(images)/bias[1], len(images)/bias[2], len(images)/bias[3]]
      print(f'Bias vector for {self.name} is {bias_vector}. Individual Accuracy is {correct/len(images)}')
      return 
    def cost(self, weights, features, labels):
      batch_loss = 0
      correct = 0
      for im, lab in zip(features, labels):
        y_hat = self.circuit(self, weights, im)
        batch_loss += sum(np.square(lab - y_hat))
        if np.argmax(y_hat) == np.argmax(lab):
          correct +=1
      self.losses.append(batch_loss)
      #if batch_size %10 == 0:
      print(f'Batch Loss: {batch_loss} Accuracy: {correct/len(features)}')
      return batch_loss.item()

#ensembles = ['Tirthak', 'Shane', 'Joan']#'Paul', 'Larry', 'Jordan', 'Sam']
ensembles = ['Tirthak','Shane','Joan']
for member in ensembles:
  ens = Quilt2('Tirthak')
  ens.train(train_data, train_labels)


correct = 0
ensembles = ['Tirthak', 'Shane', 'Joan']#, 'Paul', 'Larry', 'Jordan', 'Sam','vit']

guesses_scores = [{0: 0, 1: 0, 2:0, 3:0} for i in range(len(test_labels))]
#for member in ensembles:
ens = Quilt2('Tirthak', weights = np.load(f'weights_{member}.npy'))
guesses, bias_vector = ens.eval(test_data, test_labels)
for count, guess in enumerate(guesses):
  guesses_scores[count][guess] += 1 #bias_vector[guess]
print(guesses_scores[:10])
guesses = [max(g, key=g.get) for g in guesses_scores]
for i,j in zip(test_labels, guesses):
  if np.argmax(i) == j:
    correct +=1
print(f'Overall Accuracy: {correct/len(test_labels)}')
  



