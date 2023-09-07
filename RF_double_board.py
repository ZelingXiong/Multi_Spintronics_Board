#!/usr/bin/env python
# coding: utf-8
# #  SpinTorch
# 1. Have continuous source signal with Gaussian Noise
# 2. Have 10%-50% threshold line for intensity plots
# 3. Added FFT for Gaussian Noise
# 4. Add Testing, training and validation data and accuracy test
# 5. Add KNN classifier after CNN to better distinguish between wanted and unwanted signals 
# 6. Try simulating drone signal processing by firstly only reacting to wanted signals

# V3: Limit linear weighting to even smaller ranges near 0.0002
# V4: change signal input range to [10,70]
# V5: increase probes on board 1 to increase number of hidden neurons for better performance
# In[1]:      Parameters to change
your_path = '/rds/general/user/zx719/home/3Layer_V2'
#your_path = '/home/zx719/Documents/MresMLBD/MResProject/Double_Board_V2'

#excitation field amplitude (T)
Bt = 1e-3 
Np = 2 #number of signal types
layer1_Np = 20 #number of probes on board 1
#base dirictiry of this model when training
value = Bt*1e3
basedir = f'HN{layer1_Np}_'+f'{value}mT'+f'_{Np}sig'
loader_dir = f'spintorch/V5_{Np}'+f'_sig_Test_{value}mT'
#select previous checkpoint (-1 = don't use checkpoint)
epoch = -1
#--------------------------------------#  Physical Parameters
geometry_type = 1 #Permanent magnets 
#["1 # Flip magnets", "2 #Permanent magnets (training multiplier)", "3 #Permanent magnets"] {type:"raw"}

dx = 50e-9          # discretization (m)
dy = 50e-9          # discretization (m)  
dz = 20e-9          # discretization (m)

#size x/y  (cells)
nx = 60    
ny = 80     

#saturation magnetization (A/m)
Ms = 140e3  
#bias field (T)
B0 = 60e-3    
 
#--------------------------------------#  Training Parameters
import torch
import torch.nn.functional as F 
import numpy as np
import pickle
import seaborn as sns
sns.set_theme('paper')
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR,ReduceLROnPlateau

learn_rate = 0.05       # Learning rate
epoch_max = 50        # number of training epoch
scaling = [1/5/10**(value), 1/4/10**(value)] #weighting for linear layer
print('scaling: ', scaling)
batch_size = 10 #number of data in one batch
data_size = 6 #number of total input data = data_size*Np
dt = 20e-12           # timestep (s) 400ps for spinwave to travel 120 nx
timesteps = nx*5 + 1000  # number of timesteps for wave propagation (nx*5 is the defult wave propergation time)
fbase = 1e9             # source frequency (Hz)xx x z
noise=[True,np.random.rand()*0.4]        # Background Gaussian Noise with mean 0 and sigma 0.1*(std(original signal))
A=3.5;B=3.0;C=3.3;D=3.7;E=4.0            # define A and B signal type
f_change_list_range= [[A,B,A,B,0],[B,E,B,E,0],[C,E,C,E,0],[D,E,D,E,0],[C,D,C,D,0],[A,D,A,D,0],[B,C,B,C,0],[B,A,B,A,0],[D,B,D,B,0],[C,A,C,A,0]] #ABAB would be our desired type
f_change_list = f_change_list_range[0:Np]
change_list = [0.1,0.2,0.3,0.4,1] #where we need to change the signal frequency
labels = np.arange(0,Np)
A=1*Bt; B=1*Bt #mT
Bt_change_list=[[A,B,A,B,A],[A,B,A,B,A]]
thresh_line = [True, True, True] #Now 10% 20% and 30% threshold line will be plotted on intensity output bar
optimizer_name = 'AdamW' #can be SGD,Adam,Adagrad,RMSprop,Adadelta ... (if changed, remember to change optimizer_params as well)
optimizer_params = {'weight_decay': 1e-4}  # extra parameter for the optimizer if needed, if not use '{}'.

fnc_used = 'defult'  # loss function 
apply_softmax = False      # `True` if the loss function only takes input in range [0,1]
# In[2]: Save info
import os
get_ipython().run_line_magic('cd', '$your_path')
    
import spintorch
import numpy as np
from spintorch.utils import tic, toc, stat_cuda
from spintorch.DB_plot import wave_integrated, wave_snapshot, Intensity_time
from spintorch.data import sample_generation
from spintorch.DB_solver import MyLinearLayer

import warnings
warnings.filterwarnings("ignore", message=".*Casting complex values to real.*")

if not os.path.isdir(basedir):
  os.makedirs(basedir)

plotdir = basedir + '/plots/' 
if not os.path.isdir(plotdir):
  os.makedirs(plotdir) 

savedir = basedir + '/models/'
if not os.path.isdir(savedir):
  os.makedirs(savedir) 
  
datadir = basedir + '/data/'
if not os.path.isdir(datadir):
  os.makedirs(datadir) 

analysisdir = basedir + '/analysis/'
if not os.path.isdir(analysisdir):
  os.makedirs(analysisdir) 

# In[3]: Geometry Setting
if geometry_type == 1 :
  Ms_CoPt = 723e3 # saturation magnetization of the nanomagnets (A/m) should be set to zero to turn magnets off
  r0, dr, dm, z_off = 15, 4, 2, 10  # starting pos, period, magnet size, z distance
  rx, ry = int((nx-2*r0)/dr), int((ny-2*r0)/dr+1)
  rho = torch.zeros((rx, ry))  # Design parameter array
  geom1 = spintorch.WaveGeometryArray(rho, (nx, ny), (dx, dy, dz), Ms, B0, r0, dr, dm, z_off, rx, ry, Ms_CoPt)
  geom2 = spintorch.WaveGeometryArray(rho, (nx, ny), (dx, dy, dz), Ms, B0, r0, dr, dm, z_off, rx, ry, Ms_CoPt)
elif geometry_type == 2 :
  B1 = 50e-3      # training field multiplier (T)
  geom1 = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms)
  geom2 = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms)
else:
  geom1 = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)
  geom2 = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)

src1 = spintorch.WaveLineSource(10, 0, 10, ny-1, dim=2)

src2 = [[]]*layer1_Np
for i in range(layer1_Np):
  src2[i] = spintorch.WaveLineSource(10, int(10+(ny-20)/layer1_Np*i), 10, int(10+(ny-20)/layer1_Np*(i+1)), dim=2)
  print(f'board 2 probe {i} position: ',10,  int(10+(ny-20)/layer1_Np*i), 10, int(10+(ny-20)/layer1_Np*(i+1)))
# WaveLineSource(x starting point, y starting point, x ending point, y ending point)

probes1 = [] #act as hidden neurons
for p in range(layer1_Np):
    probes1.append(spintorch.WaveIntensityProbeDisk(nx-15, int(ny*(p+1)/(layer1_Np+1)), 2))

probes2 = [] #act as output neurons
for p in range(Np):
    probes2.append(spintorch.WaveIntensityProbeDisk(nx-15, int(ny*(p+1)/(Np+1)), 2))


# Function for setting the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

# In[4]: Training Parameters
'''Define model'''
import torch.optim as optim
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        set_seed(42)
        super().__init__()
        
        '''' imput different layers '''
        self.conv1 = spintorch.DB_solver.MMSolver(geom1, dt, [src1], probes1)
        self.conv2 = spintorch.DB_solver.MMSolver(geom2, dt, src2, probes2)
        self.linear = MyLinearLayer(in_features = layer1_Np, out_features = layer1_Np, weight_range=scaling)

    def forward(self, x):
        #print('input', x)
        self.conv1.retain_history =  True #To save wave history
        x = self.conv1(x, False) #final_layer == False
        #print('after 1st conv2',x)
        #print('.pow(2)', x.pow(2))
        x = self.linear.forward(x.pow(2))
        #print('after linear', x)
        self.conv2.retain_history = True
        x = self.conv2(x, True) #final_layer==True
        #print('after 2nd conv2d', x)
        return x
    
    def layer(self,board_num):
      '''Return layer for wave history for plotting'''
      if board_num == 0:
        return self.conv1
      elif board_num == 1:
        return self.conv2
      elif board_num == 2:
        return self.linear

model = Net()
print(model)

if torch.cuda.is_available():
  dev = torch.device('cuda')  # 'cuda' or 'cpu'
else:
  dev = torch.device('cpu') 
print('Running on', dev)
model.to(dev)   # sending model to GPU/CPU2

'''Define optimizer and lossfunction'''

optimizer_class = getattr(optim, optimizer_name) # find the optimizer class in 'torch.optim' library
optimizer = optimizer_class(model.parameters(), lr=learn_rate, **optimizer_params)
'''Using ReduceLROnPlateau schedule'''
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.8, min_lr=1e-8)

# In[9]: Load the dataloader
train_loader, test_loader = sample_generation(data_size,f_change_list,timesteps,dt,Bt,fbase,change_list,noise,batch_size,labels,loader_dir, plot_sig=True)
# def itr_merge(*itrs):
#     'https://stackoverflow.com/questions/65621414/how-to-merge-two-torch-utils-data-dataloaders-with-a-single-operation'
#     for itr in itrs:
#         for v in itr:
#             yield v
# new_test = itr_merge(test_loader, val_loader)

# In[10]: Load checkpoint
epoch_init = epoch
if epoch_init>=0:
    print('load checkpoint')
    checkpoint = torch.load(savedir + 'model_e%d.pt' % (epoch_init))
    epoch = checkpoint['epoch']
    loss_iter = checkpoint['loss_iter']
    train_accu = checkpoint['train_accu']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    loss_iter = []
    train_accu = []

# In[12]: Loss Function


import torch.nn.functional as F
def loss_func(pred, labels,show=True,mean=True,min_camp=None,fnc_used='defult',apply_softmax=True):
  loss=[]
  for i in range(len(labels)): # repeat for each cases in a batch if batch size>1
    # print('int(labels[i])],pred[i]', [int(labels[i])],pred[i])
    target_value = pred[i][int(labels[i])] #int(labels[i]) for labels starting from 0; int(labels[i])-1 for labels starting from 1
    target_loss = pred[i:i+1].sum(dim=1)/target_value-1
    if min_camp == None or target_value>min_camp[i]:
      if fnc_used=='defult': # using defult my_loss function
        loss.append(((target_loss.sum()/target_loss.size()[0]).log10()).view(1))
      else:
        log_pred=(pred[i]).log10()
        if apply_softmax == True:
          log_pred=F.softmax(log_pred,dim=0) # if the loss function require utput in range [0,1], apply softmax to it
        loss.append(fnc_used(log_pred,labels[i]).view(1))
    else:
      loss.append(torch.tensor([0]))
  loss=torch.cat(loss)

  #loss = torch.clamp(loss, min=1e-5, max=1e5)
  [print("LOSS:",loss)if show else None]
  if mean == True:  # used in training weights
    return torch.mean(loss) 
  else:             # used in analysing the outputs 
    return loss  

def my_loss(pred, labels,show=True,mean=True,min_camp=None):  # Old(classic) loss function 
  loss=[]
  for i in range(len(labels)): # repeat for each cases in a batch if batch size>1
    target_value = pred[i][int(labels[i])]
    target_loss = pred[i:i+1].sum(dim=1)/target_value-1
    if min_camp == None or target_value>min_camp[i]:
      loss.append(((target_loss.sum()/target_loss.size()[0]).log10()).view(1))
    else:
      loss.append(torch.tensor([0]))
  loss=torch.cat(loss)
  #loss = torch.clamp(loss, min=1e-5, max=1e5)
  [print("LOSS:",loss)if show else None]
  if mean == True:  # used in training 
    return torch.mean(loss) 
  else:             # used in analysing the outputs 
    return loss 


# In[14]: Training 
get_ipython().run_line_magic('cd', '$your_path')
tic()
lr_list = []
model.retain_history = True

for epoch in range(epoch_init+1, epoch_max):
  avg_loss = 0.
  correct = 0
  total = 0
  model.layer(0).m_history = []
  model.layer(1).m_history = []
  model.layer(2).m_history = []
  print(f'----------------------------- EPOCH {epoch+1}/{epoch_max} -----------------------------')
  for BATCH, data in enumerate(train_loader, 0):
    print(f'BATCH {BATCH+1}/{len(train_loader)}')
    INPUTS, OUTPUTS = data
    INPUTS = torch.unsqueeze(INPUTS, -1)
    print('correct results',OUTPUTS)
    print('inputs shape', INPUTS.shape)
    #FORWARD
    optimizer.zero_grad()
    model_out_list = model(INPUTS)
    u = model_out_list.sum(dim=1)
    print('model outputs sum u:',u)
    #SAVE DATA & PLOT
    pickle.dump(model_out_list, open(f'{datadir}'+'model_out_list_epoch{}_batch{}.dat'.format(epoch,BATCH), 'wb'))
    pickle.dump(OUTPUTS, open(f'{datadir}'+'OUTPUTS_epoch{}_batch{}.dat'.format(epoch,BATCH), 'wb'))

    #LOSS CALCULATION + BACKWARD + OPTIMIZE
    loss = loss_func(u, OUTPUTS,show=True,mean=True,fnc_used=fnc_used,apply_softmax=False)
    all_output = {'model_out_list':model_out_list,'OUTPUTS': OUTPUTS, 'loss': loss}
    torch.save(all_output, f'{savedir}/outputs&loss_epoch{epoch}_batch{BATCH}.dat')
    avg_loss += loss.item()/len(train_loader)
    
    stat_cuda('after forward')
    loss.backward()
    optimizer.step()
    stat_cuda('after backward')

    #ACCURACY TEST
    _, predicted = torch.max(u.data,1)
    print('predictions: ', predicted)
    preds=[]
    for i, value in enumerate(predicted.tolist()):
      preds.append(value)
    total += len(OUTPUTS)
    correct += sum(1 for a, b in zip(preds, OUTPUTS) if a == b)
    train_accuracy = correct / total 
    print("batch %d finished: -- Running Loss: %.4f -- Training Accuracy: %4f" % (BATCH+1, loss, train_accuracy))
    toc() 

  del predicted, preds, total, correct    

  scheduler.step(avg_loss)
  lr_list.append(optimizer.param_groups[0]['lr'])
  loss_iter.append(avg_loss)
  train_accu.append(train_accuracy)
  spintorch.DB_plot.plot_loss(loss_iter, analysisdir)
  spintorch.DB_plot.plot_accuracy(train_accu, analysisdir)
  
  #print("epoch %d finished: -- Average Loss: %.4f" % (epoch, avg_loss))
  print('Epoch {}/{} \t average_train_loss={:.4f} \t train_accuracy={:.4f} \t '.format(
      epoch + 1, epoch_max, avg_loss, train_accuracy))

  '''Save model checkpoint'''
  torch.save({
            'epoch': epoch,
            'loss_iter': loss_iter,
            'train_accu': train_accu,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, savedir + 'model_e%d.pt' % (epoch))

  '''Plot spin-wave propagation every 5 epochs '''
  if epoch % 10 == 9 or epoch == 0:
    # plot output bars
    for i in range(len(u)):   
      spintorch.DB_plot.plot_output(u[i:i+1][0,],OUTPUTS[i].numpy(),i,BATCH,epoch, plotdir,thresh_line)
    # geometry and wave propagation plots
    if model.retain_history:  
      with torch.no_grad():
        layer1 = model.layer(0)
        layer2 = model.layer(1)
        spintorch.DB_plot.geometry([layer1,layer2], epoch=epoch, plotdir=plotdir)
        mz1 = torch.stack(layer1.m_history, 1)[0,:,2,]-layer1.m0[0,2,].unsqueeze(0).cpu()
        mz2 = torch.stack(layer2.m_history, 1)[0,:,2,]-layer2.m0[0,2,].unsqueeze(0).cpu()
        mz = [mz1[timesteps-1],mz2[timesteps-1]]
        j = len(u)-1
        wave_snapshot([layer1,layer2], mz, (plotdir+'snapshot_time%d_epoch%d_X%d_label%d.png' % (timesteps,epoch,j,OUTPUTS[j])),r"$m_z$")
        wave_snapshot([layer1,layer2], [mz1[int(timesteps/2-1)],mz2[int(timesteps/2-1)]], (plotdir+'snapshot_time%d_epoch%dX%d_label%d.png' % (int(timesteps/2),epoch,j,OUTPUTS[j])),r"$m_z$")
        wave_integrated([layer1,layer2], [mz1, mz2], (plotdir+'integrated_epoch%dX%d.png' % (epoch,j)))
        Intensity_time(model_out_list[-1].detach().numpy(),OUTPUTS[-1],epoch, Np, plotdir)
        

np.savetxt(f'{datadir}'+'LRlist.dat', lr_list)
np.savetxt(f'{datadir}'+'train_acu_list.dat', train_accu)
np.savetxt(f'{datadir}'+'train_loss_list.dat', loss_iter)

# In[ ]: K Clustering 
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

testdir = basedir + '/testing/' 
if not os.path.isdir(testdir):
  os.makedirs(testdir)

#%% load more data to ensure enough training data for knn
u_knn=[]
outputs_list=[]
epoch=epoch_max-1
for BATCH, data in enumerate(train_loader, 0):
  model_out_list = pickle.load(open(f'{datadir}'+'model_out_list_epoch{}_batch{}.dat'.format(epoch,BATCH), 'rb'))
  temp = model_out_list.sum(dim=1).detach().numpy()
  for i in range(len(temp)):
    u_knn.append(temp[i])
  OUTPUTS = pickle.load(open(f'{datadir}'+'OUTPUTS_epoch{}_batch{}.dat'.format(epoch,BATCH), 'rb'))
  temp2 = OUTPUTS.detach().numpy()
  for j in range(len(temp2)):
    outputs_list.append(temp2[j])

knn_classifier = KNeighborsClassifier(n_neighbors=len(u_knn),weights='distance')
knn_regressor = KNeighborsRegressor(n_neighbors=len(u_knn),weights='distance')
knn_classifier.fit(u_knn,outputs_list)
knn_regressor.fit(u_knn,outputs_list)

#save trained knn model
knnPickle = open(f'{testdir}/knnclassifier_file', 'wb') 
pickle.dump(knn_classifier, knnPickle)    
knnPickle.close() 

knnPickle = open(f'{testdir}/knnregressor_file', 'wb') 
pickle.dump(knn_regressor, knnPickle)   
knnPickle.close() 

# In[ ]: Testing 
# load the knn model from disk
# knn_classifier = pickle.load(open(f'{testdir}/knnclassifier_file', 'rb'))
# knn_regressor = pickle.load(open(f'{testdir}/knnregressor_file', 'rb'))

avg_loss = 0.
correct = 0
knn_correct = 0
total = 0
print(f'----------------------------- TESTING -----------------------------')
for BATCH, data in enumerate(test_loader, 0):
    print(f'BATCH {BATCH}/{len(test_loader)}')
    INPUTS_test, OUTPUTS_test = data
    INPUTS_test = torch.unsqueeze(INPUTS_test, -1)
    #FORWARD
    optimizer.zero_grad()
    model_out_list = model(INPUTS_test)
    u_test = model_out_list.sum(dim=1)
    #Add KNN Classifier
    kcpred = knn_classifier.predict(u_test.detach().numpy()) #make prediction on test set
    kgpred = knn_regressor.predict(u_test.detach().numpy()) #make prediction on test set
    #kcerror = np.sqrt(mean_squared_error(OUTPUTS_test.detach().numpy(),kcpred)) #calculate rmse
    print('OUTPUTS_test',OUTPUTS_test)
    print('knn classifier prediction',kcpred)
    print('knn regressor prediction',kgpred)

    for i in range(len(u_test)):
        spintorch.DB_plot.plot_output(u_test[i:i+1][0,],OUTPUTS_test[i].numpy(),i,BATCH,0, testdir,thresh_line)

    #LOSS CALCULATION
    loss = loss_func(u_test, OUTPUTS_test,show=True,mean=True,fnc_used=fnc_used,apply_softmax=False)
    #loss=my_loss(u, OUTPUTS,show=True,mean=True)
    avg_loss += loss.item()/len(test_loader)


    #ACCURACY TEST
    _, predicted = torch.max(u_test.data, 1)
    print('Normal Prediction', predicted)
    preds=[]
    for i, value in enumerate(predicted.tolist()):
        preds.append(value)
    total += len(OUTPUTS_test)
    correct += sum(1 for a, b in zip(preds, OUTPUTS_test) if a == b)
    test_accuracy = correct / total 

    #KNN Accuracy
    knn_correct += sum(1 for a, b in zip(kcpred, OUTPUTS_test) if a == b)
    knn_accuracy = knn_correct / total 

    all_output = {'model_out_list':model_out_list,'OUTPUTS': OUTPUTS_test, 'loss': loss,'Normal_Prediction': predicted, 'kcpred':kcpred,'kgpred':kgpred, 'knn_accuracy':knn_accuracy}
    torch.save(all_output, f'{testdir}/outputs&loss_batch{BATCH}.dat')
    print("batch %d finished: -- Running Loss: %.4f -- Testing Accuracy: %4f -- Knn Accuracy: %4f " % (BATCH, loss, test_accuracy, knn_accuracy))

      
#print("epoch %d finished: -- Average Loss: %.4f" % (epoch, avg_loss))
print('TESTING \t average_testing_loss={:.4f} \t testing_accuracy={:.4f} \t knn_accuracy={:.4f}'.format(avg_loss, test_accuracy, knn_accuracy))

#%%
'''Plot spin-wave propagation for the last batch'''
if model.retain_history:
    with torch.no_grad():
        for j in range(len(u_test)):
            layer1 = model.layer(0)
            layer2 = model.layer(1)
            spintorch.DB_plot.geometry([layer1,layer2], epoch=0, plotdir=testdir)
            mz1 = torch.stack(layer1.m_history, 1)[0,:,2,]-layer1.m0[0,2,].unsqueeze(0).cpu()
            mz2 = torch.stack(layer2.m_history, 1)[0,:,2,]-layer2.m0[0,2,].unsqueeze(0).cpu()
            mz = [mz1[timesteps-1],mz2[timesteps-1]]
            wave_snapshot([layer1,layer2], mz, (testdir+'TEST_snapshot_time%d_X%d_label%d.png' % (timesteps,j,OUTPUTS_test[j])),r"$m_z$")
            wave_snapshot([layer1,layer2], [mz1[int(timesteps/2-1)],mz2[int(timesteps/2-1)]], (testdir+'TEST_snapshot_time%d_X%d_label%d.png' % (int(timesteps/2),j,OUTPUTS_test[j])),r"$m_z$")
            wave_integrated([layer1,layer2], [mz1, mz2], (testdir+'TEST_integrated_X%d.png' % (j)))




