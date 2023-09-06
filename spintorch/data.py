

# Generate Data
# Generate Continuous Souce Signal with Gaussian Noise
import torch
import torch.nn.functional as F 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import pickle
import seaborn as sns
sns.set_context('paper')


def zero(input,omega):
   #print(omega)
   if omega==0:
      return 0
   else:
      return input


def DataGenerate(a,b,dt,Bt,fbase,f_list,change_list,noise):
   'a,b are starting and ending time'
   x=np.arange(a,b,dt)
   y=list()
   omega_list = 2*np.pi*dt*fbase*np.array(f_list) #list of frequencies in radians per sample
   phi = 0; # phase accumulator
   count = int(0)
   # increment phase based on current frequency
   for j in range(len(omega_list)):
      for i in range(count,int(change_list[j]*len(x))):
         phi = phi + omega_list[j]
         c = Bt*np.sin(phi) # sine of current phase
         c = zero(c,omega_list[j])
         y.append(c)
         count+=1
   #Add Gaussian White Noise centered at 0
   if noise[0] == True:
      N_phi = np.random.normal(loc=0,scale=noise[1]*np.std(y),size=len(x))
      y += N_phi
   return x,y



def fourierTrans(dt,timesteps,y):
   from scipy.fft import fft, fftfreq
   SAMPLE_RATE = 1/dt
   DURATION = timesteps*dt
   # Number of samples in normalized_tone
   N = int(SAMPLE_RATE * DURATION)
   yf = fft(y)
   xf = fftfreq(N, 1 / SAMPLE_RATE)
   return np.abs(xf), np.abs(yf)


def data_plot(f_change_list,timesteps,dt,Bt,fbase,change_list,noise,labels,basedir):
   #----------ORIGINAL SIGNAL PLOT--------------#
   rcParams['figure.figsize'] = 12,4*len(f_change_list)
   fig, ax = plt.subplots(len(f_change_list),1,sharex=True,sharey=True)
   ax = ax.flatten()

   inputs_list = []
   outputs_list = []
   for i in range(len(f_change_list)):
      sns.lineplot(x=x, y=y, ax=ax[i], label=f'f={f_change_list[i]}GHz')
      ax[i].legend(title=f'signal{i}, label{labels[i]}',loc='upper right')
      ax[i].set(xlabel='Timesteps', ylabel='Magnitude')
      inputs_list.append(y)
      outputs_list.append(labels[i])
   plt.savefig(f'{basedir}/source_signal.png',dpi=300)

   #-----------FOURIER TANSFORM PLOT------------#
   rcParams['figure.figsize'] = 12,4*len(f_change_list)
   fig, ax = plt.subplots(len(f_change_list),1,sharex=True,sharey=True)
   ax = ax.flatten()
   for i in range(len(f_change_list)):
      x,y=DataGenerate(0, timesteps*dt,dt,Bt,fbase,f_change_list[i],change_list,noise)
      xf,yf = fourierTrans(dt,timesteps,y)
      sns.lineplot(x=xf, y=yf[:len(xf)], ax=ax[i], label=f'f={f_change_list[i]}GHz')
      #ax[i].plot(xf, yf,label=f'f={f_change_list[i]}GHz')
      ax[i].legend(title=f'signal{i} FFT, label{labels[i]}',loc='upper right')
      ax[i].set(xlabel='Frequency/Hz', ylabel='Magnitude')
   plt.savefig(f'{basedir}/source_FOURIER.png',dpi=300)

# print(np.array(inputs_list).shape)
# INPUTS_list = torch.tensor(inputs_list).unsqueeze(1).to(dev) 
# print(INPUTS_list.shape)
# INPUTS_list = torch.reshape(INPUTS_list,[len(f_change_list),timesteps,1])
# print(INPUTS_list.shape)
# OUTPUTS_list = torch.tensor(outputs_list)
# print(OUTPUTS_list.shape)

#Generate more samples
def sample_generation(data_size,f_change_list,timesteps,dt,Bt,fbase,change_list,noise,batch_size,labels,basedir, plot='False'):
   inputs_list = []
   outputs_list = []
   for j in range(data_size):
      for i in range(len(f_change_list)):
         x,y=DataGenerate(0, timesteps*dt,dt,Bt,fbase,f_change_list[i],change_list,noise)
         inputs_list.append(y)
         outputs_list.append(labels[i])

   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(inputs_list, outputs_list, test_size=0.5, random_state=42)
   # Data Processing
   import torchvision
   import torchvision.transforms as transforms
   #combine the input and label 
   xtest_new = torch.FloatTensor(X_test)
   ytest_new = torch.FloatTensor(y_test)
   xtrain_new = torch.FloatTensor(X_train)
   ytrain_new = torch.FloatTensor(y_train)

   test = torch.utils.data.TensorDataset(xtest_new,ytest_new)
   train = torch.utils.data.TensorDataset(xtrain_new,ytrain_new)
   # train, val = torch.utils.data.random_split(train_all, [int(len(train_all)*0.7),len(train_all)-int(len(train_all)*0.7)])

   train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, 
                                             shuffle=True)
   test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, 
                                          shuffle=False)
   #check the tensor size 
   print('check train_loader size')
   for BATCH, data in enumerate(train_loader, 0):
         print(BATCH)
         INPUTS, OUTPUTS = data
         INPUTS = torch.unsqueeze(INPUTS, -1)
         print(INPUTS.shape, OUTPUTS.shape)
   #Save the dataloader
   if not os.path.isdir(basedir):
      os.makedirs(basedir)

   torch.save(train_loader, f'{basedir}/train_loader.pth')
   torch.save(test_loader, f'{basedir}/test_loader.pth')

   pickle.dump(inputs_list, open(f'{basedir}/inputs_list.dat', 'wb'))
   pickle.dump(outputs_list, open(f'{basedir}/outputs_list.dat', 'wb'))

   if plot=='True':
      data_plot(f_change_list,timesteps,dt,Bt,fbase,change_list,noise,labels,basedir)

   return train_loader, test_loader