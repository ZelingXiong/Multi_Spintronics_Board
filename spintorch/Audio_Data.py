#!/usr/bin/env python
# coding: utf-8
# Base on: https://www.kaggle.com/code/maxwell110/beginner-s-guide-to-audio-data-2

# In[1]: 
import numpy as np
np.random.seed(1001)
import os
import shutil
import warnings
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms
from scipy.fft import fft, fftfreq
import IPython
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme('paper')
from tqdm import tqdm_notebook
from sklearn.model_selection import StratifiedKFold
import IPython.display as ipd 
import wave
matplotlib.style.use('ggplot')
warnings.filterwarnings("ignore", category=FutureWarning) 
from scipy.io import wavfile


def remove_silence(audio_data, threshold=0.01, min_silence_duration=100):
    non_silence_ranges = []
    current_range = None

    for i, amplitude in enumerate(audio_data):
        if np.abs(amplitude) > threshold:
            # print('Hello')
            if current_range is None:
                #print('1',current_range)
                current_range = [i, i]
            else:
                #print('2', current_range)
                current_range[1] = i
        else: # silent range
            if current_range is not None:
                if (current_range[1] - current_range[0] + 1) >= min_silence_duration:
                    # print('Hi')
                    non_silence_ranges.append(current_range)
                current_range = None

    if not non_silence_ranges:
        return np.array([])  # Return an empty array if no non-silence ranges are detected

    new_audio_data = np.concatenate([audio_data[start:end+1] for start, end in non_silence_ranges])
    return new_audio_data


def average_downsampling(audio_data,new_length = 1300):
    factor = len(audio_data)/new_length
    downsampled_audio = np.zeros(new_length)

    for i in range(new_length):
        start_idx = int(i * factor)
        end_idx = int((i + 1) * factor)
        downsampled_audio[i] = np.mean(audio_data[start_idx:end_idx])

    return downsampled_audio

def signal_smoothing(data):
    new_list = []
    for x in data:
        new_list.extend((x,x,x))
    return new_list


def amplitude_process(input_data, Bt):
    ratio = Bt/np.max(np.absolute(input_data))
    new_data = [num*ratio for num in input_data]
    return new_data

def data_for_spintronic(data, timesteps, Bt):
    original_audio = np.array(data)
    cleaned_audio = remove_silence(original_audio)
    downsampled_audio = average_downsampling(cleaned_audio,new_length = timesteps)
    amp_data = amplitude_process(downsampled_audio, Bt)
    return amp_data

def fourierTrans(dt,timesteps,y):
   SAMPLE_RATE = 1/dt
   DURATION = timesteps*dt
   # Number of samples in normalized_tone
   N = int(SAMPLE_RATE * DURATION)
   yf = fft(y)
   xf = fftfreq(N, 1 / SAMPLE_RATE)
   return np.abs(xf), np.abs(yf)

def max_finding(x_list,y_list):
    for i, x in enumerate (x_list):
        if y_list[i] == max(y_list):
            return x
        else:
            continue

def freq_shift(x_list, y_list, freq_peak):
    x_peak = max_finding(x_list, y_list)
    if x_peak ==0:
        ratio = 3e9
        new_x_list = [x + 3e9 for x in x_list]
    else:
        ratio = freq_peak/x_peak
        new_x_list = x_list*ratio
    return new_x_list

def inverse_fourier(num_samples, sampling_rate, amplitude, frequencies, shift_freq):
    time = np.arange(0, num_samples) / sampling_rate

    # Signal in the frequency domain with multiple frequency components
    signal_freq_domain = np.sum([amplitude * np.exp(2j * np.pi * freq * time) for freq in frequencies], axis=0)

    # Calculate the phase shift factors for each frequency component
    phase_shift_factors = np.exp(-2j * np.pi * (shift_freq - np.array(frequencies)) * time)

    # Apply the phase shift to each frequency component
    shifted_signal_freq_domain = signal_freq_domain * phase_shift_factors

    # Perform IFFT on the shifted signal to convert it back to the time domain
    shifted_signal_time_domain = np.fft.ifft(shifted_signal_freq_domain)

    return shifted_signal_time_domain



def data_extract(index_list, batch_size, timesteps, Bt, test_size, corresponding_names, plotdir, path):
    train = pd.read_csv(f'{path}/input/train_curated.csv')
    test = pd.read_csv(f'{path}/input/sample_submission.csv')

    train = train[train.labels.isin(test.columns[1:])]

    train['nframes'] = train['fname'].apply(lambda f: wave.open(f'{path}/input/train_curated/' + f).getnframes())

    LABELS = list(train.labels.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}
    # train.set_index("fname", inplace=True)
    # test.set_index("fname", inplace=True)
    train["label_idx"] = train.labels.apply(lambda x: label_idx[x])
    inputs_list = []
    outputs_list = []
    for i, Number in enumerate(index_list):
        new_dataset = train[train['label_idx'] == Number]
        #print(new_dataset)
        fname_list = new_dataset['fname'][:10]
        #print(fname_list)
        for fname_final in fname_list:
            print(fname_final)
            fname = f'{path}/input/train_curated/'+ fname_final
            wav = wave.open(fname)
            # print("Sampling (frame) rate = ", wav.getframerate())
            # print("Total samples (frames) = ", wav.getnframes())
            # print("Duration = ", wav.getnframes()/wav.getframerate())
            rate, raw_data = wavfile.read(fname)
            processed_data = data_for_spintronic(raw_data,timesteps,Bt)

            # # FFT
            # xf_raw,yf = fourierTrans(dt=20e-12, timesteps=len(processed_data), y=processed_data)
            # xf = [x/3 for x in xf_raw]
            # inverse_fft_raw = inverse_fourier(num_samples, sampling_rate, amplitude=yf, frequencies=xf, shift_freq=3.4e9)
            # inverse_fft = amplitude_process(inverse_fft_raw.real, Bt)
 
            zeros = [0]*(1300-len(processed_data))
            new_pro_data = np.concatenate((processed_data, zeros))

            inputs_list.append(new_pro_data)
            outputs_list.append(i)

    X_train, X_test, y_train, y_test = train_test_split(inputs_list, outputs_list, test_size=test_size, random_state=42)
    print(y_train)
    #combine the input and label 
    xtest_new = torch.FloatTensor(X_test)
    ytest_new = torch.FloatTensor(y_test)
    xtrain_new = torch.FloatTensor(X_train)
    ytrain_new = torch.FloatTensor(y_train)

    test_set = torch.utils.data.TensorDataset(xtest_new,ytest_new)
    train_set = torch.utils.data.TensorDataset(xtrain_new,ytrain_new)
    # train, val = torch.utils.data.random_split(train_all, [int(len(train_all)*0.7),len(train_all)-int(len(train_all)*0.7)])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                                                shuffle=True, num_workers=2)
    # val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, 
    #                                            shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, 
                                            shuffle=False, num_workers=2)

    print('-----Plot Training DATA-----')
    for BATCH, data in enumerate(train_loader, 0):
        print('BATCH: ', BATCH, 'Num of data within: ', len(data[0]))
        fig, axs = plt.subplots(2,5, figsize=(20, 6))
        axs = axs.flatten()
        for i, inputs_data in enumerate(data[0]):
            axs[i].plot(inputs_data, label = f' {corresponding_names[int(data[1][i])]}')
            # print(corresponding_names[int(data[1][i])])
            axs[i].legend()
            axs[i].set(title = f'label is {data[1][i]}')
            plt.tight_layout()
        plt.savefig(f'{plotdir}'+f'Training_Data_Batch{BATCH}.png', dpi=600)

    return train_loader, test_loader