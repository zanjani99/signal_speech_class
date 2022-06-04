

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
torch.manual_seed(1)
import librosa.display, librosa
import scipy
from scipy.linalg import solve_toeplitz, toeplitz
import soundfile as sf


def Autocor(signal, k):

    if k == 0:
        return np.sum(signal**2)
    else:
        return np.sum(signal[k:]*signal[:-k])


def Levinson(w_sig, p):
    r_list = [Autocor(w_sig, i) for i in range(p)]
    b_list = [Autocor(w_sig, i) for i in range(1, p+1)]
    LPC = solve_toeplitz((r_list, r_list), b_list)
    return LPC


def Center_Clip(signal, c_):

    CL = max(abs(signal))*c_

    for i in range(len(signal)):
        if abs(signal[i]) <= CL:
            signal[i] = 0
        elif signal[i] > CL:
            signal[i] -= CL
        elif signal[i] < -CL:
            signal[i] += CL

    return signal


def LPF(signal, target, sr):

    fft_signal = np.fft.rfft(signal)
    cutoff = int(np.ceil(target/sr*len(fft_signal)))
    fft_signal[cutoff:] = 0
    new_signal = np.fft.irfft(fft_signal)
    return new_signal

def Pitch_detect(signal, window, sr, overlap=0.5, center_clip=0.68, th_=0.55):

    win_len = len(window)
    overlap_length = int(len(window) * overlap)

    if len(signal) % overlap_length != 0:
        pad = np.zeros(overlap_length - (len(signal) %
                                         overlap_length))
        new_signal = np.append(signal, pad)
    else:
        new_signal = signal


    index = (len(new_signal) // overlap_length) - 1
    pitch_contour = np.zeros(index)

    for i in np.arange(index):
        # windowing
        frame = new_signal[(i*overlap_length)
                            :(i*overlap_length)+win_len] * window


        filtered_frame = LPF(frame, 900, sr)


        filtered_frame = Center_Clip(filtered_frame, center_clip)


        Rn = np.zeros(win_len)
        for k in range(win_len):
            Rn[k] = np.sum(filtered_frame[k:] * filtered_frame[:(win_len-k)])


        start = int(np.ceil(0.002*sr))
        end = int(np.ceil(0.02*sr))
        max_idx = Rn[start:end].argmax()
        threshold = Rn[0] * th_
        if Rn[max_idx + start] >= threshold:  # voice
            pitch = np.ceil(sr/(max_idx+start))
            pitch_contour[i] = pitch
        else:  # unvoiced
            pitch_contour[i] = 0

    return pitch_contour



def medianfilter(signal):
    med_sa = np.zeros((len(signal)+4))
    med_sa[2:-2] = signal  # zero-padding
    med_values = []
    for i in range(len(signal)):
        value = sorted(med_sa[i:i+5])[2]
        med_values.append(value)

    return med_values


def synthesis(signal,sr, window, p, overlap, c=0.68, th=0.3):
    """
    p: Order of the linear filter in LPC
    """
    length = len(window)

    '''padding'''
    shift = int(length*overlap)
    if len(signal) % shift != 0:
        pad = np.zeros(shift - (len(signal) % shift))
        new_signal = np.append(signal, pad)
    else:
        new_signal = signal

    index = [j*shift for j in range(len(new_signal)//shift-1)]

    '''pitch detection'''
    pitch_contour = medianfilter(Pitch_detect(
        new_signal, np.ones(length), sr, overlap=overlap, center_clip=c, th_=th))

    '''voiced region index'''
    voiced_idx = [shift*i for i in range(len(pitch_contour)) if pitch_contour[i]>50]

    syn_signal = np.zeros((len(new_signal)))
    prev_V = False

    '''make excitation'''
    for idx in index: #for each window
        w_sig = new_signal[idx:idx+len(window)]*window

        if idx in voiced_idx: #Voiced: impulse train
            F0 = pitch_contour[int(idx/shift)]
            period = np.ceil(sr/F0)

            excitation = np.zeros((length))


            if not prev_V:
                found = False
                for k in range(length):
                    if k % period == 0:
                        excitation[k] = 1 #impulse train
                        if not found and k > shift:

                            alignpoint = k-shift
                            found = True
                prev_V = True


            else:
                found = False
                for k in range(alignpoint,length):
                    if (k-alignpoint) % period ==0:
                        excitation[k] = 1
                        if not found and k > shift:
                            alignpoint = k-shift
                            found = True

        else: #Unvoiced: white noise
            sig_std = np.std(w_sig)
            excitation = np.random.normal(loc = 0.0, scale = sig_std, size = w_sig.shape)
            prev_V = False

        F_excitation = np.fft.fft(excitation, length)

        '''LPC spectrum'''
        LPC = Levinson(w_sig, p)
        LPC = np.insert(-LPC, 0, 1)
        w, h = scipy.signal.freqz([1],LPC,worN = length, whole = True, fs = sr)

        '''convolution in freq domain'''
        F_result = F_excitation*h
        result = np.fft.ifft(F_result, length)

        '''overlap-and-add'''
        syn_signal[idx:idx+length] += np.real(result)*np.hamming(length)

    return syn_signal








def visualize(signal,p):

    speech = synthesis(signal, 10000, np.hamming(256), p, 0.5, 0.6, 0.3)
    t_axis = np.linspace(0,len(speech)/sr,len(speech))

    plt.figure(figsize = (10,2))
    plt.grid(True)
    plt.xlabel('time (sec)')
    plt.title('Synthesized speech - LPC order: {}'.format(p))
    plt.plot(t_axis, speech)

    return speech



ys, sr = librosa.load('./Data/zanjani.wav',10000)
yonsei = ys[12000:24000]
t_axis = np.linspace(0,12,12000)
plt.figure(figsize = (10,2))
plt.xlabel('Time (sec)')
plt.title('Original Speech')
plt.ylabel('Amplitude')
plt.grid(True)
plt.plot(t_axis, yonsei)
plt.savefig("input_signal_Visualization.png")



pitch = medianfilter(Pitch_detect(yonsei, np.ones(512), 10000,0.5,0.68,th_ = 0.3))
plt.plot(pitch)
plt.title('Pitch')
plt.savefig("input_signal_pitch_detection.png")

sp_8 = visualize(yonsei, 8)

sf.write("output_8.wav",sp_8,samplerate=10000)




sp_32 = visualize(yonsei, 32)

sf.write("output_32.wav",sp_32,samplerate=10000)
