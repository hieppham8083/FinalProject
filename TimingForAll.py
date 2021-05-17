#!/usr/bin/env python
# coding: utf-8

# In[375]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv
import os
import sys
import glob
import os.path
from pathlib import Path
import fnmatch
from simple_colors import * # pip install simple-colors
import time
import math
import random
from itertools import cycle
from datetime import datetime
from scipy.fft import fft, fftfreq
from spectrum import *
from pylab import *
import spectrum 
from spectrum import tools
from numpy import fft
from scipy import signal
from matplotlib.backends.backend_pdf import PdfPages
import warnings #fixed any warning in terminal
import matplotlib.cbook
# Ignore DtypeWarnings from pandas' read_csv
warnings.filterwarnings('ignore', message="^Columns.*")
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


# In[ ]:


def zoomin():
    plt.close()
    plt.figure(figsize=(11,6)) #all in one figure(1)
    if folder_name.find('Spectrum') != -1 or folder_name.find('Noise') != -1:
        lst2 = [float(y) for y in input("Please input zoom in Y_axis(min, max): ").split()] 
        print(f"Zoom in view Y_axis: {lst2}")
        while len(lst2) == 1 or  len(lst2) > 2:
            print("Typo! Please try again.")
            lst2 = [float(y) for y in input("Re-enter input zoom in Y_axis(min, max): ").split()]  
    else:
        lst = [float(x) for x in input("Please input zoom in X_axis(min, max): ").split()]
        print(f"Zoom in view X_axis: {lst}")
        while len(lst) == 1 or  len(lst) > 2:
            print("Typo! Please try again.")
            lst = [float(x) for x in input("Re-enter input zoom in X_axis(min, max): ").split()]        
    for csv_file in sorted(dirs): 
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        #plt.figure(figsize=(11,6)) #multiple figure(1)
        file_name, file_extension = os.path.splitext(csv_file)
        test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
        #test_data_df = pd.read_csv(test_data_to_load) #no skip row
        #test_data_df = test_data_df[249000:-2237000] ##(equal x_axis * 1000),cut dataframe
        test_data_df["Time"] = test_data_df["Time"] * 1000
        plt.plot(test_data_df["Time"], test_data_df["Ampl"]) 
        plt.plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name) 
        plt.xlabel("Time(ms)")
        plt.ylabel("Voltage(V)")
        try:
            if len(lst2) > 1: 
                plt.ylim(lst2)
        except:
            pass
        try:
            if len(lst) > 1: 
                plt.xlim(lst)
        except:
            pass  
        #plt.xlim([-30, 300])
        #plt.ylim([-15, 25])
        #plt.title(path + file_name) #multiple figure(2)
        plt.title(folder_name) #single figure(2)
        plt.xticks(fontsize=10)
        #plt.legend() #label=file_name
        plt.legend(loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0) #legend outside
        plt.grid(True)
        plt.subplots_adjust(right=.8) 
        #plt.savefig('Output.pdf', dpi=300, bbox_inches='tight') #single figure(3)
    plt.show()
    #plt.close()


# In[376]:


def rise_time():
    if mean > 0:
        a_df = test_data_df[test_data_df["Ampl"] > (max*90/100)]  
        a_df = a_df.iloc[[1]]
        if folder_name == "DigitalPowerOn":
            b_df = test_data_df[test_data_df["Ampl"] > (max*30/100)]
        else:
            b_df = test_data_df[test_data_df["Ampl"] > (max*10/100)]
        b_df = b_df.iloc[[1]]
        if file_name == "pp3v3" or file_name == "3v3":
            signal_x1 = float(a_df.Time) - float(a_df.Time)
        else:
            signal_x1 = float(a_df.Time)
        signal_x2 = float(b_df.Time)
        return (signal_x1, signal_x1 - signal_x2)
    if mean < 0:
        a_df = test_data_df[test_data_df["Ampl"] > (min*90/100)]  
        a_df = a_df.iloc[[-1]]
        b_df = test_data_df[test_data_df["Ampl"] > (min*10/100)]  
        b_df = b_df.iloc[[-1]]
        signal_x1 = float(a_df.Time)
        signal_x2 = float(b_df.Time)
        return (signal_x1, signal_x1 - signal_x2)
        


# In[377]:


def fall_time():
    if mean > 0:
        if file_name == "vghboost":
            signal_x1 = 0
            signal_x2 = 0
            return (signal_x1, abs(signal_x1 - signal_x2))
        elif file_name == "vgh1":
            a_df = test_data_df[test_data_df["Ampl"] > (max*15/100)]  
            a_df = a_df.iloc[[-1]]
            b_df = test_data_df[test_data_df["Ampl"] > (max*85/100)]
            b_df = b_df.iloc[[-1]]
            signal_x1 = float(a_df.Time)
            signal_x2 = float(b_df.Time)
            return (signal_x1, abs(signal_x1 - signal_x2))
        else:
            a_df = test_data_df[test_data_df["Ampl"] < (max*15/100)]  
            a_df = a_df.iloc[[1]]
        if folder_name == "DigitalPowerOff":
            b_df = test_data_df[test_data_df["Ampl"] < (max*30/100)]
            b_df = b_df.iloc[[1]]
        else:
            b_df = test_data_df[test_data_df["Ampl"] < (max*85/100)]
            b_df = b_df.iloc[[1]]
        if file_name == "pp3v3" or file_name == "edp":
            signal_x1 = float(a_df.Time) - float(a_df.Time)
        else:
            signal_x1 = float(a_df.Time)
        signal_x2 = float(b_df.Time)
        return (signal_x1, abs(signal_x1 - signal_x2))
    if mean < 0:
        a_df = test_data_df[test_data_df["Ampl"] < (min*15/100)]  
        a_df = a_df.iloc[[-1]]
        b_df = test_data_df[test_data_df["Ampl"] < (min*85/100)]  
        b_df = b_df.iloc[[-1]]
        signal_x1 = float(a_df.Time)
        signal_x2 = float(b_df.Time)
        return (signal_x1, abs(signal_x1 - signal_x2))


# In[378]:


def plot():
    plt.plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name)
    plt.xlabel("Time(ms)")
    plt.ylabel("Voltage(V)")


# In[379]:


def plot_tek():
    
    test_data_df = pd.read_csv(test_data_to_load, skiprows=11) 
    test_data_df.dropna(how='all', axis=1, inplace=True) #drop empty column
    #test_data_df = test_data_df.T.drop_duplicates().T #drop duplicates column
    test_data_df["TIME"] = test_data_df["TIME"] * 1000 
    for i in test_data_df.columns:
        if i not in ("TIME", "TIME.1"):
            plt.plot(test_data_df["TIME"],test_data_df[i], label=i)
    plt.xlabel("Time(ms)")
    plt.ylabel("Voltage(V)")
  


# In[380]:


def plot_gatedriver():
    plt.plot(test_data_df["Time"], test_data_df["Ampl"], label=f"VST1 To {file_name}: {time:.1f}(us)") 
    plt.xlabel("Time(us)")
    plt.ylabel("Voltage(V)")


# In[381]:


def plot_time():
    try:
        if (folder_name.find('Digital') != -1):
            plt.plot(test_data_df["Time"], test_data_df["Ampl"], label=f"Trigger To {file_name}: {time:.3f}(ms)")
        else:
            plt.plot(test_data_df["Time"], test_data_df["Ampl"], label=f"Trigger To {file_name}: {time:.0f}(ms)")    
    except:
        pass
    plt.xlabel("Time(ms)")
    plt.ylabel("Voltage(V)")


# In[382]:


def plot_rise():
    if mean > 0 and (folder_name.find('Digital') != -1):  
        plt.plot(test_data_df["Time"],test_data_df["Ampl"],label=f"Trigger To {file_name}: {time:.2f}(ms), & Rise time: {edge:.3f}(ms)")
    elif mean > 0:
         plt.plot(test_data_df["Time"],test_data_df["Ampl"],label=f"Trigger To {file_name}: {time:.0f}(ms), & Rise time: {edge:.1f}(ms)")
    else:
        plt.plot(test_data_df["Time"],test_data_df["Ampl"],label=f"Trigger To {file_name}: {time:.0f}(ms), & Fall time: {edge:.1f}(ms)")
    plt.xlabel("Time(ms)")
    plt.ylabel("Voltage(V)")


# In[383]:


def plot_fall():
    if (folder_name.find('Digital') != -1):  
        plt.plot(test_data_df["Time"],test_data_df["Ampl"],label=f"Trigger To {file_name}: {time:.2f}(ms), & fall time: {edge:.3f}(ms)")
    elif mean < 0:
         plt.plot(test_data_df["Time"],test_data_df["Ampl"],label=f"Trigger To {file_name}: {time:.0f}(ms), & Rise time: {edge:.1f}(ms)")
    else:
        plt.plot(test_data_df["Time"],test_data_df["Ampl"],label=f"Trigger To {file_name}: {time:.0f}(ms), & Fall time: {edge:.1f}(ms)")
    plt.xlabel("Time(ms)")
    plt.ylabel("Voltage(V)")


# In[384]:


def two_signals_on():
    time1 = 0
    offset = 0
    plt.close()
    signal1, voltage_level1 = input("Enter first signal name and voltage level: ").split()
    file_name1 = signal1 + '.csv'
    while file_name1 not in dirs:
        print("Typo! Please try again.")
        signal1, voltage_level1 = input("Re-enter first signal name and voltage level: ").split()
        file_name1 = signal1 + '.csv'
    signal2, voltage_level2 = input("Enter second signal name and voltage level: ").split()
    file_name2 = signal2 + '.csv'
    while file_name2 not in dirs:
        print("Typo! Please try again.")
        signal2, voltage_level2 = input("Re-enter second signal name and voltage level: ").split()
        file_name2 = signal2 + '.csv'
    test_data_to_load = os.path.join(path,file_name1)
    test2_data_to_load = os.path.join(path,file_name2)
    voltage_level1 = float(voltage_level1)
    voltage_level2 = float(voltage_level2)
    plt.figure(figsize=(9,4)) #all in one figure(1)
    test_data_df = pd.read_csv(test_data_to_load,skiprows=4)
    test2_data_df = pd.read_csv(test2_data_to_load,skiprows=4)
    file_name1, file_extension = os.path.splitext(file_name1)
    file_name2, file_extension = os.path.splitext(file_name2)
    if file_name1 not in ("C1_vst1", "C2_vst2", "clk1", "clk2", "clk3", "clk4", "clk5", "clk6", "clk7", "clk8" ):
        test_data_df["Time"] = test_data_df["Time"] * 1000
        test2_data_df["Time"] = test2_data_df["Time"] * 1000
    else:
        if file_name1 == "clk1" or file_name2 == "clk1":
            test_data_df = test_data_df[9930:-89600]# 1M sample rate
            test2_data_df = test2_data_df[9930:-89600] #1M sample rate
        else:
            test_data_df = test_data_df[9930:-89500]# 1M sample rate
            test2_data_df = test2_data_df[9930:-89500] #1M sample rate
        test_data_df["Time"] = test_data_df["Time"] * 1000000 #GateDriver
        test2_data_df["Time"] = test2_data_df["Time"] * 1000000 #GateDriver
        trigger_fall = test_data_df[test_data_df["Ampl"] > voltage_level1] #rising
        trigger_fall = trigger_fall.iloc[[-1]] #-1 is last number
        trigger_xf = float(trigger_fall.Time)
        
    if voltage_level1 > 0 and voltage_level2 > 0: #High To High
        trigger_signal = test_data_df[test_data_df["Ampl"] > voltage_level1] #rising
        trigger_signal = trigger_signal.iloc[[1]] #-1 is last number
        trigger_x = float(trigger_signal.Time)
        trigger2_signal = test2_data_df[test2_data_df["Ampl"] > voltage_level2] #rising
        trigger2_signal = trigger2_signal.iloc[[1]] #-1 is last number
        trigger2_x = float(trigger2_signal.Time)
        delay = abs(trigger2_x - trigger_x)
    elif voltage_level1 > 0 and voltage_level2 < 0:
        trigger_signal = test_data_df[test_data_df["Ampl"] > voltage_level1] #rising
        trigger_signal = trigger_signal.iloc[[1]] #-1 is last number
        trigger_x = float(trigger_signal.Time)
        trigger2_signal = test2_data_df[test2_data_df["Ampl"] > voltage_level2] #rising
        trigger2_signal = trigger2_signal.iloc[[-1]] #-1 is last number
        trigger2_x = float(trigger2_signal.Time)
        delay = abs(trigger2_x - trigger_x)
    elif voltage_level1 < 0 and voltage_level2 < 0:
        trigger_signal = test_data_df[test_data_df["Ampl"] > voltage_level1] #rising
        trigger_signal = trigger_signal.iloc[[-1]] #-1 is last number
        trigger_x = float(trigger_signal.Time)
        trigger2_signal = test2_data_df[test2_data_df["Ampl"] > voltage_level2] #rising
        trigger2_signal = trigger2_signal.iloc[[-1]] #-1 is last number
        trigger2_x = float(trigger2_signal.Time)
        delay = abs(trigger2_x - trigger_x)
        print(delay)
    elif voltage_level1 < 0 and voltage_level2 > 0:
        trigger_signal = test_data_df[test_data_df["Ampl"] > voltage_level1] #rising
        trigger_signal = trigger_signal.iloc[[-1]] #-1 is last number
        trigger_x = float(trigger_signal.Time)
        trigger2_signal = test2_data_df[test2_data_df["Ampl"] > voltage_level2] #rising
        trigger2_signal = trigger2_signal.iloc[[1]] #-1 is last number
        trigger2_x = float(trigger2_signal.Time)
        delay = abs(trigger2_x - trigger_x)
        print(delay)
    plt.plot(test_data_df["Time"], test_data_df["Ampl"], label=f"{file_name1}") 
    plt.plot(test2_data_df["Time"], test2_data_df["Ampl"], label=f"{file_name2}")
    #plt.plot(test_data_df["Time"],test_data_df["Ampl"],label=f"{file_name} => DelayTime={time:.0f}mS)") 
    if file_name1 not in ("C1_vst1", "C2_vst2", "clk1", "clk2", "clk3", "clk4", "clk5", "clk6", "clk7", "clk8" ):
        plt.xlabel("Time(ms)")
        plt.ylabel("Voltage(V)")
        plt.title(r'DelayTime betwwen 2 signals $is$: $\bf{{{a}}}(ms)$'.format(a=int(delay)))
        plt.annotate("", xy=(trigger_x, voltage_level1), xytext=(trigger2_x, voltage_level2), arrowprops=dict(arrowstyle="<->"))
    else:
        plt.xlabel("Time(us)")
        #plt.title(r'DelayTime betwwen 2 signals $is$: $\bf{{{a}}}(us)$'.format(a=int(delay)))
        plt.ylabel("Voltage(V)")
        plt.annotate("", xy=(trigger_x, voltage_level1), xytext=(trigger2_x, voltage_level2), arrowprops=dict(arrowstyle="<->"))
        plt.annotate(f"{delay:.0f}", (trigger_x,voltage_level1+1), fontsize=11, color="red")
        if file_name1 not in ("clk6","clk7", "clk8"):
            plt.annotate("", xy=(trigger_xf, voltage_level1-2), xytext=(trigger2_x, voltage_level2-2), arrowprops=dict(arrowstyle="<->"))
            plt.annotate(f"{(trigger2_x-trigger_xf):.0f}", (trigger2_x,voltage_level2-4), fontsize=11, color="red")    
    plt.legend(loc = 'best')
    #plt.legend(loc='lower left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0) #legend outside
    plt.grid(True)
    plt.show()
    plt.close()


# In[385]:


def two_signals_off():
    plt.close()
    signal1, voltage_level1 = input("Enter first signal name and voltage level: ").split()
    file_name1 = signal1 + '.csv'
    while file_name1 not in dirs:
        print("Typo! Please try again.")
        signal1, voltage_level1 = input("Re-enter first signal name and voltage level: ").split()
        file_name1 = signal1 + '.csv'
    signal2, voltage_level2 = input("Enter second signal name and voltage level: ").split()
    file_name2 = signal2 + '.csv'
    while file_name2 not in dirs:
        print("Typo! Please try again.")
        signal2, voltage_level2 = input("Re-enter second signal name and voltage level: ").split()
        file_name2 = signal2 + '.csv'
    test_data_to_load = os.path.join(path,file_name1)
    test2_data_to_load = os.path.join(path,file_name2)
    voltage_level1 = float(voltage_level1)
    voltage_level2 = float(voltage_level2)
    plt.figure(figsize=(9,4)) #all in one figure(1)
    test_data_df = pd.read_csv(test_data_to_load,skiprows=4)
    test2_data_df = pd.read_csv(test2_data_to_load,skiprows=4)
    file_name1, file_extension = os.path.splitext(file_name1)
    file_name2, file_extension = os.path.splitext(file_name2)
    test_data_df["Time"] = test_data_df["Time"] * 1000
    test2_data_df["Time"] = test2_data_df["Time"] * 1000
    min = test_data_df["Ampl"].min()
    mean = test_data_df["Ampl"].mean()
    max = test_data_df["Ampl"].max()
    min2 = test2_data_df["Ampl"].min()
    mean2 = test2_data_df["Ampl"].mean()
    max2 = test2_data_df["Ampl"].max()
    a = max - min
    b = max2 - min2
    if voltage_level1 > 0 and voltage_level2 > 0: #High To High
        if  max > 1 and a > (max + 3):
            trigger_signal = test_data_df[test_data_df["Ampl"] > voltage_level1] #rising
            trigger_signal = trigger_signal.iloc[[-1]] #-1 is last number
        else:
            trigger_signal = test_data_df[test_data_df["Ampl"] < voltage_level1] #rising
            trigger_signal = trigger_signal.iloc[[1]] #-1 is last number
        trigger_x = float(trigger_signal.Time)
        if max2 > 1 and b > max2:
            trigger2_signal = test2_data_df[test2_data_df["Ampl"] > voltage_level2] #rising
            trigger2_signal = trigger2_signal.iloc[[-1]] #
        else:
            trigger2_signal = test2_data_df[test2_data_df["Ampl"] < voltage_level2] #rising
            trigger2_signal = trigger2_signal.iloc[[1]] #-1 is last number
        trigger2_x = float(trigger2_signal.Time)
        delay = abs(trigger2_x - trigger_x)
        print(delay)
    elif voltage_level1 > 0 and voltage_level2 < 0:
        if  max > 1 and a > (max + 3):
            trigger_signal = test_data_df[test_data_df["Ampl"] > voltage_level1] #rising
            trigger_signal = trigger_signal.iloc[[-1]] #-1 is last number
        else:
            trigger_signal = test_data_df[test_data_df["Ampl"] < voltage_level1] #rising
            trigger_signal = trigger_signal.iloc[[1]] #-1 is last number
        trigger_x = float(trigger_signal.Time)
        trigger2_signal = test2_data_df[test2_data_df["Ampl"] < voltage_level2] #rising
        trigger2_signal = trigger2_signal.iloc[[-1]] #-1 is last number
        trigger2_x = float(trigger2_signal.Time)
        delay = abs(trigger2_x - trigger_x)
        print(delay)
    elif voltage_level1 < 0 and voltage_level2 < 0:
        trigger_signal = test_data_df[test_data_df["Ampl"] < voltage_level1] #rising
        trigger_signal = trigger_signal.iloc[[-1]] #-1 is last number
        trigger_x = float(trigger_signal.Time)
        trigger2_signal = test2_data_df[test2_data_df["Ampl"] < voltage_level2] #rising
        trigger2_signal = trigger2_signal.iloc[[-1]] #-1 is last number
        trigger2_x = float(trigger2_signal.Time)
        delay = abs(trigger2_x - trigger_x)
        print(delay)
    elif voltage_level1 < 0 and voltage_level2 > 0:
        trigger_signal = test_data_df[test_data_df["Ampl"] < voltage_level1] #rising
        trigger_signal = trigger_signal.iloc[[-1]] #-1 is last number
        trigger_x = float(trigger_signal.Time)
        if max2 > 1 and b > max2:
            trigger2_signal = test2_data_df[test2_data_df["Ampl"] > voltage_level2] #rising
            trigger2_signal = trigger2_signal.iloc[[-1]] #
        else:
            trigger2_signal = test2_data_df[test2_data_df["Ampl"] < voltage_level2] #rising
            trigger2_signal = trigger2_signal.iloc[[1]] #-1 is last number
        trigger2_x = float(trigger2_signal.Time)
        delay = abs(trigger2_x - trigger_x)
        print(delay)
    plt.plot(test_data_df["Time"], test_data_df["Ampl"], label=f"{file_name1}") 
    plt.plot(test2_data_df["Time"], test2_data_df["Ampl"], label=f"{file_name2}")
    #plt.plot(test_data_df["Time"],test_data_df["Ampl"],label=f"{file_name} => DelayTime={time:.0f}mS)") 
    plt.xlabel("Time(ms)")
    plt.ylabel("Voltage(V)")
    plt.title(r'DelayTime betwwen 2 signals $is$: $\bf{{{a}}}(ms)$'.format(a=int(delay)))
    plt.annotate("", xy=(trigger_x, voltage_level1), xytext=(trigger2_x, voltage_level2), arrowprops=dict(arrowstyle="<->"))    
    plt.legend(loc = 'best')
    #plt.legend(loc='lower left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0) #legend outside
    plt.grid(True)
    plt.show()
    plt.close()


# In[386]:


def chargesharing():
    plt.close()
    #plt.figure(figsize=(12,6)) #all in one figure(1)
    offset = -9.9
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6)) #subplots(1,2) means fig left and right
    clk1_data_to_load = os.path.join(path,"clk1.csv")
    clk1_data_df = pd.read_csv(clk1_data_to_load,skiprows=4)
    clk1_data_df = clk1_data_df[100000:-890000] #1M sample rate
    #clk1_data_df = clk1_data_df[9930:-89300] #1K sample rate
    clk1_data_df["Time"] = clk1_data_df["Time"] * 1000000
    max = clk1_data_df["Ampl"].max()
    clk1_signal = clk1_data_df[clk1_data_df["Ampl"] > max*50/100]
    clk1_signal = clk1_signal.iloc[[1]] 
    a = float(clk1_signal.Time)
    #print(a)
    #display_panel = input("What is panel vendor?(SHP/LGD/BOE): ")
    #display_panel = display_panel.upper()
    for csv_file in sorted(dirs):
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        #plt.figure(figsize=(11,6)) #multiple figure(1)
        file_name, file_extension = os.path.splitext(csv_file)
        if file_name in ("clk1","clk2", "clk3", "clk4", "clk5", "clk6", "clk7", "clk8"):
            test_data_df = pd.read_csv(test_data_to_load,skiprows=4)
            test_data_df["Time"] = test_data_df["Time"] * 1000000
            offset = offset + 9.9
            test_data_df["Time"] = test_data_df["Time"] - offset
            #ax1.set_title(f"{name} - Rising Edge")
            ax1.plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name)
            #ax2.set_title(f"{name} - Falling Edge")
            ax2.plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name) 
            if a > 8 and a < 12:
                #x1 = 89
                #x2 = 128.5
                x1 = a + 79
                x2 = a + 118.5 #SHP is 39.5
                name = "SHP"
            elif a > 18 and a < 23:
                #x1 = 20.25
                #x2 = 57
                x1 = a - 0.5
                x2 = a + 36 #BOE is 35.5
                name = "BOE"
            elif a > 25 and a < 31:
                #x1 = 28.5
                #x2 = 63.5
                x1 = a - 0.5
                x2 = a + 34 # LGD is 33.5
                name = "LGD"
            ax1.set_xlim(x1 - 1.5, x1 + 1.5)
            ax2.set_xlim(x2 - 1.5, x2 + 1.5)
            ax1.grid(True)
            ax1.set_title(f"{name} - Rising Edge")
            ax2.set_title(f"{name} - Falling Edge")
            ax2.grid(True)
            ax1.set_xlabel("Time(us)")
            ax1.set_ylabel("Voltage(V)")
            ax2.set_xlabel("Time(us)")
            #plt.legend(loc = 'best')
            plt.legend(loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0) #legend outside
    print(name)
    plt.show()
    plt.close()


# In[387]:


def signal_rising():
    plt.close()
    i = 0
    signal1 = input("Please input the signal: ")
    file_name = signal1 + '.csv'
    while file_name not in dirs:
        print("Typo! Please try again.")
        signal1 = input("Please re-enter the signal: ")
        file_name = signal1 + '.csv'
    test_data_to_load = os.path.join(path,file_name)
    percent1, percent2 = input("Please input high and low Percentage(%): ").split()
    while True and i < 2:
        percent1 = int(percent1)
        percent2 = int(percent2)
        plt.figure(figsize=(9,4)) #all in one figure(1)
        test_data_df = pd.read_csv(test_data_to_load,skiprows=4)
        file_name, file_extension = os.path.splitext(file_name)
        #test_data_df = test_data_df[300000:-400000] #GateDriver
        test_data_df["Time"] = test_data_df["Time"] * 1000
        max = test_data_df["Ampl"].max()  
        mean = test_data_df["Ampl"].mean()
        min = test_data_df["Ampl"].min()
        p1 = (max * percent1)/100
        p2 = (max * percent2)/100
        p3 = (min * percent1)/100
        p4 = (min * percent2)/100
        if mean > 0:
            a_df = test_data_df[test_data_df["Ampl"] < p1]  
            a_df = a_df.iloc[[-1]]
            b_df = test_data_df[test_data_df["Ampl"] > p2]
            b_df = b_df.iloc[[1]]
            signal_x1 = float(a_df.Time)
            signal_x2 = float(b_df.Time)
            time = abs(signal_x1 - signal_x2)
            time = f"{time:.2f}"
            plt.annotate("<==", xy=(signal_x1, p1))
            plt.annotate("<==", xy=(signal_x2, p2))
        elif mean < 0:
            a_df = test_data_df[test_data_df["Ampl"] > p3]  
            a_df = a_df.iloc[[-1]]
            b_df = test_data_df[test_data_df["Ampl"] < p4]
            b_df = b_df.iloc[[1]]
            signal_x1 = float(a_df.Time)
            signal_x2 = float(b_df.Time)
            time = abs(signal_x1 - signal_x2)
            time = f"{time:.2f}"
            plt.annotate("<==", xy=(signal_x1, p3))
            plt.annotate("<==", xy=(signal_x2, p4))
        plt.plot(test_data_df["Time"], test_data_df["Ampl"], label=f"{file_name}")
        if mean > 0:  
            plt.title(r'Rising time $is$: $\bf{{{a}}}(ms)$'.format(a=float(time)))
        else:
            plt.title(r'Falling time $is$: $\bf{{{a}}}(ms)$'.format(a=float(time)))
        plt.xlabel("Time(ms)")
        plt.ylabel("Voltage(V)")
        plt.legend(loc = 'best')
        #plt.legend(loc='lower left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0) #legend outside
        plt.grid(True)
        plt.show()
        plt.close()
        i += 1
        if i < 2:
            try:
                percent1, percent2 = input("Enter new high and low Percentage(%) or enter to quit: ").split()
                
            except ValueError:
                break
        else: 
            continue


# In[388]:


def signal_falling():
    plt.close()
    i = 0
    signal1 = input("Please input the signal: ")
    file_name = signal1 + '.csv'
    while file_name not in dirs:
        print("Typo! Please try again.")
        signal1 = input("Please re-enter the signal: ")
        file_name = signal1 + '.csv'
    test_data_to_load = os.path.join(path,file_name)
    percent1, percent2 = input("Please input high and low Percentage(%): ").split()
    while True and i < 2:
        percent1 = int(percent1)
        percent2 = int(percent2)
        plt.figure(figsize=(9,4)) #all in one figure(1)
        test_data_df = pd.read_csv(test_data_to_load,skiprows=4)
        file_name, file_extension = os.path.splitext(file_name)
        #test_data_df = test_data_df[300000:-400000] #GateDriver
        test_data_df["Time"] = test_data_df["Time"] * 1000
        max = test_data_df["Ampl"].max()  
        mean = test_data_df["Ampl"].mean()
        min = test_data_df["Ampl"].min()
        p1 = (max * percent1)/100
        p2 = (max * percent2)/100
        p3 = (min * percent1)/100
        p4 = (min * percent2)/100
        if mean > 0:
            a_df = test_data_df[test_data_df["Ampl"] > p1]  
            a_df = a_df.iloc[[-1]]
            b_df = test_data_df[test_data_df["Ampl"] > p2]
            b_df = b_df.iloc[[-1]]
            signal_x1 = float(a_df.Time)
            signal_x2 = float(b_df.Time)
            time = abs(signal_x1 - signal_x2)
            time = f"{time:.2f}"
            plt.annotate("<==", xy=(signal_x1, p1))
            plt.annotate("<==", xy=(signal_x2, p2))
        elif mean < 0:
            a_df = test_data_df[test_data_df["Ampl"] < p3]  
            a_df = a_df.iloc[[-1]]
            b_df = test_data_df[test_data_df["Ampl"] < p4]
            b_df = b_df.iloc[[-1]]
            signal_x1 = float(a_df.Time)
            signal_x2 = float(b_df.Time)
            time = abs(signal_x1 - signal_x2)
            time = f"{time:.2f}"
            plt.annotate("<==", xy=(signal_x1, p3))
            plt.annotate("<==", xy=(signal_x2, p4))
        plt.plot(test_data_df["Time"], test_data_df["Ampl"], label=f"{file_name}") 
        if mean > 0:
            plt.title(r'Falling time $is$: $\bf{{{a}}}(ms)$'.format(a=float(time)))
        else:
            plt.title(r'Rising time $is$: $\bf{{{a}}}(ms)$'.format(a=float(time)))
        plt.xlabel("Time(ms)")
        plt.ylabel("Voltage(V)")
        plt.legend(loc = 'best')
        #plt.legend(loc='lower left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0) #legend outside
        plt.grid(True)
        plt.show()
        plt.close()
        i += 1
        if i < 2:
            try:
                percent1, percent2 = input("Enter new high and low Percentage(%) or enter to quit: ").split()
            except ValueError:
                break
        else: 
            continue


# In[389]:


def gate_driver_timing():
    plt.close()
    fig, axs = plt.subplots(10, sharex=True, sharey=True, figsize=(9,6))
    i = -1
    cycol = cycle('bgrcmk')
    for csv_file in sorted(dirs):
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        file_name, file_extension = os.path.splitext(csv_file)
        if folder_name.find('LGD') != -1:
            list1 = ("C1_vst1", "C2_vst2", "clk1", "clk2", "clk3", "clk4", "clk5","clk6", "clk7", "clk8")
            list2 = ("C1_vst1", "C2_vst2", "clk1", "clk2", "clk3", "clk4")   
            test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
            test_data_df = test_data_df[9930:-89745]         
        elif folder_name.find('SHP') != -1:
            list1 = ("C1_vst1", "clk1", "clk2", "clk3", "clk4", "clk5","clk6", "clk7", "clk8")
            list2 =("C1_vst1", "clk6", "clk7", "clk8")
            test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
            test_data_df = test_data_df[9930:-89750]
        else:
            list1 = ("C1_vst1", "C2_vst2", "clk1", "clk2", "clk3", "clk4", "clk5","clk6", "clk7", "clk8")
            list2 = ("C1_vst1", "C2_vst2", "clk6", "clk7", "clk8")   
            test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
            test_data_df = test_data_df[9930:-89300]        
        if file_name in list1: 
            test_data_df["Time"] = test_data_df["Time"] * 1000000
            min = test_data_df["Ampl"].min()
            mean = test_data_df["Ampl"].mean()
            max = test_data_df["Ampl"].max()  
            voltage_level = max*50/100
            trigger_signal = test_data_df[test_data_df["Ampl"] > voltage_level]
            trigger_signal = trigger_signal.iloc[[1]] 
            x1 = float(trigger_signal.Time)
            trigger2_signal = test_data_df[test_data_df["Ampl"] > voltage_level]
            if file_name not in list2:
                #trigger2_signal = trigger2_signal.iloc[[170]] #BOE
                if folder_name.find('J293') != -1:
                    trigger2_signal = trigger2_signal.iloc[[155]] #LGD (total distance from left to right axis)
                    x2 = float(trigger2_signal.Time) 
                if folder_name.find('SHP') != -1:
                    trigger2_signal = trigger2_signal.iloc[[65]] #LGD (total distance from left to right axis)
                    x2 = float(trigger2_signal.Time) 
                if folder_name.find('LGD') != -1:
                    trigger2_signal = trigger2_signal.iloc[[55]] #LGD (total distance from left to right axis)
                    x2 = float(trigger2_signal.Time) 
            else:
                trigger2_signal = trigger2_signal.iloc[[-1]] 
                x2 = float(trigger2_signal.Time) 
            #print(x2)
            i += 1
            if file_name.startswith('C'):
                file_name = file_name[3:]
            axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol)) 
            fig.suptitle('Gate Driver Timing', fontweight="bold", fontsize=20, color="red")
            plt.xlabel("Time(us)")
            plt.ylabel("Voltage(V)")
            axs[i].annotate(s='', xy=(0,voltage_level), xytext=(x1,voltage_level), arrowprops=dict(arrowstyle='<->'))
            axs[i].vlines(x=x1, ymin=min, ymax=max, colors='green', ls=':', lw=2)
            axs[i].annotate(s='', xy=(x1,voltage_level - 2 ), xytext=(x2,voltage_level - 2),arrowprops=dict(arrowstyle='<->'))
            axs[i].vlines(x=x2, ymin=min, ymax=max, colors='green', ls=':', lw=2)
            if file_name != "C1_vst1":
                axs[i].annotate(f"{(x1):.0f}", (0,voltage_level-13), fontsize=11, color="red")
            axs[i].annotate(f"{(x2-x1):.0f}", (x2-5,voltage_level-13), fontsize=11, color="red")
            plt.xticks(fontsize=10)
            axs[i].legend() 
            #axs[i].legend(loc = 'best')
            plt.subplots_adjust(right=.85) 
            plt.rc('legend',**{'fontsize':8})
            axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
            axs[i].grid(True)


# In[390]:


def scopeview():
    plt.close()
    list = os.listdir(path) 
    file_count = len(list)
    if folder_name.find('System') != -1:
        fig, axs = plt.subplots(file_count - 2, sharex=True, sharey=False, figsize=(8,7))
    elif file_count > 2:
        fig, axs = plt.subplots(file_count, sharex=True, sharey=False, figsize=(8,7))
    else:
        pass
    i = -1
    cycol = cycle('bgrcmk')
    for csv_file in sorted(dirs):
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        file_name, file_extension = os.path.splitext(csv_file)
        if file_name not in ("edp_2", "edp_3", "ls_int", "ls_vbe"):
            try:
                test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
                test_data_df["Time"] = test_data_df["Time"] * 1000
                i += 1  
                if file_name.startswith('C'):
                    file_name = file_name[3:]
                axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol)) 
            except:
                test_data_df = pd.read_csv(test_data_to_load, skiprows=11) 
                test_data_df.dropna(how='all', axis=1, inplace=True) #drop empty column
                test_data_df["TIME"] = test_data_df["TIME"] * 1000 
                #test_data_df = test_data_df[test_data_df.columns - test_data_df["TIME"]]
                #test_data_df.sub(test_data_df['TIME'], axis=0)
                fig, axs = plt.subplots(len(test_data_df.columns), sharex=True, sharey=False, figsize=(8,7))
                for c in test_data_df.columns:
                    i += 1  
                    if i != 0:
                        axs[i].plot(test_data_df["TIME"],test_data_df[c], label=c)
                        fig.suptitle(f"{path}", fontweight="bold", fontsize=20, color="red")
                        plt.rc('legend',**{'fontsize':8})
                        axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
                        axs[i].grid(True)
            fig.suptitle(f"Scope View_{folder_name}", fontweight="bold", fontsize=20, color="red")
            plt.xlabel("Time(ms)")
            plt.ylabel("Voltage(V)")
            plt.xticks(fontsize=10)
            axs[i].legend() 
            #axs[i].legend(loc = 'best')
            plt.subplots_adjust(right=.85) 
            plt.rc('legend',**{'fontsize':8})
            axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
            axs[i].grid(True)


# In[391]:


def plotsignals():
    plt.close()
    new_tuple = tuple(str(x) for x in input("Enter a list of signals separated by space: ").split())
    if len(new_tuple) == 1:
        new_tuple = new_tuple[0]
        fig, axs = plt.subplots(2, sharex=True, sharey=False, figsize=(8,7))
    else:
        #print("Typo! Please try again.")
        fig, axs = plt.subplots(len(new_tuple), sharex=True, sharey=False, figsize=(8,7))
    print(f"List of signals: {new_tuple}")
    i = -1
    cycol = cycle('bgrcmk')
    for csv_file in sorted(dirs):
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        file_name, file_extension = os.path.splitext(csv_file)
        if file_name in (new_tuple):
            test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
            test_data_df["Time"] = test_data_df["Time"] * 1000
            i += 1     
            axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol))
        elif file_name.find('ALL') != -1:
            test_data_df = pd.read_csv(test_data_to_load, skiprows=11) 
            test_data_df.dropna(how='all', axis=1, inplace=True) #drop empty column
            test_data_df["TIME"] = test_data_df["TIME"] * 1000 
            #fig, axs = plt.subplots(len(new_tuple), sharex=True, sharey=False, figsize=(8,7))
            for c in test_data_df.columns:
                if c in (new_tuple):
                    i += 1  
                    axs[i].plot(test_data_df["TIME"],test_data_df[c], label=c, c=next(cycol))
                    fig.suptitle(f"{path}", fontweight="bold", fontsize=20, color="red")
                    plt.rc('legend',**{'fontsize':8})
                    axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
                    axs[i].grid(True)
        fig.suptitle(f"Scope View_{folder_name}", fontweight="bold", fontsize=20, color="red")
        plt.xlabel("Time(ms)")
        plt.ylabel("Voltage(V)")
        plt.xticks(fontsize=10)
        axs[i].legend() 
        #axs[i].legend(loc = 'best')
        plt.subplots_adjust(right=.85) 
        plt.rc('legend',**{'fontsize':8})
        axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
        axs[i].grid(True)
        


# In[392]:


def meas_power_on_tek():
    plt.close()
    #fig, axs = plt.subplots(len(test_data_df.columns), sharex=True, sharey=False, figsize=(8,7))
    i = -1
    cycol = cycle('bgrcmk')
    test_data_df = pd.read_csv(test_data_to_load, skiprows=11) 
    test_data_df.dropna(how='all', axis=1, inplace=True) #drop empty column
    test_data_df["TIME"] = test_data_df["TIME"] * 1000 
    fig, axs = plt.subplots(len(test_data_df.columns), sharex=True, sharey=False, figsize=(8,7))
    #fig, axs = plt.subplots(len(new_tuple), sharex=True, sharey=False, figsize=(8,7))
    for c in test_data_df.columns:
        min = test_data_df[c].min()
        mean = test_data_df[c].mean()
        max = test_data_df[c].max()
        i += 1  
        if mean > 0:
            trigger_signal = test_data_df[test_data_df[c] > (max*85/100)]
            trigger_signal = trigger_signal.iloc[[1]]
        elif mean < 0:
            trigger_signal = test_data_df[test_data_df[c] > (min*85/100)]
            trigger_signal = trigger_signal.iloc[[-1]] 
        x1 = float(trigger_signal.TIME) 
        if i != 0:
            axs[i].plot(test_data_df["TIME"],test_data_df[c], label=c, c=next(cycol))
            fig.suptitle(f"{file_name}", fontweight="bold", fontsize=20, color="red")
            plt.rc('legend',**{'fontsize':8})
            axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
            axs[i].grid(True)
            plt.xlabel("Time(ms)")
            plt.ylabel("Voltage(V)")
        if c != "TIME":
            if mean > 0:
                axs[i].annotate(s='', xy=(0,(max*85/100)), xytext=(x1,(max*85/100)), arrowprops=dict(arrowstyle='<->'))
                axs[i].annotate(f"{(x1):.0f}", (0,max/2), fontsize=6, color="red")
            elif mean < 0:
                axs[i].annotate(s='', xy=(0,(min*85/100)), xytext=(x1,(min*85/100)), arrowprops=dict(arrowstyle='<->'))
                axs[i].annotate(f"{(x1):.0f}", (0,min/2), fontsize=6, color="red")
                
            axs[i].vlines(x=x1, ymin=min, ymax=max, colors='green', ls=':', lw=2)
            plt.xticks(fontsize=10)
            axs[i].legend() 
            t = np.isclose(x1, -0, atol=0.2).any()
            if (x1 > -100 or x1 != 0) and t == False:
                print(blue(f"Trigger To {c}: {x1:.3f}(ms)"))
                name = folder_name + ".txt"
                with open(results_dir + name, 'a') as output:
                    output.write(f"Trigger To {c}: {x1:.3f}(ms)"'\n')
                   
            #axs[i].legend(loc = 'best')
            plt.subplots_adjust(right=.85) 
            plt.rc('legend',**{'fontsize':8})
            axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
            axs[i].grid(True)
    with open(results_dir + name, 'a') as output:
        output.write(f"------------------------------------"'\n')


# In[393]:


def meas_power_on():
    plt.close()
    global folder_name
    list = os.listdir(path) 
    file_count = len(list)
    if folder_name.find('System') != -1 or folder_name.find('DriverTiming') != -1:
        fig, axs = plt.subplots(file_count - 2, sharex=True, sharey=False, figsize=(8,7))
    else:
        fig, axs = plt.subplots(file_count, sharex=True, sharey=False, figsize=(8,7))
    i = -1
    cycol = cycle('bgrcmk')
    for csv_file in sorted(dirs):
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        file_name, file_extension = os.path.splitext(csv_file)     
        if file_name not in ("C1_vst11", "C5_vst1", "C4_eof1","edp_2", "edp_3", "ls_int", "ls_vbe", "3v3_2"):
            test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
            test_data_df["Time"] = test_data_df["Time"] * 1000
            min = test_data_df["Ampl"].min()
            mean = test_data_df["Ampl"].mean()
            max = test_data_df["Ampl"].max()
            a = max - min         
            #if file_name in ("clk1", "eof1", "vst1", "tcon_sclk", "ls_vbe", "cd_vb", "hpd", "tcon_reset"):
            if (folder_name.find('Display') != -1 or folder_name.find('SleepWake') != -1) and file_name in ("smiso", "smosi", "ssclk", "scsl"):
                x1 = 0
            elif folder_name.find('SystemMatch') != -1 and file_name != "3v3":
                if file_name == "cscl":
                    trigger_signal = test_data_df[test_data_df["Ampl"] < (max*50/100)] 
                    trigger_signal = trigger_signal.iloc[[1]]
                    x1 = float(trigger_signal.Time)
                elif file_name == "scsl":
                    trigger_signal = test_data_df[test_data_df["Ampl"] > (max*10/100)] 
                    trigger_signal = trigger_signal.iloc[[-1]]
                    x1 = float(trigger_signal.Time)
                elif file_name in ("smiso", "smosi", "ssclk"):
                    trigger_signal = test_data_df[test_data_df["Ampl"] > (max*10/100)] 
                    trigger_signal = trigger_signal.iloc[[1]]
                    x1 = float(trigger_signal.Time)
                else:
                    trigger_signal = test_data_df[test_data_df["Ampl"] > (max*50/100)] 
                    trigger_signal = trigger_signal.iloc[[1]]
                    x1 = float(trigger_signal.Time)
            elif folder_name == "PowerOnGateDriver" and file_name != "clk1":
                if mean > 0:
                    trigger_signal = test_data_df[test_data_df["Ampl"] > (max*50/100)] 
                    trigger_signal = trigger_signal.iloc[[1]]
                    x1 = float(trigger_signal.Time)
                elif mean < 0:
                    trigger_signal = test_data_df[test_data_df["Ampl"] > (min*50/100)] 
                    trigger_signal = trigger_signal.iloc[[-1]]
                x1 = float(trigger_signal.Time)
            elif mean > 0:
                    trigger_signal = test_data_df[test_data_df["Ampl"] > (max*85/100)]
                    trigger_signal = trigger_signal.iloc[[1]]
                    x1 = float(trigger_signal.Time)
            elif mean < 0:
                if file_name in ("clk1", "eof1", "vst1", "tcon_sclk"):
                    trigger_signal = test_data_df[test_data_df["Ampl"] > (max*50/100)] 
                    trigger_signal = trigger_signal.iloc[[1]]
                    x1 = float(trigger_signal.Time)
                else:
                    trigger_signal = test_data_df[test_data_df["Ampl"] > (min*85/100)]
                    trigger_signal = trigger_signal.iloc[[-1]] 
                    x1 = float(trigger_signal.Time)
            i += 1
            axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol)) 
            fig.suptitle(f"{folder_name}", fontweight="bold", fontsize=20, color="red")
            plt.xlabel("Time(ms)")
            plt.ylabel("Voltage(V)")
            if folder_name.find('SystemMatch') != -1 and x1 < -50 or x1 > 340:
                axs[i].annotate("", (0,max/2), fontsize=10, color="red")
            elif mean > 0 and x1 != 0:
                axs[i].annotate(s='', xy=(0,(max*85/100)), xytext=(x1,(max*85/100)), arrowprops=dict(arrowstyle='<->'))
                axs[i].annotate(f"{(x1):.0f}", (0,max/2), fontsize=6, color="red")
            elif mean < 0 and x1 != 0:
                axs[i].annotate(s='', xy=(0,(min*85/100)), xytext=(x1,(min*85/100)), arrowprops=dict(arrowstyle='<->'))
                axs[i].annotate(f"{(x1):.0f}", (0,min/2), fontsize=6, color="red")
            axs[i].vlines(x=x1, ymin=min, ymax=max, colors='green', ls=':', lw=2)
            plt.xticks(fontsize=10)
            axs[i].legend() 
            t = np.isclose(x1, -0, atol=0.2).any()
            if (x1 > -100 or x1 != 0) and t == False:
                print(blue(f"Trigger To {file_name}: {x1:.3f}(ms)"))
                name = folder_name + ".txt"
                with open(results_dir + name, 'a') as output:
                    output.write(f"Trigger To {file_name}: {x1:.3f}(ms)"'\n')
                   
            #axs[i].legend(loc = 'best')
            plt.subplots_adjust(right=.85) 
            plt.rc('legend',**{'fontsize':8})
            axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
            axs[i].grid(True)
    with open(results_dir + name, 'a') as output:
        output.write(f"------------------------------------"'\n')
        


# In[394]:


def meas_power_off_tek():
    plt.close()
    #fig, axs = plt.subplots(len(test_data_df.columns), sharex=True, sharey=False, figsize=(8,7))
    i = -1
    cycol = cycle('bgrcmk')
    test_data_df = pd.read_csv(test_data_to_load, skiprows=11) 
    test_data_df.dropna(how='all', axis=1, inplace=True) #drop empty column
    test_data_df["TIME"] = test_data_df["TIME"] * 1000 
    fig, axs = plt.subplots(len(test_data_df.columns), sharex=True, sharey=False, figsize=(8,7))
    #fig, axs = plt.subplots(len(new_tuple), sharex=True, sharey=False, figsize=(8,7))
    for c in test_data_df.columns:
        min = test_data_df[c].min()
        mean = test_data_df[c].mean()
        max = test_data_df[c].max()
        i += 1  
        if c == "vghboost":
            x1 = 0
        elif mean > 0:
            trigger_signal = test_data_df[test_data_df[c] < (max*15/100)]
            trigger_signal = trigger_signal.iloc[[1]] 
        elif mean < 0:
            trigger_signal = test_data_df[test_data_df[c] < (min*15/100)]
            trigger_signal = trigger_signal.iloc[[-1]] 
        x1 = float(trigger_signal.TIME) 
        if i != 0:
            axs[i].plot(test_data_df["TIME"],test_data_df[c], label=c, c=next(cycol))
            fig.suptitle(f"{file_name}", fontweight="bold", fontsize=20, color="red")
            plt.rc('legend',**{'fontsize':8})
            axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
            axs[i].grid(True)
            plt.xlabel("Time(ms)")
            plt.ylabel("Voltage(V)")
        if c != "TIME":
            if mean > 0:
                axs[i].annotate(s='', xy=(0,(max*15/100)), xytext=(x1,(max*15/100)), arrowprops=dict(arrowstyle='<->'))
                axs[i].annotate(f"{(x1):.0f}", (0,max/2), fontsize=6, color="red")
            elif mean < 0:
                axs[i].annotate(s='', xy=(0,(min*15/100)), xytext=(x1,(min*15/100)), arrowprops=dict(arrowstyle='<->'))
                axs[i].annotate(f"{(x1):.0f}", (0,min/2), fontsize=6, color="red")
                
            axs[i].vlines(x=x1, ymin=min, ymax=max, colors='green', ls=':', lw=2)
            plt.xticks(fontsize=10)
            axs[i].legend() 
            t = np.isclose(x1, -0, atol=0.2).any()
            if (x1 > -100 or x1 != 0) and t == False:
                print(blue(f"Trigger To {c}: {x1:.3f}(ms)"))
                name = folder_name + ".txt"
                with open(results_dir + name, 'a') as output:
                    output.write(f"Trigger To {c}: {x1:.3f}(ms)"'\n')
                   
            #axs[i].legend(loc = 'best')
            plt.subplots_adjust(right=.85) 
            plt.rc('legend',**{'fontsize':8})
            axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
            axs[i].grid(True)
    with open(results_dir + name, 'a') as output:
        output.write(f"------------------------------------"'\n')


# In[395]:


def meas_power_off():
    plt.close()
    list = os.listdir(path) 
    file_count = len(list)
    if folder_name.find('System') != -1 or folder_name.find('DriverTiming') != -1:
        fig, axs = plt.subplots(file_count - 2, sharex=True, sharey=False, figsize=(8,7))
    else:
        fig, axs = plt.subplots(file_count, sharex=True, sharey=False, figsize=(8,7))
    i = -1
    cycol = cycle('bgrcmk')
    for csv_file in sorted(dirs):
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        file_name, file_extension = os.path.splitext(csv_file)
        if file_name not in ("C1_vst11", "C5_vst1", "C4_eof1","edp_2", "edp_3", "ls_int", "ls_vbe", "3v3_2"):
            test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
            test_data_df["Time"] = test_data_df["Time"] * 1000
            min = test_data_df["Ampl"].min()
            mean = test_data_df["Ampl"].mean()
            max = test_data_df["Ampl"].max()
            a = max - min
            if file_name in ("vghboost", "DCHG_VGL1", "tcon_sclk", "pp3v3", "smiso", "smosi", "ssclk", "scsl", "aux_p"):
                x1 = 0
            elif folder_name.find('Discharge') != -1 and file_name == "clk1":
                trigger_signal = test_data_df[test_data_df["Ampl"] < (min*15/100)] 
                trigger_signal = trigger_signal.iloc[[-1]]
                x1 = float(trigger_signal.Time)
            elif file_name in ("mbc", "vcom", "cd_vb", "ls_clk", "ls_int", "ls_vbe", "CDIC_data"):
                trigger_signal = test_data_df[test_data_df["Ampl"] > (max*50/100)] 
                trigger_signal = trigger_signal.iloc[[-1]]
                x1 = float(trigger_signal.Time)
            elif mean > 0:
                try:
                    trigger_signal = test_data_df[test_data_df["Ampl"] < (max*15/100)]
                    trigger_signal = trigger_signal.iloc[[1]] 
                    x1 = float(trigger_signal.Time)
                except:
                    pass
            elif mean < 0:
                trigger_signal = test_data_df[test_data_df["Ampl"] < (min*15/100)]
                trigger_signal = trigger_signal.iloc[[-1]] 
                x1 = float(trigger_signal.Time)
            i += 1
            axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol)) 
            fig.suptitle(f"{folder_name}", fontweight="bold", fontsize=20, color="red")
            plt.xlabel("Time(ms)")
            plt.ylabel("Voltage(V)")
            if x1 == 0:
                axs[i].annotate("", (0,max/2), fontsize=10, color="red")
            elif mean > 0:
                axs[i].annotate(s='', xy=(0,(max*15/100)), xytext=(x1,(max*15/100)), arrowprops=dict(arrowstyle='<->'))
                axs[i].annotate(f"{(x1):.0f}", (0,max/2), fontsize=6, color="red")
            elif mean < 0:
                axs[i].annotate(s='', xy=(0,(min*15/100)), xytext=(x1,(min*15/100)), arrowprops=dict(arrowstyle='<->'))
                axs[i].annotate(f"{(x1):.0f}", (0,min/2), fontsize=6, color="red")
            axs[i].vlines(x=x1, ymin=min, ymax=max, colors='green', ls=':', lw=2)
            plt.xticks(fontsize=10)
            axs[i].legend() 
            t = np.isclose(x1, -0, atol=0.4).any()
            if x1 != 0 and t == False:
                print(blue(f"Trigger To {file_name}: {x1:.3f}(ms)"))
                name = folder_name + ".txt"
                with open(results_dir + name, 'a') as output:
                    output.write(f"Trigger To {file_name}: {x1:.3f}(ms)"'\n')
            #axs[i].legend(loc = 'best')
            plt.subplots_adjust(right=.85) 
            plt.rc('legend',**{'fontsize':8})
            axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
            axs[i].grid(True)
    with open(results_dir + name, 'a') as output:
        output.write(f"------------------------------------"'\n')


# In[396]:


def vcom_char():
    plt.close()
    list = os.listdir(path) 
    file_count = len(list)
    fig, axs = plt.subplots(file_count, sharex=True, sharey=False, figsize=(8,7))
    i = -1
    cycol = cycle('bgrcmk')
    for csv_file in sorted(dirs):
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        file_name, file_extension = os.path.splitext(csv_file)      
        test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
        test_data_df1 = test_data_df[110000:-550000]
        test_data_df["Time"] = test_data_df["Time"] * 1000
        min = test_data_df1["Ampl"].min()
        mean = test_data_df1["Ampl"].mean()
        max = test_data_df1["Ampl"].max()
        pkpk = max - min    
        min2 = test_data_df["Ampl"].min()
        max2 = test_data_df["Ampl"].max()
        i += 1
        axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol)) 
        fig.suptitle(f"{folder_name}", fontweight="bold", fontsize=20, color="red")
        plt.xlabel("Time(ms)")
        plt.ylabel("Voltage(V)")
        plt.xticks(fontsize=10)
        axs[i].legend()   
        if file_name in ("vcom1", "vcom1FB", "vcom2", "vcom2FB"):
            print(blue(f"{file_name}_Ripple(pkpk): {pkpk:.2f}(V)"))
            print(blue(f"{file_name}_Ripple(max): {max2:.2f}(V)"))
            print(blue(f"{file_name}_Ripple(min): {min2:.2f}(V)"))
            
        name = folder_name + ".txt"
        with open(results_dir + name, 'a') as output:
            if file_name in ("vcom1", "vcom1FB", "vcom2", "vcom2FB"):
                output.write(f"{file_name}_Ripple(pkpk): {pkpk:.2f}(V)"'\n')
                output.write(f"{file_name}_Ripple(max): {max2:.2f}(V)"'\n')
                output.write(f"{file_name}_Ripple(min): {min2:.2f}(V)"'\n')          
        #axs[i].legend(loc = 'best')
        plt.subplots_adjust(right=.85) 
        plt.rc('legend',**{'fontsize':8})
        axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
        axs[i].grid(True)
    with open(results_dir + name, 'a') as output:
        output.write(f"------------------------------------"'\n')


# In[397]:


def voltage_ripple():
    plt.close()
    list = os.listdir(path) 
    file_count = len(list)
    fig, axs = plt.subplots(file_count, sharex=True, sharey=False, figsize=(8,7))
    i = -1
    cycol = cycle('bgrcmk')
    for csv_file in sorted(dirs):
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        file_name, file_extension = os.path.splitext(csv_file)      
        test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
        if folder_name.find('24Hz') != -1:
            test_data_df1 = test_data_df[110000:-860000]
        else:
            test_data_df1 = test_data_df[110000:-550000]
        test_data_df["Time"] = test_data_df["Time"] * 1000
        min = test_data_df1["Ampl"].min()
        mean = test_data_df1["Ampl"].mean()
        max = test_data_df1["Ampl"].max()
        pkpk = (max - min) * 1000     
        min2 = test_data_df["Ampl"].min()
        max2 = test_data_df["Ampl"].max()
        i += 1
        axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol)) 
        fig.suptitle(f"{folder_name}", fontweight="bold", fontsize=20, color="red")
        plt.xlabel("Time(ms)")
        plt.ylabel("Voltage(V)")
        plt.xticks(fontsize=10)
        axs[i].legend()   
        if file_name not in ("ls_init", "cd_vb", 'mbc'):
            print(blue(f"{file_name}_Ripple(pkpk): {pkpk:.2f}(mV)"))
            print(blue(f"{file_name}_Ripple(max): {max2:.2f}(mV)"))
            print(blue(f"{file_name}_Ripple(min): {min2:.2f}(mV)"))
            
        name = folder_name + ".txt"
        with open(results_dir + name, 'a') as output:
            if file_name not in ("ls_init", "cd_vb", 'mbc'):
                output.write(f"{file_name}_Ripple(pkpk): {pkpk:.2f}(mV)"'\n')
                output.write(f"{file_name}_Ripple(max): {max2:.2f}(mV)"'\n')
                output.write(f"{file_name}_Ripple(min): {min2:.2f}(mV)"'\n')          
        #axs[i].legend(loc = 'best')
        plt.subplots_adjust(right=.85) 
        plt.rc('legend',**{'fontsize':8})
        axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
        axs[i].grid(True)
    with open(results_dir + name, 'a') as output:
        output.write(f"------------------------------------"'\n')


# In[398]:


def rename():
    #folder_name = os.path.split(os.path.abspath(path))[-1] # split folder name from the path
    #path = path + '/'
    dirs = os.listdir(path)
    #mypath = "~/Desktop/Analysis_Projects/RenameFiles/"
    mypath = "/Users/hiep_pham/Desktop/Analysis_Projects/RenameFiles/"
    for file in os.listdir(mypath):
        if fnmatch.fnmatch(file, folder_name + '.txt'): 
            new_dict = {}
            with open(mypath + (folder_name + '.txt')) as mapping_file:
                for line in mapping_file:
                    # Split the line along whitespace
                    # Note: this fails if your filenames have whitespace
                    old_name, new_name = line.split()
                    new_dict[old_name] = new_name
    # List the files in the current directory
    for filename in os.listdir(path):
        root, extension = os.path.splitext(filename)
        if not extension.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        rootInMap = root.split('-')[0]
        if rootInMap in new_dict: #only match the first two characters of the filename 
            oldfilename = filename
            newfilename = new_dict[rootInMap]+extension
            os.chdir(path)
            #print("Renaming " + oldfilename + ' -> ' + newfilename)
            os.rename(oldfilename,newfilename)


# In[399]:


def rename2(f_path, new_name):
    filelist = glob.glob(f_path + "C*01.csv")
    count = 8
    for file in sorted(filelist):
        filename = os.path.split(file)
        #print(filename)
        count = count + 1
        new_filename = f_path + new_name + str(count) + ".csv"
        os.rename(f_path+filename[1], new_filename)
        #print(new_filename)


# In[400]:


def rename3(f_path, new_name):
    filelist = glob.glob(f_path + "C*02.csv")
    count = 16
    for file in sorted(filelist):
        #print("File Count : ", count)
        filename = os.path.split(file)
        #print(filename)
        count = count + 1
        new_filename = f_path + new_name + str(count) + ".csv"
        os.rename(f_path+filename[1], new_filename)     
       #print(new_filename)


# In[401]:


def eof():
    plt.close()
    fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(9,6))
    i = -1
    cycol = cycle('bgrcmk')
    for csv_file in sorted(dirs):
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        #plt.figure(figsize=(11,6)) #multiple figure(1)
        file_name, file_extension = os.path.splitext(csv_file)
        if folder_name.find('LGD') != -1:
            list = ("C1_vst1", "C3_eof2")
        else:
            list = ("C1_vst1", "C3_eof1")  
        if file_name in list:
            test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
            test_data_df = test_data_df[5000:-88500]
            #plt.xlim([-0.02,0.1])
            test_data_df["Time"] = test_data_df["Time"] * 1000000
            #plt.plot(test_data_df["Time"], test_data_df["Ampl"]) 
            #plt.plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name) 
            min = test_data_df["Ampl"].min()
            mean = test_data_df["Ampl"].mean()
            max = test_data_df["Ampl"].max()  
            voltage_level = max*50/100
            if file_name == "C1_vst1":
                trigger_signal = test_data_df[test_data_df["Ampl"] > voltage_level]
                trigger_signal = trigger_signal.iloc[[1]] 
                x1 = float(trigger_signal.Time)
                trigger_signal = test_data_df[test_data_df["Ampl"] > voltage_level]
                trigger_signal = trigger_signal.iloc[[-1]] 
                x2 = float(trigger_signal.Time)   
            else:
                trigger_signal = test_data_df[test_data_df["Ampl"] > voltage_level]
                trigger_signal = trigger_signal.iloc[[-1]] 
                x1 = float(trigger_signal.Time)   
                trigger2_signal = test_data_df[test_data_df["Ampl"] > voltage_level]
                trigger2_signal = trigger2_signal.iloc[[1]] 
                x2 = float(trigger2_signal.Time)
            #print(x2)
            i += 1
            if file_name == "C1_vst1":
                axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label="vst1", c=next(cycol))
            elif file_name == "C3_eof2":
                axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label="eof2", c=next(cycol))
            else:
                axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label="eof1", c=next(cycol)) 
            fig.suptitle('Gate Driver Timing', fontweight="bold", fontsize=16, color="red")
            plt.xlabel("Time(us)")
            plt.ylabel("Voltage(V)")
           
            axs[i].annotate(s='', xy=(0,voltage_level), xytext=(x1,voltage_level), arrowprops=dict(arrowstyle='<->'))
            #axs[i].annotate(s='', xy=(0,voltage_level), xytext=(x2,voltage_level), arrowprops=dict(arrowstyle='<->'))
            axs[i].annotate(s='', xy=(x1,voltage_level - 2 ), xytext=(x2,voltage_level - 2), arrowprops=dict(arrowstyle='<->'))
            #axs[i].annotate(f"{(x1):.0f}", (0,voltage_level-11), fontsize=11, color="red")
            if file_name != "C1_vst1":
                axs[i].annotate(f"{(x1):.0f}", (x1/2,voltage_level + 1), fontsize=11, color="red")
            axs[i].annotate(f"{(x2 -x1):.0f}", (x1+2,voltage_level - 5), fontsize=11, color="red")
            plt.xticks(fontsize=10)
            axs[i].legend() 
            #axs[i].legend(loc = 'best')   
            plt.subplots_adjust(right=.85) 
            plt.rc('legend',**{'fontsize':8})
            axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
            axs[i].grid(True)


# In[402]:


def hvlfs():
    plt.close()
    list = os.listdir(path) 
    file_count = len(list)
    fig, axs = plt.subplots(file_count, sharex=True, sharey=False, figsize=(9,6))
    i = -1
    cycol = cycle('bgrcmk')
    for csv_file in sorted(dirs):
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        #plt.figure(figsize=(11,6)) #multiple figure(1)
        file_name, file_extension = os.path.splitext(csv_file)
        test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
        test_data_df = test_data_df[9900:-89900]
        test_data_df["Time"] = test_data_df["Time"] * 1000000        
        min = test_data_df["Ampl"].min()
        mean = test_data_df["Ampl"].mean()
        max = test_data_df["Ampl"].max()  
        voltage_level = max*50/100
        if file_name == "hvlfs2" and folder_name.find('LGD') == -1:
            trigger_signal = test_data_df[test_data_df["Ampl"] < voltage_level]
            trigger_signal = trigger_signal.iloc[[-1]] 
            x1 = float(trigger_signal.Time)
        elif file_name == "hvtls":
            trigger_signal = test_data_df[test_data_df["Ampl"] < mean]
            trigger_signal = trigger_signal.iloc[[-1]] 
            x1 = float(trigger_signal.Time)
        else: 
            trigger_signal = test_data_df[test_data_df["Ampl"] > voltage_level]
            trigger_signal = trigger_signal.iloc[[-1]] 
            x1 = float(trigger_signal.Time)   
        i += 1
        axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol)) 
        fig.suptitle('Gate Driver Timing', fontweight="bold", fontsize=20, color="red")
        plt.xlabel("Time(us)")
        plt.ylabel("Voltage(V)")
        if file_name.find('hvtls') != -1:
            axs[i].annotate(s='', xy=(0,-5), xytext=(x1,-5), arrowprops=dict(arrowstyle='<->'))
            axs[i].annotate(f"{(x1):.0f}", (x1-1,-3), fontsize=11, color="red")
        elif file_name.find('hvlfs1') == -1:
            axs[i].annotate(s='', xy=(0,voltage_level), xytext=(x1,voltage_level), arrowprops=dict(arrowstyle='<->'))
            axs[i].annotate(f"{(x1):.0f}", (x1+0.2,voltage_level - 6), fontsize=11, color="red")
        plt.xticks(fontsize=10)
        axs[i].legend() #label=file_name
        #axs[i].legend(loc = 'best')
        plt.subplots_adjust(right=.85) 
        plt.rc('legend',**{'fontsize':8})
        axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
        axs[i].grid(True)


# In[403]:


def cdic_timing():
    plt.close()
    fig, axs = plt.subplots(8, sharex=True, sharey=False, figsize=(9,6))
    i = -1
    cycol = cycle('bgrcmk')
    for csv_file in sorted(dirs):
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        #plt.figure(figsize=(11,6)) #multiple figure(1)
        file_name, file_extension = os.path.splitext(csv_file)
        test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
        if folder_name.find('LGD') != -1:
            test_data_df = test_data_df[44000:-44000]
        else:
            test_data_df = test_data_df[40000:-44000]
        test_data_df["Time"] = test_data_df["Time"] * 1000000
        min = test_data_df["Ampl"].min()
        mean = test_data_df["Ampl"].mean()
        max = test_data_df["Ampl"].max()  
        voltage_level = max*50/100
        if file_name == "eof1":
            trigger_signal = test_data_df[test_data_df["Ampl"] > voltage_level]
            trigger_signal = trigger_signal.iloc[[1]] 
            x1 = float(trigger_signal.Time)
            trigger2_signal = test_data_df[test_data_df["Ampl"] > voltage_level]
            trigger2_signal = trigger2_signal.iloc[[-1]] 
            x2 = float(trigger2_signal.Time) 
        elif file_name in ("clk1", "clk2", "clk3", "y1"):
            trigger_signal = test_data_df[test_data_df["Ampl"] > voltage_level]
            trigger_signal = trigger_signal.iloc[[-1]] 
            x1 = float(trigger_signal.Time)
            trigger2_signal = test_data_df[test_data_df["Ampl"] > voltage_level]
            trigger2_signal = trigger2_signal.iloc[[1]] 
            x2 = float(trigger2_signal.Time) 
        else:
            trigger_signal = test_data_df[test_data_df["Ampl"] > voltage_level]
            trigger_signal = trigger_signal.iloc[[-1]] 
            x1 = float(trigger_signal.Time)
            trigger2_signal = test_data_df[test_data_df["Ampl"] < voltage_level]
            trigger2_signal = trigger2_signal.iloc[[1]] 
            x2 = float(trigger2_signal.Time) 
        i += 1
        axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol)) 
        fig.suptitle('Gate Driver Timing', fontweight="bold", fontsize=20, color="red")
        plt.xlabel("Time(us)")
        plt.ylabel("Voltage(V)")

        axs[i].annotate(s='', xy=(0,voltage_level), xytext=(x1,voltage_level), arrowprops=dict(arrowstyle='<->'))
        axs[i].vlines(x=x1, ymin=min, ymax=max, colors='green', ls=':', lw=2)
        axs[i].annotate(s='', xy=(x1,voltage_level - 2 ), xytext=(x2,voltage_level - 2), arrowprops=dict(arrowstyle='<->'))
        #axs[i].vlines(x=x2, ymin=min, ymax=max, colors='green', ls=':', lw=2)
        axs[i].annotate(f"{(x1):.0f}", (2,voltage_level-14), fontsize=10, color="red")
        axs[i].annotate(f"{(x2-x1):.0f}", (x2-5,voltage_level-14), fontsize=10, color="red")

        if file_name == "y1":
            axs[i].set_ylim([0, 5])
            axs[i].annotate(f"{(x1):.0f}", (0,voltage_level-0.5), fontsize=11, color="red")
            axs[i].annotate(f"{(x2-x1):.0f}", (x2+2,voltage_level+0.5), fontsize=11, color="red")
            #axs[i].annotate(s='', xy=(x1,voltage_level), xytext=(x2,voltage_level), arrowprops=dict(arrowstyle='<->'))
        else:
            axs[i].set_ylim([-20, 25])      
        plt.xticks(fontsize=10)
        axs[i].legend() #label=file_name
        #axs[i].legend(loc = 'best')
        plt.subplots_adjust(right=.85) 
        plt.rc('legend',**{'fontsize':8})
        axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
        axs[i].grid(True)


# In[404]:


def spectrum():
    plt.close()
    global folder_name
    with PdfPages(results_dir + folder_name + ".pdf") as pdf:
        for csv_file in sorted(dirs):
            plt.close()
            test_data_to_load = os.path.join(path,csv_file)
            if not csv_file.endswith('csv'):
                # File doesn't end with this extension then ignore it
                continue
            #plt.figure(figsize=(11,6)) #multiple figure(1)
            file_name, file_extension = os.path.splitext(csv_file)
            test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows, keep header
            if folder_name.find('120Hz') != -1:
                #test_data_df = test_data_df[120000:-750000]
                signal = test_data_df["Ampl"]
                time = test_data_df["Time"]   
            elif folder_name.find('60Hz') != -1:
                test_data_df = test_data_df[120000:-770000]
                signal = test_data_df["Ampl"]
                time = test_data_df["Time"]     
            elif folder_name.find('30Hz') != -1:
                test_data_df = test_data_df[120000:-850000]
                signal = test_data_df["Ampl"]
                time = test_data_df["Time"]        
            elif folder_name.find('24Hz') != -1:
                test_data_df = test_data_df[120000:-850000]
                signal = test_data_df["Ampl"]
                time = test_data_df["Time"] 
            signal = np.array(signal, dtype=float)
            N = len(signal) # total sample rate
            secs = time.iloc[-1] - time.iloc[0] #total time
            freg = 1/float(secs) 
            fs_rate = N * freg / 1000000 #sample frequency
            Ts = 1/fs_rate #Timestep between samples
            p = Periodogram(signal, sampling=fs_rate, NFFT=N, scale_by_freq=False, window='bartlett_hann')
            p.run()
            #p2 = pyule(signal, 140, sampling=fs_rate, scale_by_freq=False)
            fig = plt.figure(figsize=(4,6))
            if file_name not in ("ls_init", "cd_vb", 'mbc'):
                fig.subplots_adjust(right=1.5)
                fig.subplots_adjust(top=1.5)
                ax = fig.add_subplot(311)
                ax1 = fig.add_subplot(312)
                ax2 = fig.add_subplot(313)
                #p.plot(marker='o',ax=ax1, rasterized=True) 
                #p.plot(marker='o',ax=ax2, rasterized=True) 
                p.plot(linewidth=0.2,ax=ax1, rasterized=True) 
                p.plot(linewidth=0.2,ax=ax2, rasterized=True) 
                #p2.plot(ax=ax1, rasterized=True)
                #p2.plot(ax=ax2, rasterized=True) 
                ax.plot(time, signal, 'b', linewidth=0.2, rasterized=True)
                ax1.set_xlim(0, 0.5)
                ax2.set_xlim(0, 5)
                ax1.legend(['0 - 500KHz'])
                ax2.legend(['0 - 5MHz'])
                ax.set_title(f"{folder_name}_{file_name}")
                plt.grid(True)
                ax.grid(True)
                ax1.set_xlabel('Frequency (MHz)')
                ax2.set_xlabel('Frequency (MHz)')
                pdf.savefig(dpi=200, bbox_inches='tight')   
                plt.close()


# In[405]:


def fft():
    plt.close()
    global folder_name
    with PdfPages(results_dir + folder_name + ".pdf") as pdf:
        for csv_file in sorted(dirs):
            plt.close()
            test_data_to_load = os.path.join(path,csv_file)
            if not csv_file.endswith('csv'):
                # File doesn't end with this extension then ignore it
                continue
            #plt.figure(figsize=(11,6)) #multiple figure(1)
            file_name, file_extension = os.path.splitext(csv_file)
            test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows, keep header
            if folder_name.find('120Hz') != -1:
                test_data_df = test_data_df[120000:-750000]
                signal = test_data_df["Ampl"]
                time = test_data_df["Time"]
                
            elif folder_name.find('60Hz') != -1:
                test_data_df = test_data_df[120000:-770000]
                signal = test_data_df["Ampl"]
                time = test_data_df["Time"]
                
            elif folder_name.find('30Hz') != -1:
                test_data_df = test_data_df[120000:-850000]
                signal = test_data_df["Ampl"]
                time = test_data_df["Time"] 
                
            elif folder_name.find('24Hz') != -1:
                test_data_df = test_data_df[120000:-850000]
                signal = test_data_df["Ampl"]
                time = test_data_df["Time"]
                
            signal = np.array(signal, dtype=float)
            N = len(signal) - 1 # total sample rate
            secs = time.iloc[-1] - time.iloc[0] #total time
            freg = 1/float(secs) 
            fs_rate = N * freg # sample frequency
            Ts = 1/fs_rate # Timestep between samples
            #Ts2 = time.iloc[1] - time.iloc[0] #Ts = Ts2
            #t = np.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
            #t = t[:-1]
            #t = t * 1000
            fft = np.fft.fft(signal)
            FFT = abs(fft[0:N//2]) / 1000 #N//2 is for positive axis
            freqs = fftfreq(N, Ts)[:N//2] / 1000000 # megahertz (MHz) range
            #FFT = abs(fft)
            #freqs = fftfreq(N, Ts) / 1000000
            fig = plt.figure(figsize=(7, 10))
            #print(freg, N, fs_rate, secs, Ts)
            if file_name not in ("ls_init", "cd_vb", 'mbc'):
                plt.subplot(311)#311
                p1 = plt.plot(time, signal, "b", rasterized=True) # plotting the signal
                plt.title(f"{folder_name}_{file_name}")
                plt.xlabel('Time(s)')
                plt.ylabel('Voltage(V)')
                plt.grid(True)
                plt.subplot(312)#312
                p2 = plt.plot(freqs, FFT, ".-", rasterized=True) # plotting the complete fft spectrum
                plt.xlabel('Frequency(0 - 0.5MHz)')
                plt.ylabel('Amplitude')
                plt.xlim([0,0.5])
                if file_name in ("avddn", "avddp"):
                    plt.ylim([0, 0.4])
                else:
                    plt.ylim([0, 0.1])
                plt.grid(True)
                plt.subplot(313)#312
                p3 = plt.plot(freqs, FFT, ".-", rasterized=True) # plotting the complete fft spectrum
                plt.xlabel('Frequency(0 - 5MHz)')
                plt.ylabel('Amplitude')
                plt.xlim([0,5])
                if file_name in ("avddn", "avddp"):
                    plt.ylim([0, 0.4])
                else:
                    plt.ylim([0, 0.1])
                plt.grid(True)
                pdf.savefig(dpi=200, bbox_inches='tight')  
                plt.close()


# In[406]:


def histogram():
    plt.close()
    plt.figure(figsize=(7,7))
    with PdfPages(results_dir + folder_name + ".pdf") as pdf:
        for csv_file in sorted(dirs):
            test_data_to_load = os.path.join(path,csv_file)
            if not csv_file.endswith('csv'):
                # File doesn't end with this extension then ignore it
                continue
            file_name, file_extension = os.path.splitext(csv_file)
            test_data_df = pd.read_csv(test_data_to_load,skiprows=4)
            test_data_df["Time"] = test_data_df["Time"] * 1000
            total = test_data_df["Time"] * test_data_df["Ampl"]
            test_data_df1 = test_data_df[(test_data_df != 0).all(1)] #delete all rows with zero value
            min = test_data_df1["Time"].min()
            mean = total.sum()/test_data_df["Ampl"].sum()
            dev = ((test_data_df["Time"] - mean) * test_data_df["Ampl"])**2
            var = dev.sum()/test_data_df["Ampl"].sum()
            max = test_data_df1["Time"].max()
            std = math.sqrt(var)
            
            x1 = test_data_df["Time"].iloc[0]
            x2 = test_data_df["Time"].iloc[20]
            x3 = test_data_df["Time"].iloc[40]
            try:
                x4 = test_data_df["Time"].iloc[60]
                x5 = test_data_df["Time"].iloc[80]
                x6 = test_data_df["Time"].iloc[99]
            except:
                pass
            textstr = '\n'.join((
            f"Min = {min:.3f}",
            f"Mean = {mean:.3f}",
            f"Max = {max:.3f}",
            f"Std = {std:.3f}"))
            test_data_df.plot.bar("Time", "Ampl", legend=None)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            #plt.text(5, 95, textstr, fontsize=10, verticalalignment='top', bbox=props)
            plt.annotate(textstr, xy=(0.75, 0.75), fontsize=9, xycoords='axes fraction', bbox=props)
            plt.xlabel("Time(ms)")
            plt.ylabel("Count")
            plt.title(file_name)
            plt.xticks(fontsize=6)
            #plt.ylim([0, 120])
            #plt.xticks([])
            plt.xticks([0, 20, 40, 60, 80, 99],[f"{x1:.3f}", f"{x2:.3f}", f"{x3:.3f}",f"{x4:.3f}", f"{x5:.3f}", f"{x6:.3f}"], rotation=-0)
            #plt.xticks([0, 50, 100],[f"{min:.2f}", f"{mean:.2f}", f"{max:.2f}"], rotation=-0)
            #plt.xticks([f"{x1:.2f}", f"{x3:.2f}"], rotation=-0)
            #plt.tight_layout() #x-axis labels so that it fit into the box holding the plot.
            #plt.legend()
            plt.grid(True)
            pdf.savefig()    
            plt.close()


# In[407]:


def y1():
    plt.close()
    plt.figure(figsize=(9,6)) #all in one figure(1)
    cycol = cycle('bgrcmk')
    for csv_file in sorted(dirs): 
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        #plt.figure(figsize=(11,6)) #multiple figure(1)
        file_name, file_extension = os.path.splitext(csv_file)
        if file_name in ("clk1", "y1"):
            test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
            #test_data_df = test_data_df[9990:-89990]
            test_data_df["Time"] = test_data_df["Time"] * 1000000   
            plt.plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol))      
        plt.xlabel("Time(us)")
        plt.ylabel("Voltage(V)")
        if folder_name.find('SHP') != -1:
            plt.xlim([-28, -17]) #J316
        elif folder_name.find('LGD') != -1:
            plt.xlim([-16, -5]) #J316
        elif test_data_df["Ampl"].iloc[46000] > 2:
            plt.xlim([-46, -32]) #BOE
        else:
            plt.xlim([-57, -45]) #LGD
        #plt.ylim([-15, 25])
        #plt.title(path + file_name) #multiple figure(2)
        plt.title(folder_name) #single figure(2)
        plt.xticks(fontsize=10)
        #plt.legend() #label=file_name
        plt.legend(loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0) #legend outside
        plt.grid(True)
        plt.subplots_adjust(right=.8) 
        #plt.savefig('Output.pdf', dpi=300, bbox_inches='tight') #single figure(3)
    plt.show()
    #plt.close()


# In[ ]:





# In[408]:


while True:
    plt.close()
    path = input("Enter the path of your CSV_files: ").strip()
    folder_name = os.path.split(os.path.abspath(path))[-1] 
    path = path + '/'
    try:
        rename3(path,"C")     
    except:
        pass
    try:
        rename2(path,"C")
    except:
        pass
    try:
        rename()   
    except:
        print(f"Can not rename the files due to '{folder_name}' folder doesn't match. Please check!")
    dirs = os.listdir(path)
    p = os.path.split(os.path.abspath(path))[0] + '/'
    results_dir = os.path.join(p, 'Results/')
    found = False
    list1 = [ "Option 1: Plot all signals",
              "Option 2: Zoom In View",
              "Option 3: Plot specific signals",
              "Option 4: Scope View all signals",
              "Option 5: Time from trigger",
              "Option 6: A signal Rising/Falling",
              "Option 7: Time betwwen two signals",
              "Option 8: All measurements",
            ]
    list2 = [ "Option 1: Plot all signals",
              "Option 2: Zoom In View",
              "Option 3: Plot specific signals",
              "Option 4: Scope View all signals",
              "Option 5: Time from trigger",
              "Option 6: A signal Rising/Falling",
              "Option 7: Time betwwen two signals",
            ]
    list3 = [ "Option 1: Plot all signals",
              "Option 2: Zoom In View",
              "Option 3: Plot specific signals",
              "Option 4: Scope View all signals",
              "Option 5: Charge sharing shmoo",
            ]
    list4 = [ "Option 1: Plot all signals",
              "Option 2: Zoom In View",
              "Option 3: Plot specific signals",
              "Option 4: Scope View all signals",
              "Option 5: eof timing",
              "Option 6: Gate driver timing",
            ]
    list5 = [ "Option 1: Plot all signals",
              "Option 2: Zoom In View",
              "Option 3: Plot specific signals",
              "Option 4: Scope View all signals",
              "Option 5: CDIC Timing",
              "Option 6: cdic vs clk1",
            ]
    list6 = [ "Option 1: Plot all signals",
              "Option 2: Zoom In View",
              "Option 3: Plot specific signals",
              "Option 4: Scope View all signals",
              "Option 5: hvlfs timing",
            ]
    list7 = [ "Option 1: Plot all signals",
              "Option 2: Zoom In View",
              "Option 3: Plot specific signals",
              "Option 4: Scope View all signals",
              "Option 5: Voltage Ripple Measurements",
            ]
    list8 = [ "Option 1: Plot all signals",
              "Option 2: Zoom In View",
              "Option 3: Plot specific signals",
              "Option 4: Scope View all signals",
              "Option 5: HVLFS Timing",
            ]
    list9 = [ "Option 1: Plot all signals",
              "Option 2: Zoom In View",
              "Option 3: Plot specific signals",
              "Option 4: Scope View all signals",
              "Option 5: Vcom Ripple",
            ]
    list10 = ["Option 1: Plot all signals",
              "Option 2: Zoom In View",
              "Option 3: Plot specific signals",
              "Option 4: Scope View all signals",
            ]

    while True and len(path) > 3:
        plt.figure(figsize=(9,5)) #all in one figure(1)
        legend_properties = {'weight':'bold'}
        trigger_x = 0
        trigger_y = 0
        time = 0
        offset = 0
        if folder_name.find('Hist') != -1:
            histogram()
            plt.close()
            print (blue(f"{folder_name}.pdf is saved in {results_dir}"))
            break
        elif folder_name.find('Spectrum') != -1:
            spectrum()
            plt.close()
            print (blue(f"{folder_name}.pdf is saved in {results_dir}"))
            break
        elif folder_name.find('AnalogPower') != -1 or folder_name.find('DigitalPower') != -1:
            print(blue(f"The following options are available for {folder_name}:"))
            print(*list1,sep='\n')
            func = input("Please input the option #: ")
            print("")
        elif folder_name.find('On') != -1 or folder_name.find('Off') != -1:
            print(blue(f"The following options are available for {folder_name}:"))
            print(*list2,sep='\n')
            func = input("Please input the option #: ")
            print("")  
        elif folder_name.find('GateChargeSharing') != -1 :
            print(blue(f"The following options are available for {folder_name}:"))
            print(*list3,sep='\n')
            func = input("Please input the option #: ")
            print("")
        elif folder_name.find('GateDriverTiming') != -1:
            print(blue(f"The following options are available for {folder_name}:"))
            print(*list4,sep='\n')
            func = input("Please input the option #: ")
            print("")
        elif folder_name.find('CDICOutputTiming') != -1:
            print(blue(f"The following options are available for {folder_name}:"))
            print(*list5,sep='\n')
            func = input("Please input the option #: ")
            print("")
        elif folder_name.find('HVLFS') != -1:
            print(blue(f"The following options are available for {folder_name}:"))
            print(*list6,sep='\n')
            func = input("Please input the option #: ")
            print("")
        elif folder_name.find('Ripple') != -1:
            print(blue(f"The following options are available for {folder_name}:"))
            print(*list7,sep='\n')
            func = input("Please input the option #: ")
            print("")
        elif folder_name.find('Ripple') != -1:
            print(blue(f"The following options are available for {folder_name}:"))
            print(*list8,sep='\n')
            func = input("Please input the option #: ")
            print("")
        elif folder_name.find('Vcom') != -1:
            print(blue(f"The following options are available for {folder_name}:"))
            print(*list9,sep='\n')
            func = input("Please input the option #: ")
            print("")
        else:
            print(blue(f"The following options are available for {folder_name}:"))
            print(*list10,sep='\n')
            func = input("Please input the option #: ")
            print("")
        
        if func == "": #hit enter to quit function while loop(enter is len = 1) 
            plt.close()
            break
        for csv_file in sorted(dirs): #sort number first  
            #file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]) 
            #print(file_count)
            test_data_to_load = os.path.join(path,csv_file)
            if not csv_file.endswith('csv'):
                continue  # ignore it if not csv files
            file_name, file_extension = os.path.splitext(csv_file)
            if file_name not in ("edp_2", "edp_3", "3v3_2"):
                try:
                    test_data_df = pd.read_csv(test_data_to_load,skiprows=4)
                    test_data_df["Time"] = test_data_df["Time"] * 1000  
                    min = test_data_df["Ampl"].min()
                    mean = test_data_df["Ampl"].mean()
                    max = test_data_df["Ampl"].max()
                    try:
                        trigger_signal = test_data_df[test_data_df["Ampl"] > (max*85/100)] #trigger positive 
                        trigger_signal = trigger_signal.iloc[[1]] #Power On
                        trigger_x = float(trigger_signal.Time)
                        trigger2_signal = test_data_df[test_data_df["Ampl"] < (max*15/100)] #trigger negative 
                        trigger2_signal = trigger2_signal.iloc[[1]] #Power Off
                        trigger2_x = float(trigger2_signal.Time)
                        x = np.isclose(trigger_x, -0, atol=0.1).any() #close to zero with torrance =True (Rising)
                        y = np.isclose(trigger2_x, -0, atol=0.1).any() #close to zero with torrance =True (Falling)
                        if x == True and func != "":
                            print(f"Trigger on {file_name}")
                        elif y == True and func != "":
                            print(f"Trigger on {file_name}")
                    except:
                        if not found:
                            found = True
                except:
                    pass        
            else:
                continue 

            if func not in ("1", "2", "3", "4", "5", '6','7', '8'):
                plt.close()
                print(red("Typo! Please try again."))
                break
            elif func == "1":
                plot()
            elif func == "2" and (folder_name.find('On') != -1):
                zoomin()
                break
            elif func == "2" and (folder_name.find('Off') != -1):
                zoomin()
                break
            elif func == "2":
                zoomin()
                break
            elif func == "3":
                plotsignals()
                break
            elif func == "4":
                scopeview()
                break
            elif func == "5" and (folder_name.find('On') != -1):
                meas_power_on()
                break
            elif func == "5" and (folder_name.find('Off') != -1):
                meas_power_off()
                break
            elif func == "6" and (folder_name.find('On') != -1):
                signal_rising()
                break
            elif func == "6" and (folder_name.find('Off') != -1):
                signal_falling()
                break
            elif func == "7" and (folder_name.find('On') != -1):
                two_signals_on()
                break
            elif func == "7" and (folder_name.find('Off') != -1):
                two_signals_off()
                break     
            elif func == "8" and (folder_name.find('On') != -1):
                time, edge = rise_time()
                plot_rise() 
            elif func == "8" and (folder_name.find('Off') != -1):
                try:
                    time, edge = fall_time()
                except:
                    print(f"{file_name} is NG")
                plot_fall()     
            elif func == "2" and (folder_name.find('GateChargeSharing') != -1):
                zoomin()
                break 
            elif func == "3" and (folder_name.find('GateChargeSharing') != -1):
                plotsignals()
                break
            elif func == "4" and (folder_name.find('GateChargeSharing') != -1):
                scopeview()
                break
            elif func == "5" and (folder_name.find('GateChargeSharing') != -1):
                chargesharing()
                break    
            elif func == "2" and (folder_name.find('GateDriverTiming') != -1):
                zoomin()
                break 
            elif func == "3" and (folder_name.find('GateDriverTiming') != -1):
                plotsignals()
                break
            elif func == "4" and (folder_name.find('GateDriverTiming') != -1):
                scopeview()
                break
            elif func == "5" and (folder_name.find('GateDriverTiming') != -1):
                eof()
                break   
            elif func == "6" and (folder_name.find('GateDriverTiming') != -1):
                gate_driver_timing()
                break
            elif func == "2" and (folder_name.find('CDICOutputTiming') != -1):
                zoomin()
                break 
            elif func == "3" and (folder_name.find('CDICOutputTiming') != -1):
                plotsignals()
                break
            elif func == "4" and (folder_name.find('CDICOutputTiming') != -1):
                scopeview()
                break
            elif func == "5" and (folder_name.find('CDICOutputTiming') != -1):
                cdic_timing()
                break   
            elif func == "6" and (folder_name.find('CDICOutputTiming') != -1):
                y1()
                break
            elif func == "2" and (folder_name.find('CDICOutputTiming') != -1):
                zoomin()
                break 
            elif func == "3" and (folder_name.find('CDICOutputTiming') != -1):
                plotsignals()
                break
            elif func == "4" and (folder_name.find('CDICOutputTiming') != -1):
                scopeview()
                break
            elif func == "5" and (folder_name.find('CDICOutputTiming') != -1):
                hvlfs()
                break   
            elif func == "2" and (folder_name.find('Ripple') != -1):
                zoomin()
                break 
            elif func == "3" and (folder_name.find('Ripple') != -1):
                plotsignals()
                break
            elif func == "4" and (folder_name.find('Ripple') != -1):
                scopeview()
                break
            elif func == "5" and (folder_name.find('Ripple') != -1):
                voltage_ripple()
                break   
            elif func == "2" and (folder_name.find('HVLFS') != -1):
                zoomin()
                break 
            elif func == "3" and (folder_name.find('HVLFS') != -1):
                plotsignals()
                break
            elif func == "4" and (folder_name.find('HVLFS') != -1):
                scopeview()
                break
            elif func == "5" and (folder_name.find('HVLFS') != -1):
                hvlfs()
                break   
            elif func == "2" and (folder_name.find('Vcom') != -1):
                zoomin()
                break 
            elif func == "3" and (folder_name.find('Vcom') != -1):
                plotsignals()
                break
            elif func == "4" and (folder_name.find('Vcom') != -1):
                scopeview()
                break
            elif func == "5" and (folder_name.find('Vcom') != -1):
                vcom_char()
                break   
            else:
                plot_time()
            plt.ylabel("Voltage(V)")  
            plt.xticks(fontsize=10)
            plt.grid(True)
            if func == "8":
                plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=1)
                plt.rc('legend',**{'fontsize':8})
                plt.subplots_adjust(top=.60)   
            else:
                plt.title(f"{folder_name}", fontweight="bold") 
                plt.subplots_adjust(right=.73) 
                plt.legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0) #lg outside
                plt.rc('legend',**{'fontsize':8})
        plt.show()
        plt.close()
    if len(path) < 2: #hit enter to quit csv files while loop(enter is len = 1) 
        break   

