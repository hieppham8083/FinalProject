#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from time import sleep
import math
import random
from numba import jit
from spectrum import *
from pylab import *
from MergeAll import *
from itertools import cycle
from datetime import datetime
from scipy.fft import fft, fftfreq
from matplotlib.backends.backend_pdf import PdfPages
import warnings #fixed any warning in terminal
import matplotlib.cbook
# Ignore DtypeWarnings from pandas' read_csv
warnings.filterwarnings('ignore', message="^Columns.*")
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


# In[2]:


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


# In[3]:


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


# In[4]:


def plot_gatedriver():
    plt.plot(test_data_df["Time"], test_data_df["Ampl"], label=f"VST1 To {file_name}: {time:.1f}(us)", rasterized=True) 
    plt.xlabel("Time(us)")
    plt.ylabel("Voltage(V)")


# In[5]:


def plot_rise():
    if mean > 0 and (folder_name.find('Digital') != -1):  
        plt.plot(test_data_df["Time"],test_data_df["Ampl"],label=f"Trigger To {file_name}: {time:.2f}(ms), & Rise time: {edge:.3f}(ms)", rasterized=True)
    elif mean > 0:
         plt.plot(test_data_df["Time"],test_data_df["Ampl"],label=f"Trigger To {file_name}: {time:.0f}(ms), & Rise time: {edge:.1f}(ms)", rasterized=True)
    else:
        plt.plot(test_data_df["Time"],test_data_df["Ampl"],label=f"Trigger To {file_name}: {time:.0f}(ms), & Fall time: {edge:.1f}(ms)", rasterized=True)
    plt.xlabel("Time(ms)")
    plt.ylabel("Voltage(V)")


# In[6]:


def plot_fall():
    if (folder_name.find('Digital') != -1):  
        plt.plot(test_data_df["Time"],test_data_df["Ampl"],label=f"Trigger To {file_name}: {time:.2f}(ms), & fall time: {edge:.3f}(ms)", rasterized=True)
    elif mean < 0:
         plt.plot(test_data_df["Time"],test_data_df["Ampl"],label=f"Trigger To {file_name}: {time:.0f}(ms), & Rise time: {edge:.1f}(ms), rasterized=True")
    else:
        plt.plot(test_data_df["Time"],test_data_df["Ampl"],label=f"Trigger To {file_name}: {time:.0f}(ms), & Fall time: {edge:.1f}(ms)", rasterized=True)
    plt.xlabel("Time(ms)")
    plt.ylabel("Voltage(V)")


# In[7]:


def zoominview():
    plt.close()
    list = os.listdir(path) 
    file_count = len(list)
    fig, axs = plt.subplots(file_count, sharex=True, sharey=False, figsize=(8,6))
    i = -1
    cycol = cycle('bgrcmk')
    for csv_file in sorted(dirs):
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        file_name, file_extension = os.path.splitext(csv_file)
        test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
        test_data_df["Time"] = test_data_df["Time"] * 1000
        i += 1  
        if file_name.startswith('C'):
            file_name = file_name[3:]
        axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol),rasterized=True) 
        fig.suptitle(f"Zoom-In View_{folder_name}", fontweight="bold", fontsize=16, color="red")
        plt.xlabel("Time(ms)")
        plt.ylabel("Voltage(V)")
        plt.xticks(fontsize=10)
        if folder_name.find('LGD'):
            plt.xlim([-0.015, 0.05])
        elif folder_name.find('SHP'):
            plt.xlim([-0.015, 0.03])
        axs[i].legend() 
        #axs[i].legend(loc = 'best')
        plt.subplots_adjust(right=.85) 
        plt.rc('legend',**{'fontsize':8})
        axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
        axs[i].grid(True)


# In[8]:


def scopeview():
    plt.close()
    list = os.listdir(path) 
    file_count = len(list)
    if folder_name.find('System') != -1:
        fig, axs = plt.subplots(file_count - 4, sharex=True, sharey=False, figsize=(8,6))
    else:
        fig, axs = plt.subplots(file_count, sharex=True, sharey=False, figsize=(8,6))
    i = -1
    cycol = cycle('bgrcmk')
    for csv_file in sorted(dirs):
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            # File doesn't end with this extension then ignore it
            continue
        file_name, file_extension = os.path.splitext(csv_file)
        if file_name not in ("edp_2", "edp_3", "ls_int", "ls_vbe"):
            test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows but always keep header
            test_data_df["Time"] = test_data_df["Time"] * 1000
            i += 1  
            if file_name.startswith('C'):
                file_name = file_name[3:]
            axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol),rasterized=True) 
            fig.suptitle(f"Scope View_{folder_name}", fontweight="bold", fontsize=16, color="red")
            plt.xlabel("Time(ms)")
            plt.ylabel("Voltage(V)")
            plt.xticks(fontsize=10)
            axs[i].legend() 
            #axs[i].legend(loc = 'best')
            plt.subplots_adjust(right=.85) 
            plt.rc('legend',**{'fontsize':8})
            axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
            axs[i].grid(True)


# In[9]:


def chargesharing():
    plt.close()
    #plt.figure(figsize=(12,6)) #all in one figure(1)
    offset = -9.9
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5)) #subplots(1,2) means fig left and right
    clk1_data_to_load = os.path.join(path,"clk1.csv")
    clk1_data_df = pd.read_csv(clk1_data_to_load,skiprows=4)
    clk1_data_df = clk1_data_df[100000:-890000] 
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
        test_data_df = pd.read_csv(test_data_to_load,skiprows=4)
        test_data_df["Time"] = test_data_df["Time"] * 1000000
        offset = offset + 9.9
        test_data_df["Time"] = test_data_df["Time"] - offset
        #ax1.set_title(f"{name} - Rising Edge")
        ax1.plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, rasterized=True)
        #ax2.set_title(f"{name} - Falling Edge")
        ax2.plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, rasterized=True) 
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
    #print(name)
    plt.savefig(results_dir + folder_name + ".pdf", dpi=200, bbox_inches='tight') #single figure(3) 
    #plt.show()
    plt.close()
    print(f"{folder_name} is done!")


# In[10]:


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
            axs[i].annotate(s='', xy=(0,voltage_level), xytext=(x1,voltage_level),arrowprops=dict(arrowstyle='<->'))
            axs[i].vlines(x=x1, ymin=min, ymax=max, colors='green', ls=':', lw=2)
            axs[i].annotate(s='', xy=(x1,voltage_level - 2 ), xytext=(x2,voltage_level - 2),arrowprops=dict(arrowstyle='<->'))
            axs[i].vlines(x=x2, ymin=min, ymax=max, colors='green', ls=':', lw=2)
            if file_name != "vst1":
                axs[i].annotate(f"{(x1):.2f}", (0,voltage_level-13), fontsize=8, color="red")
            axs[i].annotate(f"{(x2-x1):.2f}", (x2-5,voltage_level-13), fontsize=8, color="red")
            plt.xticks(fontsize=10)
            axs[i].legend() 
            #axs[i].legend(loc = 'best')
            plt.subplots_adjust(right=.85) 
            plt.rc('legend',**{'fontsize':8})
            axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
            axs[i].grid(True)
    plt.savefig(results_dir + "GateDriverTiming.pdf", dpi=200, bbox_inches='tight')
    #plt.savefig('Output1.pdf', dpi=300, bbox_inches='tight') #single figure(3)
    plt.close()
    print(f"{folder_name} is done!")


# In[11]:


def rename():
    #folder_name = os.path.split(os.path.abspath(path))[-1] # split folder name from the path
    #path = path + '/'
    dirs = os.listdir(path)
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


# In[12]:


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


# In[13]:


def rename2(f_path, new_name):
    filelist = glob.glob(f_path + "C*01.csv")
    count = 8
    for file in sorted(filelist):
        #print("File Count : ", count)
        filename = os.path.split(file)
        #print(filename)
        count = count + 1
        new_filename = f_path + new_name + str(count) + ".csv"
        os.rename(f_path+filename[1], new_filename)
        #print(new_filename)


# In[14]:


def spectrum():
    plt.close()
    global folder_name
    #plt.figure(figsize=(6,6))
    with PdfPages(results_dir + folder_name + ".pdf") as pdf:
        for csv_file in sorted(dirs):
            test_data_to_load = os.path.join(path,csv_file)
            if not csv_file.endswith('csv'):
                # File doesn't end with this extension then ignore it
                continue
            #plt.figure(figsize=(11,6)) #multiple figure(1)
            file_name, file_extension = os.path.splitext(csv_file)
            test_data_df = pd.read_csv(test_data_to_load,skiprows=4) # skip 4 rows, keep header
            signal = test_data_df["Ampl"]
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
            N = len(signal) # total sample rate
            secs = time.iloc[-1] - time.iloc[0] #total time
            freg = 1/float(secs) 
            fs_rate = N * freg / 1e6 #sample frequency
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
    print(f"{folder_name} is done!")


# In[15]:


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
            test_data_df1 = test_data_df[(test_data_df != 0).all(1)]
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
            plt.annotate(textstr, xy=(0.05, 0.75), fontsize=9, xycoords='axes fraction', bbox=props)
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
    print(f"{folder_name} is done!")


# In[16]:


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
                axs[i].annotate(f"{(x1):.2f}", (x1/2,voltage_level + 1), fontsize=8, color="red")
            axs[i].annotate(f"{(x2 -x1):.2f}", (x1+0.2,voltage_level - 6), fontsize=8, color="red")
            plt.xticks(fontsize=10)
            axs[i].legend() #label=file_name
            #axs[i].legend(loc = 'best')
            plt.subplots_adjust(right=.85) 
            plt.rc('legend',**{'fontsize':8})
            axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
            axs[i].grid(True)
    plt.savefig(results_dir + "eof1.pdf", dpi=200, bbox_inches='tight') #single figure(3) 
    plt.close()
    print("EOF timing is done!")


# In[17]:


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
            axs[i].annotate(f"{(x1):.2f}", (x1+0.2,-5), fontsize=11, color="red")
        elif file_name.find('hvlfs1') == -1:
            axs[i].annotate(s='', xy=(0,voltage_level), xytext=(x1,voltage_level), arrowprops=dict(arrowstyle='<->'))
            axs[i].annotate(f"{(x1):.2f}", (x1+0.2,voltage_level - 6), fontsize=11, color="red")
        plt.xticks(fontsize=10)
        axs[i].legend() #label=file_name
        #axs[i].legend(loc = 'best')
        plt.subplots_adjust(right=.85) 
        plt.rc('legend',**{'fontsize':8})
        axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
        axs[i].grid(True)
    plt.savefig(results_dir + "hvlfs.pdf", dpi=200, bbox_inches='tight') #single figure(3) 
    plt.close()
    print("HVLFS timing is done!")


# In[18]:


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
        axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol), rasterized=True) 
        fig.suptitle('Gate Driver Timing', fontweight="bold", fontsize=16, color="red")
        plt.xlabel("Time(us)")
        plt.ylabel("Voltage(V)")

        axs[i].annotate(s='', xy=(0,voltage_level), xytext=(x1,voltage_level), arrowprops=dict(arrowstyle='<->'))
        axs[i].vlines(x=x1, ymin=min, ymax=max, colors='green', ls=':', lw=2)
        axs[i].annotate(s='', xy=(x1,voltage_level - 2 ), xytext=(x2,voltage_level - 2), arrowprops=dict(arrowstyle='<->'))
        #axs[i].vlines(x=x2, ymin=min, ymax=max, colors='green', ls=':', lw=2)
        if file_name != "eof1":
            axs[i].annotate(f"{(x1):.2f}", (2,voltage_level-14), fontsize=8, color="red")
        axs[i].annotate(f"{(x2-x1):.2f}", (x2-5,voltage_level-14), fontsize=8, color="red")

        if file_name == "y1":
            axs[i].set_ylim([0, 5])
            axs[i].annotate(f"{(x1):.2f}", (0,voltage_level-0.5), fontsize=8, color="red")
            axs[i].annotate(f"{(x2-x1):.2f}", (x2+2,voltage_level+0.5), fontsize=8, color="red")
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
    plt.savefig(results_dir + "CDICOutputTiming.pdf", dpi=200, bbox_inches='tight') #single figure(3) 
    plt.close()
    print("CDICOutputTiming is done!")


# In[19]:


def y1():
    plt.close()
    plt.figure(figsize=(9,5)) #all in one figure(1)
    cycol = cycle('bgrcmk')
    global folder_name
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
            plt.plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol), rasterized=True)
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
        plt.savefig(results_dir + "CDICOutputCLK1.pdf", dpi=200, bbox_inches='tight') #single figure(3) 
    plt.close()
    print(f"{folder_name} is done!")


# In[20]:


def meas_power_on():
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
            axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol), rasterized=True) 
            fig.suptitle(f"{folder_name}", fontweight="bold", fontsize=20, color="red")
            plt.xlabel("Time(ms)")
            plt.ylabel("Voltage(V)")
            if x1 == 0:
                axs[i].annotate("", (0,max/2), fontsize=10, color="red")
            elif mean > 0 and x1 != 0:
                if folder_name.find('SystemMatch') == -1:
                    axs[i].annotate(s='', xy=(0,(max*85/100)), xytext=(x1,(max*85/100)), arrowprops=dict(arrowstyle='<->'))
                    axs[i].annotate(f"{(x1):.0f}", (2,max/2), fontsize=6, color="red")
            elif mean < 0 and x1 != 0:
                if folder_name.find('SystemMatch') == -1:
                    axs[i].annotate(s='', xy=(0,(min*85/100)), xytext=(x1,(min*85/100)), arrowprops=dict(arrowstyle='<->'))
                    axs[i].annotate(f"{(x1):.0f}", (2,min/2), fontsize=6, color="red")
            if folder_name.find('SystemMatch') == -1:  
                axs[i].vlines(x=x1, ymin=min, ymax=max, colors='green', ls=':', lw=2)
            plt.xticks(fontsize=10)
            plt.xticks(fontsize=10)
            axs[i].legend() 
            #if x1 > -100:
                #print(f"Trigger To {file_name}: {x1:.3f}(ms)")
            #axs[i].legend(loc = 'best')
            plt.subplots_adjust(right=.85) 
            plt.rc('legend',**{'fontsize':8})
            axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
            axs[i].grid(True)
            t = np.isclose(x1, -0, atol=0.4).any()
            if (x1 > -100 or x1 != 0) and t == False:
                name = folder_name + ".txt"
                with open(results_dir + "junk.txt", 'a') as output:
                    output.write('\n')
                    output.write(f"Trigger To {file_name}: {x1:.3f}(ms)")
                    output.write('\n')
    lines_seen = set()  # holds lines already seen
    outfile = open(results_dir + name, "w")
    infile = open(results_dir + "junk.txt", "r")
    for line in infile:
        if line not in lines_seen:  # not a duplicate
            outfile.write(line)
            lines_seen.add(line)
    with open(results_dir + name) as outfile:
	    updatedfile = f"{folder_name}:" + outfile.read() 
    with open(results_dir + name, 'w') as outfile:
        outfile.seek(0, 0) 
        outfile.write(updatedfile)  
        #outfile.write(f"{folder_name}:"'\n')    
    outfile.close()
    os.remove(results_dir + "junk.txt")  


# In[21]:


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
            axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol),rasterized=True) 
            fig.suptitle(f"{folder_name}", fontweight="bold", fontsize=16, color="red")
            plt.xlabel("Time(ms)")
            plt.ylabel("Voltage(V)")
            if x1 == 0:
                axs[i].annotate("", (0,max/2), fontsize=10, color="red")
            elif mean > 0:
                if folder_name.find('SystemMatch') == -1:
                    axs[i].annotate(s='', xy=(0,(max*15/100)), xytext=(x1,(max*15/100)), arrowprops=dict(arrowstyle='<->'))
                    axs[i].annotate(f"{(x1):.0f}", (0,max/2), fontsize=6, color="red")
            elif mean < 0:
                if folder_name.find('SystemMatch') == -1:
                    axs[i].annotate(s='', xy=(0,(min*15/100)), xytext=(x1,(min*15/100)), arrowprops=dict(arrowstyle='<->'))
                    axs[i].annotate(f"{(x1):.0f}", (0,min/2), fontsize=6, color="red")
            if folder_name.find('SystemMatch') == -1:
                axs[i].vlines(x=x1, ymin=min, ymax=max, colors='green', ls=':', lw=2)
            plt.xticks(fontsize=10)
            axs[i].legend() 
            #print(f"Trigger To {file_name}: {x1:.3f}(ms)")
            #axs[i].legend(loc = 'best')
            plt.subplots_adjust(right=.85) 
            plt.rc('legend',**{'fontsize':8})
            axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
            axs[i].grid(True) 
            t = np.isclose(x1, -0, atol=0.4).any()
            if (x1 > -100 or x1 != 0) and t == False:
                name = folder_name + ".txt"
                with open(results_dir + "junk.txt", 'a') as output:
                    output.write('\n')
                    output.write(f"Trigger To {file_name}: {x1:.3f}(ms)")
                    output.write('\n')
    lines_seen = set()  # holds lines already seen
    outfile = open(results_dir + name, "w")
    infile = open(results_dir + "junk.txt", "r")
    for line in infile:
        if line not in lines_seen:  # not a duplicate
            outfile.write(line)
            lines_seen.add(line)
    with open(results_dir + name) as outfile:
	    updatedfile = f"{folder_name}:" + outfile.read() 
    with open(results_dir + name, 'w') as outfile:
        outfile.seek(0, 0) 
        outfile.write(updatedfile)  
        #outfile.write(f"{folder_name}:"'\n')    
    outfile.close()
    os.remove(results_dir + "junk.txt")   


# In[22]:


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
        plt.subplots_adjust(right=.85) 
        plt.rc('legend',**{'fontsize':8})
        axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
        axs[i].grid(True) 
        name = folder_name + ".txt"
        with open(results_dir + "junk.txt", 'a') as output:
            if file_name in ("vcom1", "vcom1FB", "vcom2", "vcom2FB"):
                output.write('\n')
                output.write(f"{file_name}_Ripple(pkpk): {pkpk:.2f}(V)"'\n')
                output.write(f"{file_name}_Ripple(max): {max2:.2f}(V)"'\n')
                output.write(f"{file_name}_Ripple(min): {min2:.2f}(V)"'\n')     
                output.write('\n')
                output.write('\n')
    lines_seen = set()  # holds lines already seen
    outfile = open(results_dir + name, "w")
    infile = open(results_dir + "junk.txt", "r")
    for line in infile:
        if line not in lines_seen:  # not a duplicate
            outfile.write(line)
            lines_seen.add(line)
    with open(results_dir + name) as outfile:
	    updatedfile = f"{folder_name}:" + outfile.read() 
    with open(results_dir + name, 'w') as outfile:
        outfile.seek(0, 0) 
        outfile.write(updatedfile)  
        #outfile.write(f"{folder_name}:"'\n')    
    outfile.close()
    os.remove(results_dir + "junk.txt") 
    plt.savefig(results_dir + folder_name + ".pdf", dpi=200, bbox_inches='tight') #single figure(3) 
    plt.close()
    print(f"{folder_name} is done!")


# In[23]:


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
        min2 = test_data_df["Ampl"].min() * 1000
        max2 = test_data_df["Ampl"].max() * 1000  
        i += 1
        axs[i].plot(test_data_df["Time"], test_data_df["Ampl"], label=file_name, c=next(cycol), rasterized=True) 
        fig.suptitle(f"{folder_name}", fontweight="bold", fontsize=20, color="red")
        plt.xlabel("Time(ms)")
        plt.ylabel("Voltage(V)")
        plt.xticks(fontsize=10)
        axs[i].legend()   
        plt.subplots_adjust(right=.85) 
        plt.rc('legend',**{'fontsize':8})
        axs[i].legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1.02, 0.25), borderaxespad=0)
        axs[i].grid(True) 
        name = folder_name + ".txt"
        with open(results_dir + "junk.txt", 'a') as output:
            if file_name not in ("ls_init", "cd_vb", 'mbc'):
                output.write('\n')
                output.write(f"{file_name}_Ripple(pkpk): {pkpk:.2f}(mV)"'\n')
                output.write(f"{file_name}_Ripple(max): {max2:.2f}(mV)"'\n')
                output.write(f"{file_name}_Ripple(min): {min2:.2f}(mV)"'\n')     
                output.write('\n')
                output.write('\n')
    lines_seen = set()  # holds lines already seen
    outfile = open(results_dir + name, "w")
    infile = open(results_dir + "junk.txt", "r")
    for line in infile:
        if line not in lines_seen:  # not a duplicate
            outfile.write(line)
            lines_seen.add(line)
    with open(results_dir + name) as outfile:
	    updatedfile = f"{folder_name}:" + outfile.read() 
    with open(results_dir + name, 'w') as outfile:
        outfile.seek(0, 0) 
        outfile.write(updatedfile)  
        #outfile.write(f"{folder_name}:"'\n')    
    outfile.close()
    os.remove(results_dir + "junk.txt")  


# In[24]:


#path1 = input("Enter the path of test items: ").strip()
#path1 = path1 + '/'
#main_dirs = os.listdir(path1)
plt.figure(figsize=(9,5)) #all in one figure(1)
legend_properties = {'weight':'bold'}
found = False
a = 0
print("Testing in progress!")
for folder_name in sorted(main_dirs):
    if folder_name.startswith('.') or folder_name == "Results" or folder_name == "Temp":
       continue
    path = os.path.join(path1,folder_name)
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
        print(f"WARNING: Can not rename the files in '{folder_name}' folder. Please check!")
    dirs = os.listdir(path)
    results_dir = os.path.join(path1, 'Results/')
    #results_dir = "/Users/hiep_pham/Desktop/Results/"
    for csv_file in sorted(dirs): #sort number first  
        #file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])  
        test_data_to_load = os.path.join(path,csv_file)
        if not csv_file.endswith('csv'):
            continue  # ignore it if not csv files
        file_name, file_extension = os.path.splitext(csv_file)
        if file_name not in ("3v3_2", "3v3_3", "edp_2"):
            test_data_df = pd.read_csv(test_data_to_load,skiprows=4)
            test_data_df["Time"] = test_data_df["Time"] * 1000  
            min = test_data_df["Ampl"].min()
            mean = test_data_df["Ampl"].mean()
            max = test_data_df["Ampl"].max()
            std = test_data_df["Time"].std()
            try:
                trigger_signal = test_data_df[test_data_df["Ampl"] > (max*85/100)] #trigger positive 
                trigger_signal = trigger_signal.iloc[[1]] #Power On
                #print(trigger_signal)
                trigger_x = float(trigger_signal.Time)
                trigger_y = float(trigger_signal.Ampl)
                trigger2_signal = test_data_df[test_data_df["Ampl"] < (max*15/100)] #trigger positive 
                trigger2_signal = trigger2_signal.iloc[[1]] #Power Off
                #print(trigger2_signal)
                trigger2_x = float(trigger2_signal.Time)
                trigger2_y = float(trigger2_signal.Ampl)
                x = np.isclose(trigger_x, -0, atol=0.1).any() #close to zero with torrance =True
                y = np.isclose(trigger2_x, -0, atol=0.1).any()
                if x == True:
                    trigger_name = file_name
                    offset = trigger_x
                    trigger_y = float(trigger_signal.Ampl)
                    #print(f"Trigger on {trigger_name}")
                elif y == True:
                    trigger_name = file_name
                    offset = trigger2_x
                    trigger2_y = float(trigger2_signal.Ampl)
                    #print(f"Trigger on {trigger_name}")
            except:
                if not found:
                    found = True
        else:
            continue
        if folder_name == "AnalogPowerOff" or folder_name == "DigitalPowerOff":
            try:
                time, edge = fall_time()
            except:
                pass
        elif folder_name == "AnalogPowerOn" or folder_name == "DigitalPowerOn":
            time, edge = rise_time()
        elif folder_name.find('PowerOn') != -1:
            try:
                meas_power_on() 
            except:
                pass 
        elif folder_name.find('PowerOff') != -1:
            try:
                meas_power_off() 
            except:
                pass
        elif folder_name.find('Ripple') != -1:
            try:
                voltage_ripple() 
            except:
                pass
        elif folder_name.find('Vcom') != -1: 
            zoominview()
            name =  "VcomZoomInView" + ".pdf"
            plt.savefig(results_dir + name, dpi=200, bbox_inches='tight') #single figure(3)
            print("VcomZoomInView is done!")
            vcom_char()
            break
        elif folder_name.find('GateCharge') != -1:
            chargesharing()
            #print(f"{folder_name} is done!")
            break 
        elif folder_name.find('GateDriverTiming') != -1:
            scopeview()
            #name = folder_name + "Full" + ".pdf"
            name =  "GateDriverTimingFull" + ".pdf"
            plt.savefig(results_dir + name, dpi=200, bbox_inches='tight') #single figure(3)
            print("GateDriverTimingFull is done!")
            gate_driver_timing()
            #print(f"{folder_name} is done!")
            eof()
            break
        elif folder_name.find('HVLFS') != -1:
            hvlfs()
            break
        elif folder_name.find('CDICOutputTiming') != -1:
            cdic_timing()
            y1()
            break
        elif folder_name.find('Histogram') != -1:
            histogram()
            #print(f"{folder_name} is done!")
            break
        elif folder_name.find('Spectrum') != -1:
            spectrum()
            #print(f"{folder_name} is done!")
            break
        else: 
            scopeview()
        if folder_name.find('AnalogPower') != -1 or folder_name.find('DigitalPower') != -1:
            plot_rise()
            plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=1)
            plt.rc('legend',**{'fontsize':8})
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            plt.annotate(folder_name, xy=(0.55, 0.75), fontsize=6, xycoords='axes fraction', bbox=props)
            plt.subplots_adjust(top=.60)   
        plt.grid(True)       
    #if folder_name not in ("GateChargeSharing", "GateDriverTiming", "HistogramAnalogPower", "HistogramDigitalPower", "SourceOutputTiming"):
    if folder_name.find('Histogram') == -1 and folder_name.find('GateCharge') == -1 and folder_name.find('CDICOutputTiming') == -1 and folder_name.find('GateDriverTiming') == -1 and folder_name.find('Spectrum') == -1 and folder_name.find('Vcom') == -1:
        name = folder_name + ".pdf"
        plt.savefig(results_dir + name, dpi=200, bbox_inches='tight') #single figure(4) 
        plt.close()
        #a += 1
        #print(f"Test item{a} is done!")
        print(f"{folder_name} is done!")
    else:
        continue


# In[25]:


list_of_functions = [merge_SystemMatchPDF, merge_VoltageRipplePDF, merge_SystemMatchText, merge_VoltageRippleText, merge_AnalogDigitalPDF, merge_GateDriverTimingPDF, merge_PowerMeasPDF, merge_3v3PDF, merge_ALPMPDF, merge_PowerMeasText, merge_ChargeSharingShmooPDF, merge_HistogramPDF, merge_PowerSpectrumPDF, merge_VcomPDF]
for f in list_of_functions: #call all functions in the list
    try:
        f()
    except:
        pass
textfiles = glob2.glob(results_dir + '*.txt')
pdffiles = glob2.glob(results_dir + '*.pdf')
for file in textfiles:
    if os.path.getsize(file) < 100:
        os.remove(file)
for file in pdffiles:
    if os.path.getsize(file) < 2000:
        os.remove(file) 
print("Done Merging!")

