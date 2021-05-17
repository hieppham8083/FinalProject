#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv
import os
import sys
import glob2
import os.path
from pathlib import Path
import fnmatch
from simple_colors import * # pip install simple-colors
import time
from time import sleep
import math
import random
import PyPDF2
from PyPDF2 import PdfFileMerger, PdfFileReader
from itertools import cycle
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import warnings #fixed any warning in terminal
import matplotlib.cbook
# Ignore DtypeWarnings from pandas' read_csv
warnings.filterwarnings('ignore', message="^Columns.*")
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


# In[ ]:


def merge_VcomPDF():
    merger = PdfFileMerger()
    filenames = glob2.glob(results_dir + 'Vcom*.pdf') 
    for filename in filenames:
        merger.append(filename)
        os.remove(filename)
    with open(results_dir + 'Results_VcomChar.pdf', "ab") as fout:
        merger.write(fout)
    merger.close()


# In[ ]:


def merge_HistogramPDF():
    merger = PdfFileMerger()
    filenames = glob2.glob(results_dir + 'Histogram*.pdf') 
    for filename in filenames:
        merger.append(filename)
        os.remove(filename)
    with open(results_dir + 'Results_Histogram.pdf', "ab") as fout:
        merger.write(fout)
    merger.close()


# In[ ]:


def merge_PowerSpectrumPDF():
    merger = PdfFileMerger()
    filenames = glob2.glob(results_dir + 'PowerSpectrum*.pdf') 
    for filename in filenames:
        merger.append(filename)
        os.remove(filename)
    with open(results_dir + 'Results_PowerSpectrum.pdf', "ab") as fout:
        merger.write(fout)
    merger.close()


# In[61]:


def merge_ChargeSharingShmooPDF():
    merger = PdfFileMerger()
    filenames = glob2.glob(results_dir + 'GateChargeSharing*.pdf') 
    for filename in filenames:
        merger.append(filename)
        os.remove(filename)
    with open(results_dir + 'Results_ChargeSharingShmoo.pdf', "ab") as fout:
        merger.write(fout)
    merger.close()


# In[62]:


def merge_3v3PDF():
    merger = PdfFileMerger()
    filenames = glob2.glob(results_dir + '3v3*.pdf') 
    for filename in filenames:
        merger.append(filename)
        os.remove(filename)
    with open(results_dir + 'Results_3v3.pdf', "ab") as fout:
        merger.write(fout)
    merger.close()


# In[63]:


def merge_ALPMPDF():
    merger = PdfFileMerger()
    filenames = ["ALPM_30Hz.pdf", "ALPM_60Hz.pdf"]
    for filename in filenames:
        merger.append(results_dir + filename)
        os.remove(results_dir + filename)
    with open(results_dir + 'Results_ALPM.pdf', "ab") as fout:
        merger.write(fout)
    merger.close()


# In[64]:


def merge_PowerMeasPDF():
    merger = PdfFileMerger()
    filenames = ["PowerOnTcon.pdf", "PowerOffTcon.pdf", "PowerOnCDIC.pdf", "PowerOffCDIC.pdf", "PowerOnGateDriver.pdf", "PowerOffGateDriver.pdf"]
    for filename in filenames:
        merger.append(results_dir + filename)
        os.remove(results_dir + filename)
    with open(results_dir + 'Results_PowerMeas.pdf', "ab") as fout:
        merger.write(fout)
    merger.close()


# In[65]:


def merge_GateDriverTimingPDF():
    merger = PdfFileMerger()
    filenames = ["GateDriverTimingFull.pdf", "GateDriverTiming.pdf", "eof1.pdf", "hvlfs.pdf", "CDICOutputTiming.pdf", "CDICOutputCLK1.pdf"]
    for filename in filenames:
        merger.append(results_dir + filename)
        os.remove(results_dir + filename)
    with open(results_dir + 'Results_GateDriverTiming.pdf', "ab") as fout:
        merger.write(fout)
    merger.close()


# In[66]:


def merge_AnalogDigitalPDF():
    merger = PdfFileMerger()
    filenames = ["AnalogPowerOn.pdf", "AnalogPowerOff.pdf", "DigitalPowerOn.pdf", "DigitalPowerOff.pdf"]
    for filename in filenames:
        merger.append(results_dir + filename)
        os.remove(results_dir + filename)
    with open(results_dir + 'Results_AnalogDigital.pdf', "ab") as fout:
        merger.write(fout)
    merger.close()


# In[67]:


def merge_VoltageRippleText():
    filenames = glob2.glob(results_dir + 'VoltageRipple*.txt')  # list of all .txt files in the directory
    with open(results_dir + 'Results_VoltageRipple.txt', 'a') as f:
        for file in filenames:
            with open(file) as infile:
                f.write(infile.read()+'\n')
                os.remove(file)


# In[68]:


def merge_PowerMeasText():
    filenames = glob2.glob(results_dir + 'Power*.txt')  # list of all .txt files in the directory
    with open(results_dir + 'Results_PowerMeas.txt', 'a') as f:
        for file in filenames:
            with open(file) as infile:
                f.write(infile.read()+'\n')
                os.remove(file)


# In[69]:


def merge_SystemMatchText():
    filenames = glob2.glob(results_dir + 'SystemMatch*.txt')  # list of all .txt files in the directory
    with open(results_dir + 'Results_SystemMatch.txt', 'a') as f:
        for file in filenames:
            with open(file) as infile:
                f.write(infile.read()+'\n')
                os.remove(file)


# In[70]:


def merge_VoltageRipplePDF():
    merger = PdfFileMerger()
    filenames = glob2.glob(results_dir + 'VoltageRipple*.pdf') 
    for filename in filenames:
        merger.append(filename)
        os.remove(filename)
    with open(results_dir + 'Results_VoltageRipple.pdf', "ab") as fout:
        merger.write(fout)
    merger.close()


# In[71]:


def merge_SystemMatchPDF():
    #list = os.listdir(results_dir) 
    #number_files = len(list)
    merger = PdfFileMerger()
    filenames = glob2.glob(results_dir + 'SystemMatch*.pdf') 
    for filename in filenames:
        merger.append(filename)
        os.remove(filename)
    with open(results_dir + 'Results_SystemMatch.pdf', "ab") as fout:
        merger.write(fout)
    merger.close()


# In[72]:



path1 = input("Enter the path of test items: ").strip()
path1 = path1 + '/'
main_dirs = os.listdir(path1)
for folder_name in sorted(main_dirs):
    if folder_name.startswith('.') or folder_name == "Results" or folder_name == "Temp":
       continue
    path = os.path.join(path1,folder_name)
    path = path + '/'
    dirs = os.listdir(path)
    results_dir = os.path.join(path1, 'Results/')
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

