"""
==========================================================================================

PHASE 1 = EXPLORATION - Observing first patient data

==========================================================================================

record = 31-03-26 signal preview

"""

import os 
import sys
import glob
import wfdb
import numpy as np
import matplotlib.pyplot as plt

#route to the file where the patients data is downloaded. 
DATA_DIR = "/home/singular1ty/Documents/_PROJECTS/eeg-ml-project/patients_data_raw/physionet.org/files/i-care/2.1/training"

#patient ID from the downloaded patients
PATIENT_ID = "0342"

