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

"""
==========================================================================================
FIRST STEP: UNDERSTAND THE STRUCTURE OF THE PATIENT FILES
==========================================================================================
"""
def explore_patient_files(data_dir, patient_id):

    patient_dir = os.path.join(data_dir, patient_id)

    #if segment in the case the folder is not found
    if not os.path.exists(patient_dir):
        print(f"X ERROR: the folder {patient_dir} was not found")
        print(f"Please verify that DATA_DIR & PATIENT_ID are correct")
        print(f"Folder searched: {patient_dir}")
        sys.exit(1)
        







