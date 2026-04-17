
"""
PHASE 1: INVENTORY 

OBJECTIVE: 

This script is created to collect all the patients metadata, and the EEG features before start the preprocessing phase 
what this will do is help us understand the whole picture of the patients and create a tabular resume

WHAT WE HAVE AT THE END:

1. a table with all the patients and their clinical features
2. technical information about each patient
3. Classes balance (good vs poor)
4. CSV file that we will use as reference for the WHOLE project

"""

import os
import sys 
import wfdb
import numpy as np
import pandas as pd

DATA_DIR = "/home/singular1ty/Documents/_PROJECTS/eeg-ml-project/patients_data_raw/physionet.org/files/i-care/2.1/training"

#MAIN FUNCTION ORGANIZE A SINGLE PATIENT 

def inventory_single_patient(data_dir, patient_id):

    """This script will extract the data from a single patient and create a dictionary with 
    the clinic metadata, technical EEG information & the binary tag (0 | 1) for the model"""

    patient_dir = os.path.join(data_dir, patient_id)
    patient_info = {'patient_id': patient_id}

    
