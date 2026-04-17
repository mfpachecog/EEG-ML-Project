
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

    # PART 1: CLINIC METADATA
    metadata_path = os.path.join(patient_dir, f"{patient_id}.txt")

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    patient_info[key.lower().replace(' ', '_')] = value
    else:
        print(f"WARNING {patient_id} METADATA FILE **NOT** FOUND")
        return patient_info
    
    #now create the binary tag for the model
    #cpc 1-2: good(1), cpc 3-5: poor(0)
    cpc_value = patient_info.get('cpc', None)
    if cpc_value is not None:
        try:
            cpc_num = int(cpc_value)
            patient_info['cpc_numeric'] = cpc_num
            patient_info['binary_outcome'] = 1 if cpc_num <= 2 else 0
            patient_info['outcome_label'] = 'good' if cpc_num <= 2 else 'poor'
        except ValueError:
            patient_info['cpc_numeric'] = None
            patient_info['binary_outcome'] = None
            patient_info['outcome_label'] = 'Unknown'
    
    