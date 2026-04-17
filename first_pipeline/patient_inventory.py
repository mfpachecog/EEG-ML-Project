
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
    
    # PART 2: EEG TECHNICAL INFORMATION
    
    #first we have to count all the available EEG segments
    all_files = os.listdir(patient_dir)
    hea_files = sorted([f for f in all_files if f.endswith('.hea')])
    eeg_hea_files = [f for f in hea_files if 'EEG' in f.upper()]

    patient_info['n_eeg_segments'] = len(eeg_hea_files)
    patient_info['n_total_files'] = len(all_files)

    #Now we have to extract the available hours from the file names
    #format: {patient_id}_{segment}_{hour}_EEG.hea
    hours_available = []
    for f in eeg_hea_files:
        parts = f.replace('hea', '').split('_')
        if len(parts) >= 3:
            try:
                hour = int(parts[2])
                hours_available.append(hour)
            except ValueError:
                pass
    
    if hours_available:
        patient_info['hours_available'] = sorted(set(hours_available))
        patient_info['min_hour'] = min(hours_available)
        patient_info['max_hour'] = max(hours_available)
        patient_info['n_unique_hours'] = len(set(hours_available))
    else:
        patient_info['hours_available'] = []
        patient_info['min_hour'] = None
        patient_info['max_hour'] = None
        patient_info['n_unique_hours'] = 0

