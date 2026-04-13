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

This script is function is just a comprobation, what it will do is confirm that the path to the 
patient data is well confiured and confirm all the files resides inside a single patient
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

    #this will execute if the folder was found 
    print("=" * 90)
    print(f"EXPLORING PATIENT: {patient_id}")
    print("=" * 90)

    #This segment will list all the files found inside the folder 
    all_files = sorted(os.listdir(patient_dir))
    print(f"\nFiles founded:{len(all_files)}")

    #Separate files by each type. 
    txt_files = [f for f in all_files if f.endswith('.txt')]
    hea_files = [f for f in all_files if f.endswith('.hea')]
    mat_files = [f for f in all_files if f.endswith('.mat')]

    print(f"TXT files founded (.txt): {len(txt_files)}")
    print(f"HEA files founded (.hea): {len(hea_files)}")
    print(f"MAT files founded (.mat): {len(mat_files)}")

    return patient_dir , hea_files, txt_files

"""
========================================================================================
STEP 2: READING THE PATIENT CLINIC METADATA 

This script will read the .txt file of the patient and understand it.
========================================================================================
"""

def read_patient_metadata(patient_dir, patient_id):

    metadata_path = os.path.join(patient_dir, f"{patient_id}.txt")

    if not os.path.exists(metadata_path):
        print(f"METADATA FILE NOT FOUND: {metadata_path}")
        return None
    
    metadata = {}
    print(f"\n{'='*70}")
    print("PATIENT CLINIC METADATA")
    print(f"\n{'='*70}")

    with open(metadata_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                #this creates the dict extracting the patients data, as the file has the format "data:value"
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                metadata[key] = value
                print(f" {key:20s}: {value}")
    
    #now it's important to interpret the information forthe future model

    if 'Outcome' in metadata or 'CPC' in metadata:
        cpc_key = 'CPC' if 'CPC' in metadata else 'Outcome'
        try:
            cpc = int(metadata[cpc_key])
            outcome = "GOOD" if cpc <= 2 else "POOR"
            print(f"\n -> For the binary model: {outcome}")
            print(f"    CPC {cpc} -> Tag = {1 if cpc <= 2 else 0}")
        except ValueError:
            print(f"\n CPC IS NOT A NUMERIC VALUE: {metadata[cpc_key]}")

        return metadata

"""
THIRD STEP READ A EEG SIGNAL SEGMENT

What we will do on this function is open a wfdb file that is the format created by physionet and not .edf

"""

def read_eeg_segment(patient_dir, hea_files):

    # verification of .hea files in patient dir
    if not hea_files:
        print("FILES .HEA NOT FOUND FOR THIS PATIENT")
        return None, None
    
    #filering hea files to contain ONLY EEG files
    eeg_hea_files = [f for f in hea_files if 'EEG' in f.upper() or '_eeg' in f.lower()]

    #case scenario where there's not EEG files in patient dir, take all the other .hea files
    if not eeg_hea_files:
        eeg_hea_files = hea_files
    
    #sorting files to get the first temporal segment
    eeg_hea_files.sort()
    first_file = eeg_hea_files[0]

    #we separate teh .hea extension for the future wfdb creation
    record_name = first_file.replace('.hea', '')
    record_path = os.path.join(patient_dir, record_name)







