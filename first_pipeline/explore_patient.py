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

    return patient_dir, hea_files, txt_files




