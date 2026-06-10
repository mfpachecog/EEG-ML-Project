
"""
====================================================================================================================================

PHASE 2 - STEP 0. CHANNELS VERIFICATION SCRIPT

====================================================================================================================================

the objective of this script is to setup a initial and crucial step, as before we start with the formal processing of the information 
we need to confirm WHAT channels each patient have, and to verify if the 19 clinical standard channels are available in all the patients. 

This is a very crucial step as we cannot fall into the bias of starting the preprocessing and asume that all the patients come with the 19 standard 
channels, bc when we ran the patient_inventory script we saw how many channels exists, but not WHAT channels are. A patient could have 19 channels, 
but not have the standards on them. For insance extra channels or bad labeled channels. 

If we start the processing assuming that and turns out at the end that there are missing channels in some patients, we will have to rewrite 
and iterate code again. 
"""

import os
import sys
import wfdb
import pandas as pd
from collections import Counter

"""
====================================================================================================================================
CONFIGURATION 
====================================================================================================================================
"""

DATA_DIR = "/home/singular1ty/Documents/_PROJECTS/eeg-ml-project/patients_data_raw/physionet.org/files/i-care/2.1/training"

# create the set variable with the 19 standard channels

STANDARD_10_20 = {
    'Fp1', 'Fp2',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    '01', '02'
}

# mapping of the old nomenclature in case it appears

MODERN_TO_OLD = {
    'T7':'T3', 'T8':'T4',
    'P7':'T5', 'P8':'T6'
}

"""
====================================================================================================================================
FUNCTIONS
====================================================================================================================================
"""

#this funct reads only the header of the patient first EEG segment and returns the list of channels
def get_first_segment_channels(patient_dir:str, patient_id:str) -> list:
    
    all_files = os.listdir(patient_dir)
    eeg_hea_files = sorted([f for f in all_files if f.endswith('.hea') and 'EEG' in f.upper()])

    if not eeg_hea_files:
        return None
    
    first_record = eeg_hea_files[0].replace('.hea', '')
    first_path = os.path.join(patient_dir, first_record)

    try:
        header = wfdb.rdheader(first_path)
        return header.sig_name
    except Exception as e:
        print(f" Error reading {patient_id}: {e}")
        return None

