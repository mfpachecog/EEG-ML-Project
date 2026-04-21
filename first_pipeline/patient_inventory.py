
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

def inventory_single_patient(data_dir:str, patient_id:str) -> dict:

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

    #read the **first segment** to obtain sampling rate and the channels information
    if eeg_hea_files:
        first_record_name = eeg_hea_files[0].replace('.hea', '')
        first_record_path = os.path.join(patient_dir, first_record_name)

        try:
            #We use the method .rdheader to read only the header file, and make the process quicker
            #with this we don't load the WHOLE signal and it is faster than rdrecord()
            header = wfdb.rdheader(first_record_path)
            patient_info['sampling rate'] = header.fs
            patient_info['n_channles'] = header.n_sig
            patient_info['channel_names'] = header.sig_name
            patient_info['segment_duration_sec'] = header.sig_len / header.fs
        except Exception as e:
            print(f" WARNING {patient_id} error reading the header: {e}")
            patient_info['sampling rate'] = None
            patient_info['n_channles'] = None
    
    return patient_info

"""
CREATE PATIENTS INVENTORY FUNCTION 

This function will search all the patient folders, and create the inventory
"""

def inventory_all_patients(data_dir):

    #start detecting the patients folders (numeric folders)
    all_items = sorted(os.listdir(data_dir))
    patient_ids = [d for d in all_items if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]

    print("=" * 80)
    print(f"PILOT SET INVENTORY (20 PATIENTS)")
    print(f"Directory: {data_dir}")
    print(f"Founded patients {len(patient_ids)}")
    print("=" * 80)

    #inventory each patient
    all_patients = []

    for i, pid in enumerate(patient_ids):
        print(f" [{i+1}/{len(patient_ids)}] Processing patient {pid}...", end="")
        info = inventory_single_patient(data_dir, pid)
        all_patients.append(info)

        #show quick resume about the collected data 
        outcome = info.get('outcome_label', '?')
        fs = info.get('samplint_rate', '?')
        n_seg = info.get('n_eeg_segments', 0)
        n_hours = info.get('n_unique_hours', 0)
        print(f"CPC={info.get('cpc_numeric', '?')}"
              f"({outcome})"
              f"| fs={fs} Hz"
              f"| {n_seg} segments"
              f"| {n_hours} hours")
    return all_patients, patient_ids

"""
FUNCTION TO CREATE PATIENTS RESUME, AND THE REPORT.
"""

def generate_report(all_patients):
    #Generate a table resume from inventory, and store it as CSV.

    #first create dataframe with most important columns
    summary_data = []

    for p in all_patients:
        row = {
            'patient_id': p.get('patient_id'),
            'hospital': p.get('hospital'),
            'age': p.get('age'),
            'sex': p.get('sex'),
            'cpc': p.get('cpc_numeric'),
            'outcome' :p.get('outcome_label'),
            'binary_label' :p.get('binary_outcome'),
            'shockable_rythm': p.get('shockable_rhythm'),
            'ohca': p.get('ohca'),
            'ttm': p.get('ttm'),
            'rosc' : p.get('rosc'),
            'sampling_rate_hz' : p.get('sampling_rate'),
            'n_channels' : p.get('n_channels'),
            'n_eeg_segments' : p.get('n_eeg_segments'),
            'n_unique_hours' : p.get('n_unique_hours'),
            'min_hour' : p.get('min_hour'),
            'max_hour' : p.get('max_hour'),
            'units' : p.get('units')
        }
        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    #REPORT CREATION

    print("\n" + "=" * 80)
    print("INVENTORY RESUME")
    print("=" * 80)

    #CLASSES BALANCE
    print("\n CLASSES BALANCE:")
    print("-" * 40)
    if 'outcome' in df.columns:
        outcome_counts = df['outcome'].value_counts()
        total = len(df)
        for outcome, count in outcome_counts.items():
            pct = (count / total) * 100
            bar = "█" * int(pct /2)
            print(f" {outcome:6s}: {count:3d} pacientes ({pct:.1f}%) {bar}")
    
    #CPC DISTRIBUTION
    print("\n CPC DISTRIBUTION: ")
    print("-" * 40)
    if 'cpc' in df.columns:
        cpc_counts = df['cpc'].value_counts().sort_index()
        for cpc, count in cpc_counts.items():
            label = {1: 'Good Recovery', 2:"Moderate Disability", 3: "Severe Disability", 4: "Vegetative", 5: "Death"}.get(cpc, 'unknown')
            group = 'GOOD' if cpc <= 2 else 'POOR'
            print(f" CPC {cpc}: {count:3d} patients - {label} [{group}]")
    
    #SAMPLING RATE
    print("\n SAMPLING RATES:")
    print("-" * 50)
    if 'sampling_rate' in df.columns:
        fs_counts = df['sampling_rate_hz'].value_counts().sort_index()
        for fs, count in fs_counts.items():
            print(f"    {fs} Hz: {count} patients")
            if len(fs_counts) > 1:
                print("\n WARNING THERE ARE MULTIPLE SAMPLING RATES")
                print(f"    IT WILL BE NECESSARY TO RESAMPLE TO A COMMON FREQUENCY")
    
    #HOSPITALS
    print("\n HOSPITALS")
    print("-" * 50)
    if 'hospital' in df.columns:
        hosp_counts = df['hospital'].value_counts().sort_index()
        for hosp, count in hosp_counts.items():
            #this will show what is the SAMPLING RATE each hospital uses.
            hosp_fs = df[df['hospital'] == hosp]['sampling_rate_hz'].unique()
            print(f" Hospital {hosp}: {count} patients (fs={hosp_fs})")

    #AVAILABLE HOURS
    print("\n TEMPORARY COVERAGE:")
    print("-" * 50)
    if 'max_hours' in df.columns:
        print(f"    MINIMUM HOUR REGISTERED: {df['min_hour'].min()}")
        print(f"    MAXIMUM HOUR REGISTERED: {df['max_hour'].max()}")
        print(f"    MEDIAN HOURS PER PATIENT: {df['n_unique_hours'].median():.0f}")

    