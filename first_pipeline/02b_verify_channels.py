
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
def get_first_segment_channels(data_dir:str, patient_id:str) -> list:
    
    patient_dir = os.path.join(data_dir, patient_id)
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

#function to normalize the channel names to the standard in the case they appear. 

def normalize_channel_name(ch_name: str) -> str:

    #capitalization normalization only on channels FP1 -> Fp1
    ch_norm = ch_name.strip()

    if ch_norm in MODERN_TO_OLD:
        return MODERN_TO_OLD[ch_norm]
    
    upper = ch_norm.upper()
    if upper.startswith('FP'):
        return 'Fp' + upper[2: ]
    
    return ch_norm

def verify_patient_channels(channels: list) -> dict:
    
    #this function verifies what standard channels the patient has, and which are missing

    normalized = [normalize_channel_name(ch) for ch in channels]
    normalized_set = set(normalized)

    #now we separate standard channels vs extra channels 
    standard_present = normalized_set & STANDARD_10_20
    missing_standard = STANDARD_10_20 - normalized_set
    extra_channels = normalized_set - STANDARD_10_20

    return {
        'total_channels': len(channels),
        'raw_channel_names':channels,
        'normalized_names':normalized,
        'standard_present':sorted(standard_present),
        'missing_standard':sorted(missing_standard),
        'extra_channels':sorted(extra_channels),
        'has_all_19_standard':len(missing_standard) == 0
    }

def  main():
    #detect patients

    all_items = sorted(os.listdir(DATA_DIR))
    patients_ids = [d for d in all_items if os.path.isdir(os.path.join(DATA_DIR, d)) and d.isdigit()]

    print("=" * 80)
    print("STANDARD 10 - 20 CHANNEL VERIFICATION")
    print(f"Patients to verify: {len(patients_ids)}")
    print("=" * 80)

    #verify each patient

    results = {}
    all_extra_channels = Counter()

    for pid in patients_ids:
        channels = get_first_segment_channels(DATA_DIR, pid )

        if channels is None:
            print(f" WARNING {pid}: could not be read")
            continue

        verification = verify_patient_channels(channels)
        results[pid] = verification

        #acumulate extra channels for the global analysis
        for ch in verification['extra_channels']:
            all_extra_channels[ch] += 1
        
        #show resume per patient
        status = "CHECK" if verification['has_all_19_standard'] else "WRONG"
        print(f" {status} {pid}:"
              f"{verification['total_channels']} total channels, "
              f"{len(verification['standard_present'])}/ 19 standard"
              + (f", ARE MISSING: {verification['missing_standard']}"
                 if verification['missing_standard'] else ""))

    """
    GLOBAL ANALYSIS OF THE INFORMATION
    """

    print("\n" + "=" * 90)
    print("GLOBAL ANALYSIS")
    print("=" * 90)

    #how many patients have the complete 19 channels
    complete = [pid for pid, v in results.items() if v['has_all_19_standard']]
    incomplete = [pid for pid, v in results.items() if not v['has_all_19_standard']]

    print(f"COVERAGE OF THE COMPLETE 10 - 20 SYSTEM")
    print("-" * 60)
    print(f" Patients with the 19 standard channels: {len(complete)}/{len(results)}")
    print(f" Patients with missing channels: {len(incomplete)}/{len(results)}")

    if incomplete:
        print(f"\n patients with missing channels:")
        for pid in incomplete:
            v = results[pid]
            print(f"    {pid}: missing {v['missing_standard']}")
    
    #count how many times each standard channel appears
    print(f" PRESENCE OF EACH STANDARD CHANNEL (in how many patients it appears):")
    print("-" * 60)
    channel_presence = Counter()
    for v in results.values():
        for ch in v['standard_present']:
            channel_presence[ch] +=1
    
    #sort by channel name
    for ch in sorted(STANDARD_10_20):
        count = channel_presence.get(ch, 0)
        pct = (count / len(results)) * 100
        bar = "|" * int(pct/5)
        status = "CHECK" if count == len(results) else "WRONG"
        print(f"    {status} {ch:5s}: {count:2d}/{len(results)} patients ({pct:.0f}%) {bar}")

    #extra channels NO STANDARD
    print(f"EXTRA CHANNELS (who doesn't belong to the 10 - 20 standard system):")
    print("-" * 60)
    if all_extra_channels:
        for ch, count in all_extra_channels.most_common():
            print(f"    {ch:10s}: Appear in {count}/{len(results)} patients")
    else:
        print("(None)")

    """
    DESIGN DECISION
    """

    print("\n" + "=" * 90)
    print("IMPLICATION TO THE PREPROCESSING PIPELINE")
    print("=" + 90)

    if len(complete) == len(results):
        print(f"""
            All the patients have the 19 complete standard channels. 
            PIPELINE: choosing directly the 19 channels -> without necessary imputation"""
              )
    elif len(complete) >= len(results) * 0.9:
        print(f"""
        the majority ({len(complete)}/{len(results)})  have the 19 channels,
        but some patients have missing channels

        PIPELINE: strategy reconsideration needed 
        - Option A: exclude the {len(incomplete)} incomplete patients (small pilot)
        - Option B: imput the missing channes by spacial interpolation
        """) 
    else: 
        print(f"""
        Many patients ({len(incomplete)}/{len(results)}) have missing channels.

        PIPELINE: Strategy reconsideration needed:
        - Use estrict intersection of common channels (less than 19)
        - implement spatial imputation
        """)

    # Keep results

    output_df = pd.DataFrame([
        {
            'patient_id':pid,
            'total_channels':v['total_channels'],
            'n_standard_present':len(v['standard_present']),
            'has_all_19':v['has_all_19_standard'],
            'missing_channels':','.join(v['missing_standard']) if v['missing_standard'] else 'none',
            'extra_channels':','.join(v['extra_channels']) if v['extra_channels'] else 'none',
            'raw_channel_names':','.join(v['raw_channel_names'])
        }
        for pid, v in results.items()
    ])

    output_df.to_csv('channel_verification.csv', index=False)
    print("\n results stored in: channel_verification.csv")

if __name__ == "__main__":
    main()
    
