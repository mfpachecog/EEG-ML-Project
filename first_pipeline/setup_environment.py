"""
================================================================================================================================================================================================================
ENVIRONMENT SET UP

CHECK REQUIRED LIBRARIES FOR WORKING ON THE PROJECT
================================================================================================================================================================================================================

"""

def verify_installation() -> bool:

    #function created to match the required libraries with the installed ones. 

    libraries = {'wfdb':'WFDB archives lecture (I-CARE format)', 
                 'numpy':'numeric computation',
                 'matplotlib':'data and graphics visualization',
                 'scipy':'signals processing and statistics',
                 'pandas':'tabular data manipulation',
                 'sklearn':'Machine Learning (scikit-learn)',
                 'seaborn':'statistic visualization'}
    
    print("=" * 60)
    print("Environment Verification ")
    print("Project: EEG post cardiac arrest prognosis")
    print("=" * 60)

    all_ok = True




