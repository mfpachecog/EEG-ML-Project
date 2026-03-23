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
    for lib, description in libraries.items():
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'N/A')
            print(f":) {lib:15s} v{version:10s} -- {description}")
        except ImportError:
            print(f" :( {lib:15s} NOT INSTALLED - {description}")
            all_ok = False

    print("=" * 60)
    print("Requirements check")
    print("="*60)


    if all_ok:
        print("Everythin is installed properly, And ready to begin")
        print("\nNext step start with the first patient preview")
    else:
        print("There are some libraries not installed")

    print("=" * 60)
    return all_ok

if __name__== "__main__":
    verify_installation()




