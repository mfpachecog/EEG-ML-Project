
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