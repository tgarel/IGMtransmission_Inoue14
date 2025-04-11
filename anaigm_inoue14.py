import numpy as np

def read_inoue14_IGMcoeff(inputpath):
    """
    Reads LAF and DLA coefficients from Inoue (2014) data files.
    """
    # File paths
    laf_file = f"{inputpath}/LAFcoeff.txt"
    dla_file = f"{inputpath}/DLAcoeff.txt"

    # Read LAF data
    laf_data = []
    try:
        with open(laf_file, 'r') as f:
            for line in f:
                values = list(map(float, line.split()))
                laf_data.append(values)
    except FileNotFoundError:
        raise FileNotFoundError("*** Cannot open the LAF file! ***")
    
    laf_data = np.array(laf_data)
    ### toto = laf_data[:, 0]
    lam1 = laf_data[:, 1]
    ALAF = laf_data[:, 2:5]  # Assuming ALAF has 3 columns
    
    # Read DLA data
    dla_data = []
    try:
        with open(dla_file, 'r') as f:
            for line in f:
                values = list(map(float, line.split()))
                dla_data.append(values)
    except FileNotFoundError:
        raise FileNotFoundError("*** Cannot open the DLA file! ***")
    
    dla_data = np.array(dla_data)
    ADLA = dla_data[:, 2:4]  # Assuming ADLA has 2 columns
    
    return lam1, ALAF, ADLA



# Function to compute Lyman series optical depth by the DLA component
def tLSDLA(zS, lobs, lam1, A):
    z1DLA = 2.0  # Redshift parameter for DLA
    tLSDLA = 0.0 # Initialize optical depth

    
    #for j in range(1, 39):  # WARNING : this omits Lya... k starts at 2 in .f90, make it start at 1 in python omits Lya...
    # k=2 in f90 corresponds to j=0 in py because Inoue stores lam1 in a 40-array whereas lam1 has only 39 elements : lam1(1) in f90 is 0....
    for j in range(len(lam1)):
        if ((lobs < lam1[j]*(1.+zS)) and (lobs > lam1[j])):   
            if lobs < lam1[j] * (1. + z1DLA):
                tLSDLA += A[j, 0] * (lobs / lam1[j])**2.0
            else:
                tLSDLA += A[j, 1] * (lobs / lam1[j])**3.0
    
    return tLSDLA

# Function to compute Lyman series optical depth by the LAF component
def tLSLAF(zS, lobs, lam1, A):
    z1LAF = 1.2  # First redshift parameter for LAF
    z2LAF = 4.7  # Second redshift parameter for LAF
    tLSLAF = 0.0  # Initialize optical depth
    
    #for j in range(1, 39):  # WARNING : this omits Lya... k starts at 2 in .f90, make it start at 1 in python omits Lya...
    # k=2 in f90 corresponds to j=0 in py because Inoue stores lam1 in a 40-array whereas lam1 has only 39 elements : lam1(1) in f90 is 0....
    for j in range(len(lam1)):
        ### print('QQQ : ',j,lam1[j])
        if ((lobs < lam1[j] * (1 + zS)) and (lobs >  lam1[j])):   
            if lobs < lam1[j] * (1 + z1LAF):
                tLSLAF += A[j, 0] * (lobs / lam1[j])**1.2
            elif lobs < lam1[j] * (1 + z2LAF):
                tLSLAF += A[j, 1] * (lobs / lam1[j])**3.7
            else:
                tLSLAF += A[j, 2] * (lobs / lam1[j])**5.5
    
    return tLSLAF

# Function to compute Lyman continuum optical depth by the DLA component
def tLCDLA(zS, lobs):
    z1DLA = 2.0   # Redshift parameter for DLA
    lamL = 911.8  # Lyman limit wavelength in Angstroms (restframe)
    
    if lobs > lamL * (1 + zS):
        return 0.0  # No absorption beyond Lyman limit
    elif zS < z1DLA:
        return (0.2113 * (1 + zS)**2 - 0.07661 * (1 + zS)**2.3 * (lobs / lamL)**(-0.3)
                - 0.1347 * (lobs / lamL)**2)
    elif lobs > lamL * (1 + z1DLA):
        return (0.04696 * (1 + zS)**3 - 0.01779 * (1 + zS)**3.3 * (lobs / lamL)**(-0.3)
                - 0.02916 * (lobs / lamL)**3)
    else:
        return (0.6340 + 0.04696 * (1 + zS)**3 - 0.01779 * (1 + zS)**3.3 * (lobs / lamL)**(-0.3)
                - 0.1347 * (lobs / lamL)**2 - 0.2905 * (lobs / lamL)**(-0.3))

# Function to compute Lyman continuum optical depth by the LAF component
def tLCLAF(zS, lobs):
    z1LAF = 1.2  # Redshift parameter1 for LAF
    z2LAF = 4.7  # Redshift parameter2 for LAF
    lamL = 911.8  # Lyman limit wavelength in Angstroms (restframe)
    
    if lobs > lamL * (1 + zS):
        return 0.0
    elif zS < z1LAF:
        return 0.3248 * ((lobs / lamL)**1.2 - (1 + zS)**(-0.9) * (lobs / lamL)**2.1)
    elif zS < z2LAF:
        if lobs > lamL * (1 + z1LAF):
            return 0.02545 * ((1 + zS)**1.6 * (lobs / lamL)**2.1 - (lobs / lamL)**3.7)
        else:
            return 0.02545 * (1 + zS)**1.6 * (lobs / lamL)**2.1 + 0.3248 * (lobs / lamL)**1.2 - 0.2496 * (lobs / lamL)**2.1
    else:
        if lobs > lamL * (1 + z2LAF):
            return 0.0005221 * ((1 + zS)**3.4 * (lobs / lamL)**2.1 - (lobs / lamL)**5.5)
        elif lobs > lamL * (1 + z1LAF):
            return 0.0005221 * (1 + zS)**3.4 * (lobs / lamL)**2.1 + 0.2182 * (lobs / lamL)**2.1 - 0.02545 * (lobs / lamL)**3.7
        else:
            return 0.0005221 * (1 + zS)**3.4 * (lobs / lamL)**2.1 + 0.3248 * (lobs / lamL)**1.2 - 0.0314 * (lobs / lamL)**2.1
        


# Function to compute optical depth and IGM transmission
def igm_transmission_inoue(z_source, lambda_obs, lam1, ALAF, ADLA):
    tau_igm        = (tLSLAF(z_source, lambda_obs, lam1, ALAF) +
                      tLSDLA(z_source, lambda_obs, lam1, ADLA) +
                      tLCLAF(z_source, lambda_obs) +
                      tLCDLA(z_source, lambda_obs))

    # Returns the IGM transmission at lambda_obs for source at z_source
        
    return np.exp(-tau_igm)
