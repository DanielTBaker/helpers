import numpy as np

def rebin(arr,rbf=1,rbt=1):
    """
    Rebins a 2D array along both axes. If you have coordinate arrays for the axes you should rebin those by the same amount

    Parameters
    arr - A 2D numpy array to be rebinned
    rbf - The factor to rebin along the frequency (1st) axis
    rbt - The factor to rebin along the time (2nd) axis
    """

    ##The array might not be an integer multiple of the rebininng factors
    ##We need to find the size of the largest acceptable subarray
    # Number of time bins anfter rebinning
    nf=arr.shape[0]//rbf
    # Number of time bins anfter rebinning
    nt=arr.shape[1]//rbt

    #Rebin
    out = np.nanmean(np.reshape(arr[:rbf*nf,:rbt*nt],(nf,rbf,nt,rbt)),(1,3))
    return(out)

def badFix(arr,bad=None):
    """
    Removes all bad points by setting them to nan.

    Parameters
    arr - The array to be fixed
    bad - (optional) A predetermined set of bad points.
    """
    ## Copy array for output
    fixed=np.copy(bad)

    ## If bad is given we use that, otherwise we generate a new bad
    if bad:
        ## Set all bad points to nan
        fixed[bad]=np.nan
    else:
        # Find all nans and infs
        bad=np.invert(np.isfinite(arr))
    ##Set bad points to the mean of remaining points
    fixed[bad]=np.nanmean(fixed)
    return(fixed)

def freqScale(dspec, freq, fd, fref):
    CS = np.fft.fftshift(np.fft.fft(dspec, axis=1), axes=1)
    for i in range(freq.shape[0]):
        interp = interp1d(
            (fd * (fref / freq[i])).value, CS[i, :], bounds_error=False, fill_value=0
        )
        CS[i, :] = interp(fd.value)
    CS = np.fft.fftshift(np.fft.fft(CS, axis=0), axes=0)
    return CS
