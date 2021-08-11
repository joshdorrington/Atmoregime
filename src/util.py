import cftime
import numpy as np

def xarr_times_to_ints(time_coord):
    conversion=(1000*cftime.UNIT_CONVERSION_FACTORS["day"])
    return time_coord.to_numpy().astype(float)/conversion
