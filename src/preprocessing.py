
import os
os.chdir("C:\\Users\Josh\Coding\Atmoregime\src")

import xarray as xr
import numpy as np
import cftime
from deseasonaliser import Agg_Deseasonaliser, Sinefit_Deseasonaliser

class Preprocessor(object):

    def __init__(self,data,id="unnamed"):

        self.id=id

        #If not a dict make it a dict
        if type(data)!=dict:
            data={"input":data}

        self.data=data

        self._check_data_types()
        self._create_data_attributes()
        self.deseasonalisers={}

    def _check_data_types(self):
        datarray_type=type(xr.DataArray())
        try:
            self.data={m:x if type(x)==datarray_type else xr.DataArray.from_iris(x)\
             for m,x in self.data.items()}

        except:
            raise(IOError("All input data must be xarray DataArrays or iris Cubes"))

    def _create_data_attributes(self):

        for m in self.data:
            setattr(self,m,self.data[m])
        return

    def __repr__(self):
        return f"Preprocessor object: {self.id}"

    def __str__(self):
        out=[self.__repr__()]
        for m,x in self.data.items():
            out.append(f"\n{m}: {x.shape}")
        return "".join(out)

    def _update_data_attr(self, attr,new_val):
        self.data[attr]=new_val
        setattr(self,attr,new_val)
        return

    #Align data
    def align(self,data_keys=None,**kwargs):

        #if data is None we align all the data
        #otherwise just the named variables

        if data_keys is None:
            data_keys=list(self.data.keys())

        in_data=[self.data[d] for d in np.atleast_1d(data_keys)]

        aligned_data=xr.align(*in_data,**kwargs)
        for m,x in zip(data_keys,aligned_data):
            self._update_data_attr(m,x)

    def subset(self,coords,coord_bounds,data_keys=None):
            if data_keys is None:
                data_keys=list(self.data.keys())

            coords=np.atleast_1d(coords)
            coord_bounds=np.atleast_2d(coord_bounds)
            subsetted_data={}

            for key in data_keys:
                filter=None

                for coord,(b0,b1) in zip(coords,coord_bounds):
                    #Is true when the coord is between the inclusive limits
                    condition=(self.data[key].coords[coord]>=b0)&(self.data[key].coords[coord]<=b1)
                    #multiply each filter together
                    if filter is None:
                        filter=condition
                    else:
                        filter=filter*condition

                subsetted_data[key]=self.data[key].where(filter,drop=True)

            #We dont change any data until the whole function
            #runs without errors.
            for m,x in subsetted_data.items():
                self._update_data_attr(m,x)
            return
    def squeeze(self,data_keys=None):
        if data_keys is None:
            data_keys=list(self.data.keys())

        for m in data_keys:
            self._update_data_attr(m,self.data[m].squeeze())
        return


    def _detrend_arr(self,data,dim,deg):

        p = data.polyfit(dim=dim, deg=deg)
        fit = xr.polyval(data[dim], p.polyfit_coefficients)
        detrended=data-fit
        return detrended

    def detrend(self,data_keys=None,dim="time",deg=1):
        if data_keys is None:
            data_keys=list(self.data.keys())
        for m in data_keys:
            self._update_data_attr(m,self._detrend_arr(self.data[m],dim,deg))
        return

    def fit_seasonal_cycle(self,data_keys=None,deseasonaliser=Agg_Deseasonaliser,**ds_kwargs):
        if data_keys is None:
            data_keys=list(self.data.keys())
        for m in data_keys:
            dsnlsr=deseasonaliser()
            dsnlsr.fit_cycle(self.data[m])
            self.deseasonalisers[m]=dsnlsr

        return

    def evaluate_seasonal_cycle(self,data_key=None):

        try:
            dsnlsr=self.deseasonalisers[data_key]
        except KeyError as err:
            try:
                dsnlsr=self.deseasonalisers["input"]
            except:
                raise(err)

        return dsnlsr.evaluate_cycle()

    #By default removes the seasonal cycle of every dataset from itself, using
    #default kwargs of get_seasonal_cycle if the cycle has not been computed.
    #If compute_cycle=False will instead return error if cycle does not exist.
    #If cycle is a single value then that cycle will be removed from all data_keys.
    def deseasonalise(self,data_keys=None,cycle=None,compute_cycle=True):
        if data_keys is None:
            data_keys=list(self.data.keys())
        if cycle is None:
            cycle=data_keys
        elif type(cycle)==str:
            cycle=np.repeat(cycle,len(data_keys))

        for c,m in zip(cycle,data_keys):
            try:
                dsnlsr=self.deseasonalisers[c]
            except KeyError as err:
                if compute_cycle:
                    self.fit_seasonal_cycle(data_keys=[c])
                    dsnlsr=self.deseasonalisers[c]
                else:
                    raise(err)

            deseasonalised_data=self.data[m]
            cycle=dsnlsr.evaluate_cycle(data=self.data[m])
            deseasonalised_data.data=deseasonalised_data.data-cycle.data
            self._update_data_attr(m,deseasonalised_data)
        return

    #TO DO

    #Interpolating data to match
    #Take EOFs
    #Match time coords
