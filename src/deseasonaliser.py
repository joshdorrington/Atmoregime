import numpy as np
import xarray as xr
import cftime
import pandas as pd
from scipy.optimize import leastsq
import os
import matplotlib.pyplot as plt
import time
import cftime
from util import xarr_times_to_ints

class Agg_Deseasonaliser(object):

    def __init__(self):
        self.data=None
        self.cycle_coeffs=None


    def fit_cycle(self,arr,dim="time",agg="dayofyear"):
        var=dim+"."+agg
        self.data=arr
        self.dim=dim
        self.agg=agg
        self.cycle_coeffs=arr.groupby(var).mean(dim=dim)
        return

    def evaluate_cycle(self,data=None):
        if data is None:
            data=self.data
        return self.cycle_coeffs[getattr(data[self.dim].dt,self.agg).data-1]

class Sinefit_Deseasonaliser(Agg_Deseasonaliser):


    def _func_residual(self,p,x,t,N,period):
        return x - self._evaluate_fit(t,p,N,period)

    def _evaluate_fit(self,t,p,N,period):
        ans=p[1]
        for i in range(0,N):
            ans+=p[2*i+2] * np.sin(2 * np.pi * (i+1)/period * t + p[2*i+3])
        return ans

    def _lstsq_sine_fit(self,arr,t,N,period):

        #Guess initial parameters
        std=arr.std()
        p=np.zeros(2*(N+1))
        for i in range(0,N):
            p[2+2*i]=std/(i+1.0)
        plsq=leastsq(self._func_residual,p,args=(arr,t,N,period))
        return plsq[0]

    def fit_cycle(self,arr,dim="time",N=4,period=365.25):

            dims=arr.dims
            self.dim=dim
            self.data=arr
            self.N=N
            self.period=period
            t=xarr_times_to_ints(arr[dim])
            self.coeffs= xr.apply_ufunc(
                self._lstsq_sine_fit,
                arr,
                input_core_dims=[[dim]],
                output_core_dims=[["coeffs"]],
                vectorize=True,
                kwargs={"t":t,"N":N,"period":period})
            return

    def evaluate_cycle(self,data=None):
        if data is None:
            data=self.data
        dims=data.dims
        t=xarr_times_to_ints(data[self.dim])

        cycle=np.array([[self._evaluate_fit(t,c2.data,self.N,self.period)\
                for c2 in c1] for c1 in self.coeffs])

        return data.transpose(...,"time").copy(data=cycle).transpose(*dims)
