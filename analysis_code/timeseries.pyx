import numpy as np
cimport numpy as np


class Timeseries:
    """
    Timeseries class

    Args:
    ----
        data: the Time series data that user inputs (shape: (N,...)) where N is the number of time steps
        time: total time length (numpy array)
    """ 
    def __init__(self,data,time):
        self.n = len(data)
        self.time = time
        self.data = data

    def blockstats(self,size_block,num_draws):
        block = np.zeros((num_draws,))
        
        for i in range(num_draws):
            b = np.random.choice(self.data,size_block)
            block[i] = b.mean()

        return (block.mean(),block.std()) 

    def normalize(self):
        """
        Function that normalizes the data set

        Return:
        ------
            Normalized timeseries data 
        """
        time = self.time 
        data = self.data
        data = (data - self.mean())/self.std()

        return Timeseries(data,time)

    def mean(self):
        """
        Function that calculates the mean of the dataset

        Return:
        ------
            Mean of the dataset 
        """

        return self.data.mean()
    
    def std(self):
        """
        Function that calculates the standard deviation

        Return:
        ------
            Std of the dataset
        """

        return self.data.std()

    def moving_average(self,window):
        """
        ignore the first (window) points and calculate the moving average
        """
        filter_ = np.ones((window,))*(1/window) 
        smoothed_points = np.convolve(self.data,filter_,'valid')
        
        return Timeseries(smoothed_points,self.time[window-1:])

    def autocorrelation(self,delta_t=1):
        """
        Using fourier transform to perform autocorrelation, this is a much much much faster way to compute
        autocorrelation function

        theory:
            Refer to Milner notes pg95-96
            Consider a autocorrelation function:
                    Cj = 1/n*\sum_{i=0}^{n-1}A_{i}A_{i+j}
            If we perform Fourier Transform on this, the right hand side can be thought of as an autocorrelation, so
            using the convolution theorem where FT[f convolve g]=FT[f]*FT[g]

            Since f and g are the same function, we get Ctildek = Ak^{2} where Ak is the Fourier transform of the data set 
            Ai and Ctildek is the fourier transform of the autocorrelation function

            then Cj is calculated by IFFT[Ctildek]

        returns:
            a tuple of (lag_time, The autocorrelation of shape (N,))
        """
        # First normalize the data (Timeseries object)
        normalized = self.normalize().data
        N = len(normalized)
        lags = np.arange(0,N*delta_t,delta_t)
        
        # zero pad the middle of the data
        normalized = np.r_[normalized[N//2:],np.zeros_like(normalized),normalized[:N//2]]

        # First perform fourier transform on the data set
        ft = np.fft.fft(normalized)

        # Square the fourier transform output
        sq_ft = (ft.real)**2+(ft.imag)**2

        # Now perform invert fourier transform on the sq_ft
        AC = np.fft.ifft(sq_ft)

        # take only the real part of AC and divide it by N
        AC = AC.real/N

        return (lags,AC[:N])

    def AC_tau(self,cutoff=False):
        """
        Function that calculates autocorrelation time of a time series according to the definition provided by

        Chodera, J. D., Swope, W. C., Pitera, J. W., Seok, C. & Dill, K. A. Use of the weighted histogram analysis method for the analysis of simulated and parallel tempering simulations. J. Chem. Theory Comput. 3, 26–41 (2007).
        where:
            \tau = sum_{t=1}^{N-1}(1-t/N)Ct

        cutoff: if True, then the sum will be cut off when AC crosses 0

        returns:
            autocorrelation time (float)
        """
        _,ac = self.autocorrelation()
        coeff_ = 1-np.arange(1,self.n+1)/self.n
        
        if cutoff is False:
            sum_ = (coeff_*ac).sum()
        else:
            sum_ = 0
            for i in range(len(ac)):
                if ac[i] >= 0:
                    sum_ += coeff_[i]*ac[i]
                else:
                    break

        return sum_

    def __len__(self):
        return len(self.data)

    def __mul__(self,other):
        if isinstance(other,Timeseries):
            return self.data * other.data
        else:
            return self.data * other
        
    def __getitem__(self,ix):
        """
        handles indexing of the timeseries object
        ix: could be slice or int

        slice:
        ix could be inputted in the form of [start:end:step]
        this returns a timeseries object where the data and time are sliced to be
        data = data[start:end:step]
        time = time[start:end:step]

        int:
        returns an integer in the array data[ix]
        """
        if isinstance(ix,slice):
            st = ix.start
            if st is None:
                st = 0
            else:
                st = np.searchsorted(self.time,st,"left")

            s = ix.step
            if s is None:
                s = 1

            e = ix.stop
            if e is None:
                e = -1
            else:
                e = np.searchsorted(self.time,e,"left")

            data = self.data[st:e:s]
            time = self.time[st:e:s]
            return Timeseries(data,time)
        else: 
            return self.data[ix]
