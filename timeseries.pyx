import numpy as np
cimport numpy as np


class Timeseries:
    def __init__(self,data,time):
        """
        data: the Time series data that user inputs (shape: (N,...)) where N is the number of time steps
        time: total time length
        """
        self.time = time
        self.data = data
        self.n = len(data)

    def blockstats(self,size_block,num_draws):
        cdef np.ndarray block = np.zeros((num_draws,))
        
        for i in range(num_draws):
            b = np.random.choice(self.data,size_block)
            block[i] = b.mean()

        return (block.mean(),block.std()) 

    def mean(self):
        return self.data.mean()
    
    def std(self):
        return self.data.std()

    def moving_average(self,window):
        """
        ignore the first (window) points and calculate the moving average
        """
        cdef np.ndarray filter = np.ones((window,))*(1/window) 
        cdef np.ndarray smoothed_points 
        smoothed_points = np.convolve(self.data,filter,'valid')
        
        return Timeseries(smoothed_points,self.time[window-1:])

    def autocorrelation(self,lags):
        """
        function that calculates the autocorrelation of a timeseries
        lags: the range of lags to perform (default to be len(self.data)-1) 
        """
        cdef float mean = self.mean()
        cdef float denom = ((self.data-mean)**2).sum()
        cdef np.ndarray ac = np.zeros((lags,)) 
        cdef float numer
        cdef np.ndarray lag = np.arange(lags)

        for l in range(lags):
            numer = 0.0 
            for i in range(len(self.data)-l):
                numer += (self.data[i]-mean)*(self.data[i+l]-mean)

            ac[l] = numer/denom 
        return [ac,lag]

    def AC_tau(self):
        """
        Function that calculates autocorrelation time of a time series according to the definition provided by 
        (Box and Jenkins 1976), where
                    \tau = 1/2 + \sum_{k=1}^{N}A(k)

        returns:
                autocorrelation time (float)
        """
        ac,_ = self.autocorrelation(self.n-1)
        sum_ = 0
        for num in range(len(ac)):
            sum_ += (1-num/self.n)*ac[num]


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



