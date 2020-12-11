import numpy as np
from scipy.special import erf

def h(alpha_i,alpha_c,sigma,amin,amax):
    """
    This is the function h(alpha) used in the paper by Amish on INDUS
    This function that the form 

    h(alpha_i) = \int_{amin}^{amax} \Phi(alpha-alpha_i) dr
    where 

    \phi(alpha_i) = k^-1*[e^{-alpha^{2}/(2sigma^{2})} - e^{-alphac^{2}/(2sigma^{2})}]
    where k is the normalizing constant
    
    inputs:
        alpha_i: the input alpha's, it can be within range range (can be a numpy array or float or int)
        alpha_c: the alpha_c in the equation
        sigma: the sigma in the equation
        amin: the lower bound of integral
        amax: the upper bound of integral

    output:
        a float/int or numpy array depending on the input alpha_i
        if alpha_i is float/int, then output will be int that corresponds to h(alpha_i)
        else if alpha_i is numpy array, then output will be numpy array that corresponds to h(alpha_i)
    """
    # normalizing constant
    k = -2*np.exp(-alpha_c**2/(2*sigma**2))*alpha_c+np.sqrt(2*np.pi*sigma**2)*erf(alpha_c/np.sqrt(2*sigma**2))

    # the low and high of the function, beyond these will be zero
    low_ = amin - alpha_c
    high = amax + alpha_c
    h = np.heaviside(alpha_i - (amin - alpha_c),1) - np.heaviside(alpha_i - (amax + alpha_c),1)
    
    # set appropriate boundary depending on the alpha_i
    a = np.where(np.abs(alpha_i - amin) < alpha_c,amin,alpha_i-alpha_c)
    b = np.where(np.abs(alpha_i - amax) < alpha_c,amax,alpha_i+alpha_c)

    # return the integrated value/values
    return h/k*((a-b)*np.exp(-alpha_c**2/(2*sigma**2))+\
            np.sqrt(np.pi/2)*sigma*(erf((alpha_i-a)/np.sqrt(2*sigma**2))\
                                    -erf((alpha_i-b)/np.sqrt(2*sigma**2)))) 

def equilibrium_k0(LC,xmin,xmax,ymin,ymax,zmin,zmax,start_t,end_t,alpha_c=0.02,sigma=0.01,skip=1,verbose=False):
    """
    A function that approximates k0 using rule of thumb 
    This function is used specifically for a cuboidal probe volume 
    The rule of thumb provided by Nick Rego is as follows:
    (a) Assume underlying free energy U0 is gaussian with <N>0 and var<N>0. 
        U0 = k0/2(N-N0)^2 so k0~1/var(N)0
    inputs:
        LC: Liquid crystal object 
        mumin: minimum mu of the probe volume (all 6)
        start_t: the starting time of the calculation
        end_t: the ending time of the calculation
        skip: the number of frames to skip between start_t, end_t
        alpha_c: alpha_c for cut-off of INDUS function h 
        sigma: the width for the INDUS function h 

    """
    time = LC.time
    time_idx = np.linspace(0,time,len(LC)) # the time index during the simulation
    start_idx = np.searchsorted(time_idx,start_t,side='left')
    end_idx = np.searchsorted(time_idx,end_t,side='right')
    calc_time = np.arange(start_idx,end_idx,skip) 

    N_tilde_tot = np.zeros((len(calc_time),))
    N_tot = np.zeros((len(calc_time),))
    ix = 0
    if LC.bulk == True:
        u = LC['universe']
        atoms = u.select_atoms("resname {}CB".format(LC.n))
    else:
        u = LC['universe']
        atoms = u.select_atoms("all")
    

    for idx in calc_time:
        u.trajectory[idx]
        pos = atoms.positions  

        # satisfy x positions
        pos = pos[pos[:,0]>=xmin]
        pos = pos[pos[:,0]<=xmax]
        # satisfy y positions
        pos = pos[pos[:,1]>=ymin]
        pos = pos[pos[:,1]<=ymax]
        # satisfy z positions
        pos = pos[pos[:,2]>=zmin]
        pos = pos[pos[:,2]<=zmax]

        h_x = h(pos[:,0],alpha_c,sigma,xmin,xmax)
        h_y = h(pos[:,1],alpha_c,sigma,ymin,ymax)
        h_z = h(pos[:,2],alpha_c,sigma,zmin,zmax)

        h_ = h_x*h_y*h_z
        N_tilde = h_.sum()
        N_tilde_tot[ix] = N_tilde
        N_tot[ix] = len(pos)
        ix += 1

        if verbose:
            print("time at {} is done,Ntilde is {}".format(idx,N_tilde))

    return (N_tilde_tot,N_tot,calc_time)
 
