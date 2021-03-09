import numpy as np


def rdf(pos1,pos2,box,max_,nbins=100):
    """
    Function that calculates radial distribution function between a pair of representation of molecules (Center of Mass,Atom)

    Args:
    ----
    pos1(numpy.ndarray): Position matrix of the first matrix (N1,3)
    pos2(numpy.ndarray): Position matrix of the second matrix (N2,3)
    box(numpy.ndarray): The sides of the box (in A) (3,)
    max_(float): the maximum number at which the radial distribution should bin to
    nbins(int): The number of bins that the rdf should bin (default 100)
    
    Return:
    ------
        1.bins_(np.ndarray)=The array of bins
        2.gr(np.ndarray)=radial distribution function
    """
    N = pos1.shape[0]
    N2 = pos2.shape[0]
    assert N == N2
    gr = np.zeros((nbins,))

    
    # Find the density first
    rho = N/(box[0]*box[1]*box[2])
    bins_ = np.linspace(0,max_,nbins)
    deltar = bins_[1]-bins_[0]
    volume_vec = 4*np.pi*(bins_[1:]**2)*deltar
    
    # will result in shape (N1,N2,3)
    dr = abs(pos1[:,np.newaxis,:] - pos2)
    # Check if the absolute value of dx, dy, dz exceed half of the box length
    cond = dr > box/2
    # PBC
    dr = box*cond - dr
    
    # this will be of shape (N1,N2)
    distance = np.sqrt((dr**2).sum(axis=-1))    

    # Only take the upper triangular part as this matrix is symmetric
    distance = np.triu(distance)

    # eliminate the self distance terms
    np.fill_diagonal(distance,0)

    distance = distance.flatten()
    distance = distance[distance!=0]
    distance = distance[distance <= max_]
    
    # right = True such that bins[i-1] < x <= bins[i]
    digitized = np.digitize(distance,bins_,right=True)

    binned_vec  = np.array([(digitized == i).sum() for i in range(1,nbins)])
    gr[1:] = 2*binned_vec/(rho*volume_vec*N)
    
    return bins_,gr
