import numpy as np
cimport numpy as np


cpdef np.ndarray COM_rdf(np.ndarray c1,np.ndarray c2,float Lx,float Ly, float Lz,int upper_limit,int bins=100):
    """
    function that calculates the distance between each pair of representation of a molecule (Center of mass, a single atom in the molecule etc.)
    c1: coordinate 1 (in shape (n1_molecules,3))
    c2: coordinate 2 (in shape (n2_molecules,3))
    volume: in units of A^3
    upper_limit: in units of A
    bins: number of bins
    """

    cdef int n1_molecules = len(c1)
    cdef int n2_molecules = len(c2)
    assert n1_molecules==n2_molecules
    cdef int ix = 0
    cdef float dx,dy,dz
    cdef np.ndarray pair_distances = np.zeros((n1_molecules*n2_molecules-n1_molecules,)) 
    cdef binned_vector = np.linspace(0,upper_limit,num=bins)
    cdef float distance
    cdef float dr = binned_vector[1] - binned_vector[0]
    cdef float rho = n1_molecules/(Lx*Ly*Lz) # density in units of molecule/A^{3}
    cdef np.ndarray volume_vec = 4*np.pi*(binned_vector[1:]**2)*dr #volume vector for each spherical shell
    cdef np.ndarray digitized
    cdef np.ndarray gr = np.zeros((bins,))

    for i in range(n1_molecules):
        for j in range(n2_molecules):
            if i != j:
                dx = abs(c1[i,0] - c2[j,0])
                dy = abs(c1[i,1] - c2[j,1])
                dz = abs(c1[i,2] - c2[j,2])

                # apply the minimum image convention
                dx = min(dx,Lx-dx)
                dy = min(dy,Ly-dy)
                dz = min(dz,Lz-dz)

                distance = np.sqrt(dx**2+dy**2+dz**2)
                if distance > upper_limit:
                    distance = 0

                pair_distances[ix] = distance
                ix += 1

    # sort the pair distances since np.digitze needs it to be sorted 
    pair_distances = np.sort(pair_distances)  
    pair_distances = pair_distances[pair_distances!=0]

    digitized = np.digitize(pair_distances,binned_vector)

    # binned_count has one less indices than binned_vector
    binned_count = [(digitized == i).sum() for i in range(1,bins)]
    binned_count = np.array(binned_count)

    gr[1:]  = binned_count/(rho*volume_vec*n1_molecules)

    return gr
    

