import numpy as np
cimport numpy as np
import MDAnalysis as mda
from analysis_code.md import *
from analysis_code.timeseries import *

class nCB(simulation):
    """
    Args:
    ----
        path(string): path to the MD_folder, in liquid crystals, it has to have the following files
        1. {md_name}.tpr
        2. {md_name}_pbc.xtc
        3. {md_name}_properties.xtc
        The folder that contains these files should be name {md_name}

        time(int): the total simulation time (ns)

        n(int): the n in nCB

        bulk(bool): a boolean that tells whether or not the simulation is bulk nCB or mixture

        prop(list): a list that tells the LC object to load certain files, they need to have the name of 
              {md_name}_property.xvg

        p2(bool): a boolean that tells whether or not p2 vector needs to be computed and saved in the folder

        verbose(bool): whether or not the program shall print results
    """
    def __init__(self,path,time,n,bulk=True,prop=None,p2=False,verbose=False):
        self.n = n
        self.bulk = bulk
        self.prop = prop

        if self.prop != None:
            super().__init__(path,time,prop)
            # Convert nm to A
            if "volume" in prop:
                self.properties["volume"].data = self.properties["volume"]*(10**3) #convert to A^3
            if "Lx" in prop:
                self.properties['Lx'].data = self.properties['Lx']*10 # convert to A
            if "Ly" in prop:
                self.properties['Ly'].data = self.properties['Ly']*10 
            if "Lz" in prop:
                self.properties['Lz'].data = self.properties['Lz']*10 
        else:
            super().__init__(path,time)
             
        if self.bulk == False:
            LC_atoms = self.properties["universe"].select_atoms("resname {}CB".format(n))
            self.n_atoms = len(LC_atoms.residues[0].atoms)
            self.n_molecules = len(LC_atoms.residues)            
        else:
            self.n_atoms = len(self.properties["universe"].residues[0].atoms)
            self.n_molecules = len(self.properties["universe"].residues)
        
        if self.n == 5:
            self.segments = {"CN":np.arange(0,2),\
                    "benzene1":np.arange(2,8),\
                    "benzene2":np.arange(8,14),\
                    "HC_tail":np.arange(15,19)}

        if self.n == 8:
            self.segments = {"CN":np.arange(0,2),\
                    "benzene1":np.arange(2,8),\
                    "benzene2":np.arange(8,14),\
                    "HC_tail":np.arange(15,22)}

        # find p2 order parameter if specified by user
        if p2 == True:
            if "{}_{}.npy".format(self.mdname,"p2") not in self.files:
                if verbose:
                    print("{}_{}.npy not found, calculating p2 time series for the Liquid crystal simulation".format(self.mdname,"p2"))
                data = self.p2()
                np.save(path+"/{}_{}.npy".format(self.mdname,"p2"),data)
                t = np.linspace(0,time,len(data))
                self.properties["p2"] = Timeseries(data,t)
            else:
                if verbose:
                    print("found {}_{}.npy in path, extracting data".format(self.mdname,"p2"))
                data = np.load(path+"/{}_{}.npy".format(self.mdname,"p2"))
                t = np.linspace(0,time,len(data))
                self.properties["p2"] = Timeseries(data,t)

    def CN_vec(self,ts):
        """
        Function that calculates the CN vector at time ts

        Args:
        ----
        ts: the time step of the trajectory

        Return:
        ------
        numpy array of CN vectors (N,3)
        """
        u = self.properties["universe"]
        u.trajectory[ts]

        if self.bulk:
            residues = u.residues
        else:
            residues = u.select_atoms("resname {}CB".format(self.n)).residues
        
        CN = np.zeros((len(residues),3))
        ix = 0
        for res in residues: 
            N = res.atoms[0].position
            C = res.atoms[1].position
            CN_vec = (N - C)/np.sqrt(((N-C)**2).sum())
            CN[ix] = CN_vec
            ix += 1

        return CN

    def director_MOI(self,ts):
        """
        Function that finds the director of each of the molecule using moment of inertia tensor

        Args:
        ----
            ts(int): The time frame

        Return:
        ------
            director_mat(numpy.ndarray): A matrix of director of shape (N,3)
        """
        u = self.properties["universe"]
        u.trajectory[ts]
        N = len(u.residues)
        director_mat = np.zeros((N,3))

        for i in range(N):
            director_mat[i] = u.select_atoms("resnum {}".format(i)).principal_axes()[-1]

        return director_mat

    def Qmatrix(self,ts,MOI=False):
        """
        calculates the Q matrix at a particular time step of the trajectory

        Args:
        ----
            ts: the time step at which the Qmatrix is calculated

        Return:
            Q:
            \sum_{l=1}^{N} (3*u_{l}u_{l}t - I)/2N
            (3,3)
        """ 
        if MOI:
            CN_vec = self.director_MOI(ts)
        else:
            CN_vec = self.CN_vec(ts)

        ix = 0
        n = self.n_molecules
        I = np.eye(3)

        Q = 3/(2*n)*np.dot(CN_vec.T,CN_vec) - 1/2*I

        return Q,CN_vec
    
    def director(self,Q):
        """
        Calculates the system director at time ts
        the system director corresponds to the eigenvector which corresponds to the largest
        eigenvalue of the Q matrix

        Args:
        ----
            Q: the Q matrix of LC system (3,3)

        Return:
        ------
            director in shape(3,1)
        """
        eigv,eigvec = np.linalg.eig(Q)
            
        order = np.argsort(-abs(eigv))
        eigvec = eigvec[:,order]
        director = eigvec[:,0:1]
        order2 = np.argsort(eigv)

        return director,eigv[order2]

    def COM(self,ts,segment='whole'):
        """
        Args:
        ---
            ts: the time step at which this center of mass measurement is at 
            segment: the segment at which the center of mass is measured on

        Return:
        ------
            center of mass matrix of shape (N_molecules,3)
        """
        self.properties["universe"].trajectory[ts]

        if self.bulk == True:
            residues = self.properties['universe'].residues
        else:
            residues = self.properties['universe'].select_atoms("resname {}CB".format(self.n)).residues

        if segment in self.segments.keys():
            idx = self.segments[segment]
        elif segment == 'whole': 
            idx = np.arange(0,self.n_atoms)
        else:
            raise NotImplementedError("The segment {} is not currently implemented".format(segment))

        COM = np.zeros((self.n_molecules,3)) 
        ix = 0

        for res in residues:
            num = res.atoms[idx].center_of_mass()
            COM[ix] = num
            ix += 1

        return COM 
             
    def p2(self):
        """
        Function calculating p2 order parameter for nCB molecule
        Q matrix is calculated as following:
        Q = \sum_{l}^{N}(3u_{l}u_{l}^{T} - I)/2N
        we choose p2 to be -2*lambda_{0} where lambda_{0} is the second largest eigenvalue of Q
        u_{l} is chosen to be the normalized vector between C and N in nCB

        Return:
        ------
            P2 vector as a function of time
        """
        ix = 0
        t = len(self.properties["universe"].trajectory)
        p2_vec = np.zeros((t,))
        time = np.linspace(0,self.time,t)

        for ts in range(t):
            Q = self.Qmatrix(ts)
            _,eigv = self.director(Q)

            p2_vec[ix] = eigv[1]*(-2)
            ix += 1

        return p2_vec

    
    def pcost_z(self,start_time,end_time,director,min_,max_,direction='z',segment='whole',skip=1,bins_z=100,bins_t=100,verbose=False):
        """
        Function that calculates a heat map of p(cos(theta) as a function of z. 

        Args:
        ----
            start_time: the starting time in ns

            end_time: the ending time in ns

            direction: at which direction is the calculation being performed along ('x','y','z')

            segment: which segment of the LC molecule to calculate COM for 

            skip: number of time frames to skip 

            bins_z: the bins along the direction where the calculation is being performed

            bins_t: the bins of theta for p(cos(theta))

            verbose: whether to be verbose during execution

        Return:
            2d array contains p(cos(theta)) as a function of z  (bins_z-1,bins_t-1)
        """
        u = self.properties['universe']
        t_binned = np.linspace(-1,1,bins_t)
        z_binned = np.linspace(min_,max_,bins_z)


        # find the time frame indexes of the simulation
        t_idx = np.linspace(0,self.time,len(u.trajectory))
        start_idx = np.searchsorted(t_idx,start_time,side='left')
        end_idx = np.searchsorted(t_idx,end_time,side='right')
        time_idx = np.arange(start_idx,end_idx,skip)


        pcost_theta_director = np.zeros((bins_z-1,bins_t-1)) # a 2-d array that holds p(cos(theta)) in shape (bins_z-1,bins_t-1)

        if direction == 'x':
            d = 0
        
        if direction == 'y':
            d = 1

        if direction == 'z':
            d = 2
        
        for ts in time_idx:
            # first find Center of mass matrix at time step "ts"
            COM_mat = self.COM(ts,segment) #(n_molecues,3)

            # take only the dth dimension number of all COM 
            COM_mat = COM_mat[:,d] #(n_molecules,)

            # find the CN vectors of all the molecules 
            CN_vec = self.CN_vec(ts)
            cost = (CN_vec*director).sum(axis=1)
            
            prob,_,_ = np.histogram2d(COM_mat,cost,[z_binned,t_binned])
            pcost_theta_director += prob
             
            if verbose:
                print("time step {} is done".format(ts))
                   
        return pcost_theta_director/len(time_idx)

cpdef director_z(LC,ts,segment='whole',bins_z=100,direction='z',Broken_interface=None,verbose=False):
    """
    finds director as a function of z along the direction provided
   
    Args:
    -----
        LC: Liquid crystal object
        ts: the time step at which the calculation is performed 
        segment: The segment at where we want to take COM at 
        bins_z: the number of bins that we want to break the analysis into
        direction:'x','y' or 'z'
        
        Broken_interface: whether or not the interface is broken (if not 
                          None, should have the following inputs
                          Broken_interface = (Lz,draw_line)) where Lz
                          and draw line are int. Lz is the entire height of the box
                          and draw line is where the line would be drawn

    returns:
        p2z matrix in shape (T, n_molecules*n_atoms)
    """
    cdef np.ndarray directorz = np.zeros((bins_z-1,3))
    cdef np.ndarray COM_mat
   
    if direction == 'x':
        d = 0

    if direction == 'y':
        d = 1

    if direction == 'z':
        d = 2
    
    # first find Center of mass matrix at time step "ts"
    COM_mat = LC.COM(ts,segment) #(n_molecues,3)

    # take only the dth dimension number of all COM 
    COM_mat = COM_mat[:,d] #(n_molecules,)

    # set the universe trajectory to "ts" time step
    LC['universe'].trajectory[ts]

    if LC.bulk == True:
        residues = LC['universe'].residues
    else:
        residues = LC['universe'].select_atoms("resname {}CB".format(LC.n)).residues

    COM_vec,Ntop = fix_interface(COM_mat,Broken_interface=Broken_interface,bins=bins_z,verbose=verbose)

    for i in range(bins_z-1):
        if Ntop != 0:
            if i >= Ntop:
                j = i + 1
            else:
                j = i
        else:
            j = i

        less = COM_vec[j]
        more = COM_vec[j+1]

        index = np.argwhere(((COM_mat >= less) & (COM_mat < more)))
        index = index.flatten()
        if index.size != 0:
            Q = np.zeros((3,3))
            I = np.eye(3)
            number = len(index) # number of molecules with COM in range (less, more)
            for idx in index:
                res = residues[idx]
                N = res.atoms[0].position
                C = res.atoms[1].position
                CN_vec = (N - C)/np.sqrt(((N-C)**2).sum())

                Q += (3*np.outer(CN_vec,CN_vec)-I)/(2*number)
            eigval,eigvec = np.linalg.eig(Q)
            order = np.argsort(-np.abs(eigval))
            eigvec = eigvec[:,order]
            directorz[i] = eigvec[:,0]

    return directorz


# find p2 as a function of z where z represents one direction in the Cartesian coordinates
cpdef p2_z(LC,start_t,end_t,director=None,segment='whole',skip=None,bins_z=100,direction='z',Qmatrix=True,Broken_interface=None,verbose=False):
    """
    finds p2 as a function of z along the direction provided
    
    start_t: the time at which the evaluation starts (in ns)
    end_t: the time at which the evaluation ends (in ns)
    direction:'x','y' or 'z'
    segment: The segment at where we want to take COM at 

    returns:
        p2z matrix in shape (T, n_molecules*n_atoms)
    """
    cdef int t = len(LC) # total amount of time frames in LC simulation 
    cdef np.ndarray time = np.linspace(0,LC.time,t) # find the list of simulation times in ns 
    cdef int start_timeidx = np.searchsorted(time,start_t,side='left') # find the index of the starting time in list of simulation times (in frame)
    cdef int end_timeidx = np.searchsorted(time,end_t,side='right') # find the index of the ending time in list of simulation times (in frame)
    cdef np.ndarray time_idx = np.arange(start_timeidx,end_timeidx,skip) 
    cdef int n = len(time_idx)
    cdef np.ndarray p2z = np.zeros((bins_z-1,))
    cdef np.ndarray COM_mat
    cdef int ix = 0
   
    if direction == 'x':
        d = 0

    if direction == 'y':
        d = 1

    if direction == 'z':
        d = 2
    

    for ts in time_idx:
        if verbose:
            print("performing calculations for t={}".format(ts))
        # first find Center of mass matrix at time step "ts"
        COM_mat = LC.COM(ts,segment) #(n_molecues,3)

        # take only the dth dimension number of all COM 
        COM_mat = COM_mat[:,d] #(n_molecules,)

        # set the universe trajectory to "ts" time step
        LC['universe'].trajectory[ts]
        if LC.bulk == True:
            residues = LC['universe'].residues
        else:
            residues = LC['universe'].select_atoms("resname {}CB".format(LC.n)).residues

        COM_vec,Ntop = fix_interface(COM_mat,Broken_interface=Broken_interface,bins=bins_z,verbose=verbose) 

        for i in range(bins_z-1):
            if Ntop != 0:
                if i >= Ntop:
                    j = i + 1
                else:
                    j = i
            else:
                j=i

            less = COM_vec[j]
            more = COM_vec[j+1]

            index = np.argwhere(((COM_mat >= less) & (COM_mat < more)))
            index = index.flatten()
            if index.size != 0:
                if Qmatrix:
                    Q = np.zeros((3,3))
                    I = np.eye(3)
                else:
                    p2 = 0
                number = len(index) # number of molecules with COM in range (less, more)
                for idx in index:
                    res = residues[idx]
                    N = res.atoms[0].position
                    C = res.atoms[1].position
                    CN_vec = (N - C)/np.sqrt(((N-C)**2).sum())

                    if Qmatrix:
                        Q += (3*np.outer(CN_vec,CN_vec)-I)/(2*number)
                    else:
                        cos_t = np.dot(CN_vec,director)
                        p2 += ((cos_t**2)*3-1)/(2*number)
                if Qmatrix:
                    eigval,eigvec = np.linalg.eig(Q)
                    order = np.argsort(eigval)
                    eigval = eigval[order]
                    p2 = -2*eigval[1]

                    diff_order = np.argsort(np.abs(eigval))
                    eigvec = eigvec[:,diff_order]
                    p2z[i] += p2
                else:
                    p2z[i] += p2
    p2z /= n
    return (p2z,time_idx)



# find cos(theta) between CN and the global director of LC systems
cpdef CN_director_theta(LC,start,end,skip=None,verbose=False,director=None):
    """
    Function that calculates the cos(theta) between the CN vector of eaceh LC molecule
    and the director of the system.
    The director at each time frame is chosen to be the first eigenvector of the Q tensor
    Q = \sum_{i}^{N} (3*u_{i} u_{i}t - I)/2N


    start: The time to start calculation (in ns)
    end: The time to end calculation (in ns)
    skip: the time step in which to skip (in time index)
    verbose: whether or not to print messages
    director: If this is included, then director is used instead of Qmatrix

    return:
        cos(theta) in shape (n,n_molecules*n_atoms)
    """
    cdef np.ndarray Q,eigv,eigvec  
    if skip is None:
        skip = 1 
    cdef int t = len(LC)
    cdef np.ndarray time = np.linspace(0,LC.time,t)
    cdef int start_timeidx = np.searchsorted(time,start,side='left')
    cdef int end_timeidx = np.searchsorted(time,end,side='right')
    cdef np.ndarray time_idx = np.arange(start_timeidx,end_timeidx,skip)
    cdef int n = len(time_idx)
    cdef np.ndarray costheta=np.zeros((n,LC.n_molecules*LC.n_atoms))
    cdef int ix = 0

    use_director = False
    if isinstance(director,np.ndarray):
        use_director = True
        director = director[:,np.newaxis] # shape (3,1)
    
    if verbose:
        if use_director == True:
            print("Using director ", director)
        else:
            print("Using eigenvector of Q matrix to calculate direction")

    for ts in time_idx:  
        Q,CN_direction = LC.Qmatrix(ts) # CN_direction (N,3) 
        if use_director != True:
            director,_ = LC.director(Q) # director (3,1)

        b = np.dot(CN_direction,director) #of shpae (N,1)
        b = np.repeat(b,LC.n_atoms,axis=1).flatten() # of shape (N*N_atoms)
        costheta[ix] = b

        if verbose:
            print("time frame {} is being calculated".format(ts))
        ix += 1
    return (costheta,time_idx)

# find probability distribution of cos(theta) as a function of z 

# write the data to pdb file in the beta factor
cpdef pdb_bfactor(LC,pdb_name,data,verbose=False,sel_atoms=None): 
    """
    saves the data as beta-factor which is in shape (n,n_molecules*n_atoms) into pdb file specified by pdb_name
    
    LC: Liquid crystal object
    pdb_name: the name of the pdb out file
    data: a tuple containing (actual_data, time idx)
    output:
        saves a file with beta-factor with name pdb_name into the folder specified in LC object
    """
    cdef int ix = 0
    data_array,time_idx = data 
    LC.properties['universe'].add_TopologyAttr("tempfactors")
    with mda.Writer(LC.path+'/'+pdb_name, multiframe=True,bonds=None,n_atoms=LC.n_atoms) as PDB:
        for ts in time_idx: 
            LC.properties["universe"].trajectory[ts]
            if sel_atoms != None:
                uni = LC.properties['universe'].select_atoms(sel_atoms)
                uni.atoms.tempfactors = data_array[ix]
                PDB.write(uni.atoms)
            else:
                LC.properties["universe"].atoms.tempfactors = data_array[ix]
                PDB.write(LC.properties["universe"].atoms)

            if verbose:
                print("time frame {} has been written".format(ts))
            ix += 1     

# find cos(theta) as a function of r where r is the distance between the center of maasses, theta is the angle each molecule's
# CN bond forms with the director
cpdef cost_r(LC,int ts,np.ndarray COM_dist_mat,float min_,float max_,int bins=100):
    """
    calculates cos(theta) between two pairs of Liquid crystal molecules as a function of R between the Center of mass distances, this
    can only be performed on bulk liquid crystals

    LC: Liquid crystal object

    ts: the time frame at which this calculation is performed

    COM_dist_mat: a matrix with shape (n1,n2) which contains the distances between 
    group N1 and group N2. This should be a upper trigonal matrix for performace reasons

    min_: minimum distance between COM to consider

    max_: maximum distance between COM to consider

    bins: number of bins to bin the separation between min_ and max_
    """
    cdef np.ndarray bin_vec = np.linspace(min_,max_,bins)  
    cdef np.ndarray CN = LC.segments["CN"]
    cdef np.ndarray indices
    cdef int idx0,idx1
    cdef float cost_CN
    cdef np.ndarray cost_r = np.zeros((len(bin_vec)-1,))

    LC.properties["universe"].trajectory[ts]
    if LC.bulk == True:
        residues = LC.properties["universe"].residues
    else:
        residues = LC.properties["universe"].select_atoms("resname {}CB".format(LC.n)).residues
    
    for i in range(len(bin_vec)-1):
        low = bin_vec[i]
        high = bin_vec[i+1]
        indices = np.argwhere((COM_dist_mat >= low) & (COM_dist_mat < high))
        angle_sum = 0
        for idx in indices:
            idx0 = idx[0]
            idx1 = idx[1]
            res0 = residues[idx0]
            res1 = residues[idx1]
            N0 = res0.atoms[CN][0].position
            C0 = res0.atoms[CN][1].position

            CN0_vec = (N0 - C0)/np.sqrt(((N0 - C0)**2).sum())

            N1 = res1.atoms[CN][0].position
            C1 = res1.atoms[CN][1].position

            CN1_vec = (N1 - C1)/np.sqrt(((N1 - C1)**2).sum())
            cost_CN = np.dot(CN0_vec,CN1_vec)
            angle_sum += cost_CN
        if len(indices) == 0:
            cost_r[i] = 0
        else:
            cost_r[i] = angle_sum/len(indices)

    return cost_r

# find the probability distribution of cos(theta) where theta is the angle between a pair of nCB molecules
cpdef pcost_CN(LC,int ts,np.ndarray COM_dist_mat,str segment='benzene1',float distance_constraint=np.inf,int bins=100,g1=False):
    """
    find the p(cos(theta)) between CN subject to the constraint that segment is within distance_constraint

    ts: the time step for which p(cos(theta)) between CN vectors are calculated
    COM_dist_mat: the COM distance matrix (size NxN) while is it really just a upper triangular matrix without diagonal 
    segment: the segment at which the constraint is upon
    distance_constraint: the distance between segment
    bins: The number of bins to split between (-1,1)
    """
    cdef np.ndarray bin_vec = np.linspace(-1,1,bins)  
    cdef np.ndarray CN = LC.segments["CN"]
    cdef np.ndarray indices = np.argwhere((COM_dist_mat <= distance_constraint) & (COM_dist_mat > 0))
    cdef int ix=0,idx0,idx1
    cdef float cost_CN
    cdef np.ndarray cost_CN_vec = np.zeros((len(indices),))
    cdef np.ndarray digitized
    cdef np.ndarray npbin_count
    cdef np.ndarray normalized_pcost
    cdef list bin_count


    LC.properties["universe"].trajectory[ts]

    for idx in indices:
        idx0 = idx[0]
        idx1 = idx[1]
        res0 = LC.properties['universe'].residues[idx0]
        res1 = LC.properties['universe'].residues[idx1]
        N0 = res0.atoms[CN][0].position
        C0 = res0.atoms[CN][1].position

        CN0_vec = (N0 - C0)/np.sqrt(((N0 - C0)**2).sum())

        N1 = res1.atoms[CN][0].position
        C1 = res1.atoms[CN][1].position

        CN1_vec = (N1 - C1)/np.sqrt(((N1 - C1)**2).sum())
        cost_CN = np.dot(CN0_vec,CN1_vec)

        cost_CN_vec[ix] = cost_CN
        ix += 1

    cost_CN_vec = np.sort(cost_CN_vec)
    cost_CN_vec = cost_CN_vec[cost_CN_vec != np.inf]
    digitized = np.digitize(cost_CN_vec,bin_vec)
    bin_count = [(digitized == i).sum() for i in range(1,len(bin_vec))]
    npbin_count = np.array(bin_count)
     

    return (bin_vec[:-1],npbin_count) 

def density_z_atoms(LC,start_time,end_time,direction='z',skip=1,bins_z=100,verbose=False,Broken_interface=None):
    """
    calculates density of LC as a function of z

    LC: Liquid crystal object 
    start_time: the starting time of the calculation
    end_time: the ending time of the calculation
    direction: the direction where we can perform calculations along
    skip: the number of time frames to skip 
    
    returns: 
        density as a function of z (bins_z-1,)
    """
    t_idx = np.linspace(0,LC.time,len(LC))
    start_idx = np.searchsorted(t_idx,start_time,side='left')
    end_idx = np.searchsorted(t_idx,end_time,side='right')
    time_idx = np.arange(start_idx,end_idx,skip)
    density_z = np.zeros((bins_z-1,)) 

    if direction == 'x':
        d = 0
    
    if direction == 'y':
        d = 1

    if direction == 'z':
        d = 2

    u = LC["universe"]
    if Broken_interface is not None:
        Lx,Ly,Lz, draw_line = Broken_interface


    for ts in time_idx:
        # set the time frame to be ts
        u.trajectory[ts]
        pos = u.select_atoms("resname {}CB".format(LC.n)).atoms.positions

        # take only the dth dimension number of all COM 
        pos = pos[:,d] #(n_atoms,)
    
        if Broken_interface is not None:
            bot_idx = np.argwhere(pos <= draw_line)
            pos[bot_idx] += Lz

        pos_vec = np.linspace(pos.min(),pos.max(),bins_z)
 
        for j in range(bins_z-1):
            less = pos_vec[j]
            more = pos_vec[j+1]

            index = np.argwhere(((pos >= less) & (pos < more)))
            index = index.flatten()
            density_z[j] += len(index)/(Lx*Ly*(more-less))
        if verbose:
            print("time step {} is done".format(ts))        
    return density_z/len(time_idx)

def density_z_COM(LC,start_time,end_time,direction='z',segment='whole',skip=1,bins_z=100,verbose=False,Broken_interface=None):
    """
    calculates density of LC as a function of z

    LC: Liquid crystal object 
    start_time: the starting time of the calculation
    end_time: the ending time of the calculation
    direction: the direction where we can perform calculations along
    segment: the segment at which we calculate COM for in LC molecules
    skip: the number of time frames to skip 
    
    returns: 
        density as a function of z (bins_z-1,)
    """
    t_idx = np.linspace(0,LC.time,len(LC))
    start_idx = np.searchsorted(t_idx,start_time,side='left')
    end_idx = np.searchsorted(t_idx,end_time,side='right')
    time_idx = np.arange(start_idx,end_idx,skip)
    density_z = np.zeros((bins_z-1,)) 

    if direction == 'x':
        d = 0
    
    if direction == 'y':
        d = 1

    if direction == 'z':
        d = 2

    for ts in time_idx:
        # first find Center of mass matrix at time step "ts"
        COM_mat = LC.COM(ts,segment) #(n_molecues,3)

        # take only the dth dimension number of all COM 
        COM_mat = COM_mat[:,d] #(n_molecules,)

        # set the universe trajectory to "ts" time step
        LC['universe'].trajectory[ts]

        if LC.bulk == True:
            residues = LC['universe'].residues
        else:
            residues = LC['universe'].select_atoms("resname {}CB".format(LC.n)).residues

        COM_vec,Ntop = fix_interface(COM_mat,Broken_interface=Broken_interface,verbose=verbose,bins=bins_z)

        for i in range(bins_z-1):
            if Ntop != 0:
                if i >= Ntop:
                    j = i + 1
                else:
                    j = i
            else:
                j = i

            less = COM_vec[j]
            more = COM_vec[j+1]

            index = np.argwhere(((COM_mat >= less) & (COM_mat < more)))
            index = index.flatten()
            density_z[i] += len(index)
        if verbose:
            print("time step {} is done".format(ts))        
    return density_z/len(time_idx)

def fix_interface(vec,Broken_interface=None,verbose=False,bins=100):
    """
    Function that identifies broken interface and fixes it

    vec: a vector that holds the quantity that is broken up by interface (N,)
    Broken_interface:
        (a) None
        (b) a tuple that holds (Lz,draw_line)
    verbose: whether to be verbose during execution
    bins: number of bins to bin the direction z 
    
    returns:
        a new_vec of length (bins,)
    """
    vec_min,vec_max = vec.min(),vec.max()

    if Broken_interface == None:
        new_vec = np.linspace(vec_min,vec_max+1,bins)
        N_top = 0
        Nbot = bins-N_top
    else:
        Lz,draw_line = Broken_interface
        # first check if the max_COM - min_COM is larger than Lz
        if vec_max - vec_min >= Lz:
            # find COM above the line and find COM below the line
            vec_top = vec[vec>draw_line]
            vec_bot = vec[vec<=draw_line]
            N_top = len(vec_top)
            N_bot = len(vec_bot)
            N_top = round(N_top/(N_bot+N_top)*bins)
            N_bot = bins-N_top

            min_top = vec_top.min()
            max_top = vec_top.max()
            min_bot = vec_bot.min()
            max_bot = vec_bot.max()
            if verbose:
                print("min_top:{}".format(min_top))
                print("max_top:{}".format(max_top))
                print("min_bot:{}".format(min_bot))
                print("max_bot:{}".format(max_bot))

            bot_vec = np.linspace(min_bot,max_bot,N_bot)
            top_vec = np.linspace(min_top,max_top,N_top+1)
            new_vec = np.concatenate((top_vec,bot_vec))
        else:
            N_top = 0
            new_vec = np.linspace(vec_min,vec_max+1,bins)

    return new_vec,N_top
