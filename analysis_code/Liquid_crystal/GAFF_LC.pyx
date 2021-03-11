import numpy as np
import MDAnalysis as mda
from topology.sep_top import *

class GAFF_LC:
    """
    A class that represents Liquid crystals that is represented by GAFF which stands for General Amber force field from the paper J. Comput. Chem. 25, 1157–1174 (2004) and modified later by other group to be more suited for Liquid crystals in paper Phys. Chem. Chem. Phys. 17, 24851–24865 (2015).

    Args:
    ----
    itp(string): The path to the .itp file of the Liquid crystal molecule
    top(string): The path to the .tpr file of the Liquid crystal molecule
    xtc(string): The path to the .xtc file of the Liquid crystal molecule
    u_vec(string): The atoms at which the direction of the LC molecule is defined (default C11-C14 for the mesogen in the literature 2 shown)
    """
    def __init__(self,itp,top,xtc,u_vec='C11-C14'):
        self.itp = itp
        self.top = top
        self.xtc = xtc
        self.u = mda.Universe(self.top,self.xtc)
        self.septop = topology(self.itp)
        self.atom1 = u_vec.split("-")[0]
        self.atom2 = u_vec.split("-")[1]
        self.initialize()
       
    def initialize(self):
        u = self.u
        r = u.residues[0]

        for i in range(len(r.atoms)):
            atom = r.atoms[i]
            if atom.name == self.atom1:
                self.aidx1 = i
            elif atom.name == self.atom2:
                self.aidx2 = i

    def pos(self,ts):
        """
        Function that returns the position of each of the atoms in the system at time ts

        Args:
            ts(int): The time frame at which the calculation is performed upon

        Return:
            pos(numpy.ndarray): The position matrix at time ts of shape (N,3)
        """
        u = self.u
        u.trajectory[ts]

        return u.atoms.positions

    def get_celldimension(self,ts):
        """
        Function that returns the cell dimension that each of the time frame 

        Args:
            ts(int): The time frame that the user want to obtain cell dimension at 

        Return:
            cell_dimension(numpy.ndarray): The cell dimension of the box as a (3,) numpy array
        """
        u = self.u
        u.trajectory[ts]

        cell_dimension = u.dimensions[:3]

        return cell_dimension

    def COM(self,ts,segment=None):
        """
        Function that calculates the center of mass of the Liquid crystal molecule at time ts

        Args:
            ts(int): The time frame at which the calculation is performed on 
            segment(str): The segment that the user wants to calculate center of mass for (default None)

        Return:
            COM_mat(numpy.ndarray): The center of mass matrix of shape (N,3)
        """
        u = self.u
        u.trajectory[ts]
        N = len(u.residues)
        COM_mat = np.zeros((N,3))

        for i in range(N):
            res = u.select_atoms("resnum {}".format(i))
            if segment is None:
                COM_mat[i] = res.center_of_mass()
            else:
                num1,num2 = int(segment.split("-")[0]),int(segment.split("-")[1])
                atom_grp = res.atoms[num1:num2]
                COM_mat[i] = atom_grp.center_of_mass()

        return COM_mat
 
    def director_mat(self,ts,MOI=False):
        """
        Function that finds the director vector of all the residues in the system and put it in a matrix of shape (N,3). This can also find the director matrix using Moment of inertia tensor (the eigenvector that corresponds to the lowest eigenvalue of MOI tensor)

        Args:
            ts(int): The time frame of the simulation that this operation is performed upon
            MOI(bool): Whether or not to find director matrix using Moment of Inertia tensor

        Return:
            vec(numpy.ndarray): The director matrix of all the residues in the system

        """
        u = self.u
        u.trajectory[ts]
        aidx1 = self.aidx1
        aidx2 = self.aidx2
        director_mat = np.zeros((len(u.residues),3))

        for i in range(len(u.residues)):
            if MOI:
                director_mat[i] = u.select_atoms("resnum {}".format(i)).principal_axes()[-1]
            else:
                res = u.residues[i]
                a1 = res.atoms[aidx1].position
                a2 = res.atoms[aidx2].position

                r = (a2-a1)/np.sqrt(((a2-a1)**2).sum())
                director_mat[i] = r

        return director_mat

    def Qmatrix(self,ts,MOI=False):
        """
        Function that calculates the Qmatrix of the system at time ts.

        Args:
            ts(int): The time frame of the simulation

        Return:
            1.Qmatrix(numpy.ndarray)= The Q matrix of the liquid crystalline system
            2.eigvec(numpy.ndarray)=The director of the system at time ts
            3.p2(numpy.ndarray)=The p2 value of the system at time ts
        """
        d_mat = self.director_mat(ts,MOI=MOI)
        u = self.u
        N = len(u.residues)
        I = np.eye(3)

        Q = 3/(2*N)*np.matmul(d_mat.T,d_mat) - 1/2*I
        eigval,eigvec = np.linalg.eig(Q)
        order = np.argsort(eigval)
        eigval = eigval[order]
        eigvec = eigvec[:,order]

        return Q,eigvec[:,-1],-2*eigval[1]
    

    def match_dihedral(self,d_match):
        """
        Function that finds the dihedrals in the molecule by type
        
        Args:
        -----
            d_match(str): A string that contains the dihedral information passed in like "a1 a2 a3 a4"
        
        Return:
        -------
            dnum_list(list): A list of lists where each list contains 4 numbers corresponding to the index of the atom 
        """
        septop = self.septop
        d_list = septop.dihedrals_list
        d_match = d_match.split()
        dnum_list = []
        
        for d in d_list:
            if d.type() == d_match:
                dnum_list.append(d.atnum())
            elif list(reversed(d.type())) == d_match:
                dnum_list.append(d.atnum())
        
        return dnum_list
    
    def find_dangle_type(self,d_match,ts):
        """
        Function that finds the dihedral angles for all the dihedrals that matches the user specified dihedral in 
        the molecule
        
        Args:
        ----
            d_match(string): A string that contains the information passed in form "a1 a2 a3 a4"
            ts(int): The time frame that the user wants to dihedral angle to be calculated at 
            
        Return:
        ------
            dihedral(numpy.ndarray): shape (Nresidue*Ndihedral,) of all the dihdral angles that matches
            d_match in the molecule in degree
        
        """
        m = np.array(self.match_dihedral(d_match))
        u = self.u
        u.trajectory[ts] 
        pos = np.zeros((len(u.residues),len(m),4,3))

        for i in range(len(u.residues)):
            pos[i] = u.residues[i].atoms[m].positions
            
        pos = pos.reshape(-1,4,3)
        
        return self.d_angle(pos)*180/np.pi
                
    def d_angle(self,pos):
        """
        Function that finds the dihedral angle
        
        Args:
        -----
        pos(numpy.ndarray): A (N,4,3) matrix that contains the positions of the four atoms in the dihedral
        
        Return:
        -------
        Angle(float): The dihedral angle between the four atoms
        """
        a1,a2,a3,a4 = pos[:,0,:],pos[:,1,:],pos[:,2,:],pos[:,3,:]
        
        bond1 = (a1-a2)/np.sqrt(((a1-a2)**2).sum(axis=-1,keepdims=True)) # shape(N,3)
        bond2 = (a3-a2)/np.sqrt(((a3-a2)**2).sum(axis=-1,keepdims=True)) # shape(N,3)
        cosangle123 = (bond1*bond2).sum(axis=-1,keepdims=True) # shape (N,1)
        sinangle123 = np.sqrt(1 - cosangle123**2)
        n123 = np.cross(bond1,bond2,axis=-1)/sinangle123 # shape (N,3)
        
        bond3 = (a2-a3)/np.sqrt(((a2-a3)**2).sum(axis=-1,keepdims=True)) # shape (N,3)
        bond4 = (a4-a3)/np.sqrt(((a4-a3)**2).sum(axis=-1,keepdims=True)) #shape (N,3)
        cosangle234 = (bond3*bond4).sum(axis=-1,keepdims=True) #shape (N,1)
        sinangle234 = np.sqrt(1 - cosangle234**2)
        n234 = np.cross(bond3,bond4,axis=-1)/sinangle234 #shape (N,3)

        sign = (n123*bond4).sum(axis=-1,keepdims=True) #shape (N,1)
        sign = np.sign(sign)

        l = (n123*n234).sum(axis=-1,keepdims=True) 
        dangle = np.arccos(l)*sign
        if np.isnan(dangle).any():
            print(l[np.isnan(dangle)])
        dangle[dangle < 0] += 2*np.pi 

        return dangle
    
    def __len__(self):
        return len(self.u.trajectory)

def cost_r(LC,ts,min_,max_,MOI=False,nbins=101):
    """
    calculates cos(theta) between two pairs of Liquid crystal molecules as a function of R between the Center of mass distances, this
    can only be performed on bulk liquid crystals
    
    Args: 
        LC: Liquid crystal object
        ts(int): the time frame at which this calculation is performed
        min_(float): minimum distance between COM to consider
        max_(float): maximum distance between COM to consider
        MOI(bool): Whether or not to find director using moment of inertia tensor
        nbins(int): number of bins to bin the separation between min_ and max_
    """
    bin_vec = np.linspace(min_,max_,nbins)  
    cost_r = np.zeros((len(bin_vec),))
    # (N,3) matrix
    COM = LC.COM(ts)
    box = LC.get_celldimension(ts)

    # (N,N,3)
    COM_dist = abs(COM - COM[:,np.newaxis,:])
    # check pbc
    cond = COM_dist > box/2
    COM_dist = abs(COM_dist - cond*box)
    COM_dist = np.sqrt((COM_dist**2).sum(axis=-1)) # (N,N)
    COM_dist = np.triu(COM_dist)

    digitized = np.digitize(COM_dist,bin_vec,right=True)
    director = LC.director_mat(ts,MOI=MOI)
    costtheta = director.dot(director.T) #(N,N) matrix of cos(thetas) 
   
    for i in range(1,len(bin_vec)):
        where = np.argwhere(digitized == i)
        if len(where) == 0:
            cost_r[i] = 0
        else:
            cost_r[i] = costtheta[where[:,0],where[:,1]].sum()/len(where)

    return cost_r
