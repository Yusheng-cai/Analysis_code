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

    def COM(self,ts):
        """
        Function that calculates the center of mass of the Liquid crystal molecule at time ts

        Args:
        -----
            ts(int): The time frame at which the calculation is performed on 

        Return:
            COM_mat(numpy.ndarray): The center of mass matrix of shape (N,3)
        """
        u = self.u
        u.trajectory[ts]
        N = len(u.residues)
        COM_mat = np.zeros((N,3))

        for i in range(N):
            res = u.select_atoms("resnum {}".format(i))
            COM_mat[i] = res.center_of_mass()

        return COM_mat

 
    def director_mat(self,ts):
        """
        Function that finds the director vector of all the residues in the system and put it in a matrix of shape (N,3)

        Args:
        -----
        ts(int): The time frame of the simulation that this operation is performed upon

        Return:
        ------
        vec(numpy.ndarray): The director matrix of all the residues in the system

        """
        u = self.u
        u.trajectory[ts]
        aidx1 = self.aidx1
        aidx2 = self.aidx2
        director_mat = np.zeros((len(u.residues),3))
        ix = 0

        for res in u.residues:
            a1 = res.atoms[aidx1].position
            a2 = res.atoms[aidx2].position

            r = (a2-a1)/np.sqrt(((a2-a1)**2).sum())
            director_mat[ix] = r
            ix += 1

        return director_mat

    def Qmatrix(self,ts):
        """
        Function that calculates the Qmatrix of the system at time ts.

        Args:
        ----
        ts(int): The time frame of the simulation

        Return:
        ------
        1.Qmatrix(numpy.ndarray)= The Q matrix of the liquid crystalline system
        2.eigvec(numpy.ndarray)=The director of the system at time ts
        3.p2(numpy.ndarray)=The p2 value of the system at time ts
        """
        d_mat = self.director_mat(ts)
        u = self.u
        N = len(u.residues)
        I = np.eye(3)

        Q = 3/(2*N)*np.matmul(d_mat.T,d_mat) - 1/2*I
        eigval,eigvec = np.linalg.eig(Q)
        order = np.argsort(eigval)
        eigval = eigval[order]
        eigvec = eigvec[order]

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
