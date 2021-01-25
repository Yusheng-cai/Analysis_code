import numpy as np 
from analysis_code.timeseries import Timeseries 
cimport numpy as np 
import MDAnalysis as mda 
import os 

class simulation:
    """
    path: path to the MD folder, it has to have the follwing files
    1. {md_name}.tpr
    2. {md_name}_pbc.xtc
    
    xvg_files: any xvg files that are not the above two, should come in as a list, contains 
    but not limited to the following
    1. density
    2. pressure 
    3. volume
    etc.
     
    attributes:
    self.time: simulation time of the MD (in ns)
    self.properties: Each of the properties held in this object which includes 
                    1. Universe
                    2. everything in the xvg files (e.g. density, volume)
                    3. kwargs**
    """
    def __init__(self,path,time,xvg_files=None):
        # check if .tpr & .xtc files are found
        self.mdname = path.split("/")[-1]
        self.path = path
        self.time = time
        self.properties = {}
        self.files = os.listdir(path)

        if "{}.tpr".format(self.mdname) not in self.files:
            raise Exception("{}.tpr is not found in designated path!".format(self.mdname))
        tpr = path+"/{}.tpr".format(self.mdname)

        if "{}_pbc.xtc".format(self.mdname) not in self.files:
            raise Exception("{}_pbc.xtc is not found in designated path!".format(self.mdname))
        xtc = path+"/{}_pbc.xtc".format(self.mdname)
        

        u = mda.Universe(tpr,xtc)
        self.properties["universe"] = u

        if xvg_files is not None:
            for name in xvg_files:
                if "{}_{}.xvg".format(self.mdname,name) not in self.files:
                    raise Exception("{}_{}.xvg is not found in designated path!".format(self.mdname,name))
                data = self.xvg_reader(path+ "/{}_{}.xvg".format(self.mdname,name))
                t = np.linspace(0,self.time,len(data))
                self.properties[name] = Timeseries(data,t) 
        
        self.properties_keys = [key for key in self.properties]

    def xvg_reader(self,file):
        """
        function that reads xvg files

        Args:
        ----
            file(str): input file path

        Return:
        ------
            data(numpy.ndarray) that is contained within the xvg file
        """
        f = open(file)

        lines = f.readlines()

        f.close()
        # define the comment symbols which are ignored in .xvg files
        comment_symbol = ['#','@']

        # ignore the lines which starts with comment symbols and take the second number (value of interest)
        xvgdata = np.array([float(line.rstrip(" ").lstrip(" ").split()[1]) for line in lines if line[0] not in comment_symbol])
    
        return xvgdata
   
    def get_full_traj(self):
        """
        this is a function that extracts all the trajectory information from the xtc file
        """
        u_traj = np.zeros((len(self.properties["universe"].trajectory),len(self.properties["universe"].atoms),3))
        time = np.linspace(0,self.time,num=len(u_traj)) 
        tix = 0

        for ts in self.properties["universe"].trajectory:
            u_traj[tix] = self.properties["universe"].atoms.positions
            tix += 1

        self.properties['trajectory'] = Timeseries(u_traj,time) 

    def __getitem__(self,key):
        return self.properties[key]

    def __len__(self):
        return len(self.properties["universe"].trajectory)
