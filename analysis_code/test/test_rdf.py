import sys
sys.path.insert(0,"../")
from rdf import *
import matplotlib.pyplot as plt


def run():
    pos = np.load("rdf_data/pos.npy")
    Lx=Ly=Lz = 58.88542198858854
    box = np.array([Lx,Ly,Lz])
    max_ = 28
    nbins = 100

    bins,gr = rdf(pos,pos,box,max_,nbins)

    return bins,gr

def test_rdf():
    bins,gr = run()
    reference_gr = np.load("rdf_data/reference.npy")
    diff = np.linalg.norm(reference_gr - gr,2)/len(gr)

    assert diff < 1e-10

if __name__ == "__main__":
    bins,gr = run()
    reference_gr = np.load("rdf_data/reference.npy")
    print(np.linalg.norm(reference_gr-gr,2)/len(gr))
    plt.plot(bins,reference_gr,label='reference')
    plt.plot(bins,gr,label='calculated')
    plt.legend()
    plt.savefig("rdf_data/test.png")
