import numpy as np
import matplotlib.pyplot as plt

# specifically tailored for the 1D-polyacetylene chain in this numerical class
# should not be used anywhere else
# except for the help, no comments have been added to the code
# not supposed to be understood
# to be used as a black-box

def wannier(ham,N,state,pos,atom,alpha1=1.0,alpha2=1.0,beta1=-1.1,beta2=-0.9):
    """
    Computes Wannier orbitals for a hamiltonian of the polyacetylene. In the hamiltonian, d is set to 2 in this function.
    
    Mandatory arguments:
    * ham: function defined in this notebook, returns a 2x2 numpy array representing the hamiltonian
    * N: number of unit cells in real space
    * state: band number, equals 0 or 1
    * pos: position of the center of the Wannier orbital, float between 0 and N
    * atom: group in unit cell, 0 for A and 1 for B
    
    Optional arguments:
    * alpha1,alpha2,beta1,beta2: parameters of the hamiltonian
      default to 1.0, 1.0, -1.1 and -0.9
    
    Returns:
    x: an array of atomic positions in real space
    w: the real Wannier function for band 'state' at position 'pos' for atom 'atom'
    
    """
    
    uA=np.zeros((N+1,2),dtype=complex)
    uB=np.zeros((N+1,2),dtype=complex)
    eps=np.zeros((N+1,2))
    kpt=np.linspace(-np.pi/2,np.pi/2,N+1)
    
    for ik,k in enumerate(kpt):
        H=ham(alpha1,alpha2,beta1,beta2,2,k)
        e,v=np.linalg.eigh(H)
        eps[ik,:]=e
        uA[ik,:]=v[0,:]*np.exp(-1j*np.angle(v[0,:]))
        uB[ik,:]=v[1,:]*np.exp(-1j*np.angle(v[0,:]))
    
    psiA=uA[:-1,state]
    psiB=uB[:-1,state]*np.exp(1j*kpt[:-1])
    
    wA=np.fft.ifft(psiA*np.exp(-1*1j*kpt[:-1]*pos))
    wB=np.fft.ifft(psiB*np.exp(-1*1j*kpt[:-1]*pos))
    
    x=np.arange(0,N)*2
    
    if atom==0:
        return x,np.real(wA*np.exp(1j*np.pi/2*x))
    else:
        return x+1,np.real(wB*np.exp(1j*np.pi/2*x))

def wannierML(ham,N,state,pos,atom,alpha1=1.0,alpha2=1.0,beta1=-1.1,beta2=-0.9):
    """
    Computes maximally localised Wannier orbitals for a hamiltonian of the polyacetylene. In the hamiltonian, d is set to 2 in this function.
    
    Mandatory arguments:
    * ham: function defined in this notebook, returns a 2x2 numpy array representing the hamiltonian
    * N: number of unit cells in real space
    * state: band number, equals 0 or 1
    * pos: position of the center of the Wannier orbital, float between 0 and N
    * atom: group in unit cell, 0 for A and 1 for B
    
    Optional arguments:
    * alpha1,alpha2,beta1,beta2: parameters of the hamiltonian
      default to 1.0, 1.0, -1.1 and -0.9
    
    Returns:
    x: an array of atomic positions in real space
    w: the real maximally localised Wannier function for band 'state' at position 'pos' for atom 'atom'
    
    """
    
    uA=np.zeros((N+1,2),dtype=complex)
    uB=np.zeros((N+1,2),dtype=complex)
    eps=np.zeros((N+1,2))
    kpt=np.linspace(-np.pi/2,np.pi/2,N+1)
    
    for ik,k in enumerate(kpt):
        H=ham(alpha1,alpha2,beta1,beta2,2,k)
        e,v=np.linalg.eigh(H)
        eps[ik,:]=e
        uA[ik,:]=v[0,:]*np.exp(-1j*np.angle(v[0,:]))
        uB[ik,:]=v[1,:]*np.exp(-1j*np.angle(v[0,:]))
    
    M=np.zeros(N,dtype=complex)
    for ik in range(N):
        M[ik]=np.conj(uA[ik,state])*uA[ik+1,state]+np.conj(uB[ik,state])*uB[ik+1,state]
    Macc=np.zeros(N,dtype=complex)
    Macc[0]=M[0]
    for ik in range(1,N):
        Macc[ik]=M[ik]*Macc[ik-1]
    phase=-np.angle(Macc[-1])
    
    uAML=np.zeros(N+1,dtype=complex)
    uBML=np.zeros(N+1,dtype=complex)
    for ik in range(N):
        uAML[ik]=uA[ik,state]*np.exp(-1j*(phase*ik/N+np.angle(Macc[ik])))
        uBML[ik]=uB[ik,state]*np.exp(-1j*(phase*ik/N+np.angle(Macc[ik])))
    uAML[N]=uA[N,state]
    uBML[N]=uB[N,state]
    
    psiAML=uAML[:-1]
    psiBML=uBML[:-1]*np.exp(1j*kpt[:-1])
    
    wAML=np.fft.ifft(psiAML*np.exp(-1*1j*kpt[:-1]*pos))
    wBML=np.fft.ifft(psiBML*np.exp(-1*1j*kpt[:-1]*pos))
    
    x=np.arange(0,N)*2
    
    if atom==0:
        return x,np.real(wAML*np.exp(1j*np.pi/2*x))
    else:
        return x+1,np.real(wBML*np.exp(1j*np.pi/2*x))
