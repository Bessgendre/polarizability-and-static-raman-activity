# static polarizability module

from pyscf import gto, dft, grad, scf
import numpy as np
from pyscf.scf.hf import RHF
from pyscf.dft.rks import RKS
from pyscf.scf.hf import dip_moment

# Hartree-Fork SCF Method

def hcore_caused_by_elec_field(mol, e_field=np.array([0,0,0])):
    h1 = mol.intor('int1e_r')
    return np.einsum('xij,x->ij', h1, e_field)

class ModifiedRHF(RHF):
    def __init__(self, mol, e_field=np.zeros(3)):
        RHF.__init__(self, mol)
        self.e_field = e_field
        
    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return RHF.get_hcore(self, mol) + hcore_caused_by_elec_field(mol, self.e_field)

def get_dipole_with_field_RHF(mol, e_field=np.zeros(3)):
    mf = ModifiedRHF(mol, e_field=e_field)
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 100
    mf.kernel()
    dm = mf.make_rdm1()
    return dip_moment(mol, dm, unit='AU')

def get_dipole_at_grid_RHF(mol, direction, h):
    # preparation for central difference
    if direction == 'x':
        elec_grid = np.array([[i * h, 0, 0] for i in range(-2,3)])
    elif direction == 'y':
        elec_grid = np.array([[0, i * h, 0] for i in range(-2,3)])
    elif direction == 'z':
        elec_grid = np.array([[0, 0, i * h] for i in range(-2,3)])
    
    dipole_at_grid = []
    for e_field in elec_grid:
        dipole_at_grid.append(get_dipole_with_field_RHF(mol, e_field))
    
    return np.array(dipole_at_grid)

def get_polarizability_RHF(mol):
    # 4th order central difference
    h = 0.001
    central_difference_coeff = np.array([1/12, -2/3, 0, 2/3, -1/12])/h
    
    # calculate polarizability
    polarizability_caused_by_ex = np.dot(get_dipole_at_grid_RHF(mol, 'x', h).T, central_difference_coeff)
    polarizability_caused_by_ey = np.dot(get_dipole_at_grid_RHF(mol, 'y', h).T, central_difference_coeff)
    polarizability_caused_by_ez = np.dot(get_dipole_at_grid_RHF(mol, 'z', h).T, central_difference_coeff)
    
    # keep the column vector format
    return np.array([polarizability_caused_by_ex, polarizability_caused_by_ey, polarizability_caused_by_ez]).T

def get_beta_RHF(mol):
    alpha = get_polarizability_RHF(mol)
    a = np.trace(alpha)/3
    beta = alpha - a*np.identity(3)
    return beta

# DFT Method

class ModifiedRKS(RKS):
    def __init__(self, mol, xc='LDA,VWN', e_field=np.zeros(3)):
        RKS.__init__(self, mol, xc=xc)
        self.e_field = e_field
        
    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return RKS.get_hcore(self, mol) + hcore_caused_by_elec_field(mol, self.e_field)

def get_dipole_with_field_DFT(mol, e_field=np.zeros(3)):
    mf = ModifiedRKS(mol, e_field=e_field, xc='b3lyp')
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 100
    mf.kernel()
    dm = mf.make_rdm1()
    return dip_moment(mol, dm, unit='A.U.')

def get_dipole_at_grid_DFT(mol, direction, h):
    if direction == 'x':
        elec_grid = np.array([[i * h, 0, 0] for i in range(-2,3)])
    elif direction == 'y':
        elec_grid = np.array([[0, i * h, 0] for i in range(-2,3)])
    elif direction == 'z':
        elec_grid = np.array([[0, 0, i * h] for i in range(-2,3)])
    
    dipole_at_grid = []
    for e_field in elec_grid:
        dipole_at_grid.append(get_dipole_with_field_DFT(mol, e_field))
    
    return np.array(dipole_at_grid)

def get_polarizability_DFT(mol):
    # 4th order central difference
    h = 0.001
    central_difference_coeff = np.array([1/12, -2/3, 0, 2/3, -1/12])/h
    
    # calculate polarizability
    polarizability_caused_by_ex = np.dot(get_dipole_at_grid_DFT(mol, 'x', h).T, central_difference_coeff)
    polarizability_caused_by_ey = np.dot(get_dipole_at_grid_DFT(mol, 'y', h).T, central_difference_coeff)
    polarizability_caused_by_ez = np.dot(get_dipole_at_grid_DFT(mol, 'z', h).T, central_difference_coeff)
    
    # keep the column vector format
    return np.array([polarizability_caused_by_ex, polarizability_caused_by_ey, polarizability_caused_by_ez]).T

def get_beta_DFT(mol):
    alpha = get_polarizability_DFT(mol)
    a = np.trace(alpha)/3
    beta = alpha - a*np.identity(3)
    return beta