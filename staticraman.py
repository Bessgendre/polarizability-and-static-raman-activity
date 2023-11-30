import polarizability as pol

from pyscf import gto, dft, grad, scf
from pyscf.hessian import thermo
from pyscf.hessian import rhf
import numpy as np

def displacement_each_mode(mol, mode, magnitude):
    new_mol = mol.copy()
    new_atom_coords = mol.atom_coords(unit='Angstrom') + mode * magnitude
    elements = mol.elements
    
    atom_lines = []
    for i in range(len(elements)):
        each_atom = elements[i] + ' ' + str(new_atom_coords[i][0]) + ' ' + str(new_atom_coords[i][1]) + ' ' + str(new_atom_coords[i][2])
        atom_lines.append(each_atom)
        
    new_geometry = '; '.join(atom_lines)
    new_mol.atom = new_geometry
    new_mol.build()
    return new_mol

def polarizability_derivative_each_mode_RHF(mol, mode):
    h = 0.001
    # 4th order central difference
    displacement_grid = np.array([i * h for i in range(-2,3)])
    central_difference_coeff = np.array([1/12, -2/3, 0, 2/3, -1/12])/h
    
    polarizability_at_grid = []
    for displacement in displacement_grid:
        new_mol = displacement_each_mode(mol, mode, displacement)
        polarizability_at_grid.append(pol.get_polarizability_RHF(new_mol))
        
    polarizability_at_grid = np.array(polarizability_at_grid)
    polarizability_derivative = np.einsum('xij,x->ij', polarizability_at_grid, central_difference_coeff)
    
    return polarizability_derivative

def activity_each_mode(polar_derivative):
    
    # space average invariants
    a_2 = (np.trace(polar_derivative)/3)**2
    gamma_2 = 0.5*(polar_derivative[0][0] - polar_derivative[1][1])**2 + 0.5*(polar_derivative[1][1] - polar_derivative[2][2])**2 + 0.5*(polar_derivative[2][2] - polar_derivative[0][0])**2 + 3*polar_derivative[0][1]**2 + 3*polar_derivative[1][2]**2 + 3*polar_derivative[2][0]**2
    
    # calculate raman activity
    activity_of_mode = (45*a_2+7*gamma_2)/45
    return activity_of_mode

def get_raman_activities_RHF(mol): 
    # calculate normal modes
    mf = scf.RHF(mol)
    mf.kernel()
    hessian_calculator = rhf.Hessian(mf)
    hess = hessian_calculator.kernel()
    
    # Perform harmonic analysis
    freq = thermo.harmonic_analysis(mol, hess)
    normal_modes = freq['norm_mode']
    
    # calculate raman activity
    raman_activities = []
    for mode in normal_modes:
        polar_derivative = polarizability_derivative_each_mode_RHF(mol, mode)
        raman_activities.append(activity_each_mode(polar_derivative))
        
    return np.array(raman_activities)