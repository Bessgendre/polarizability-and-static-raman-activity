import polarizability as pol

from pyscf import gto, dft, grad, scf
from pyscf.hessian import thermo
import numpy as np

def displacement_each_mode(mol, mode, magnitude):
    new_mol = mol.copy()
    new_mol.set_geom_(mol.atom_coords(unit='Angstrom') + mode * magnitude, unit='Angstrom')
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

def activity_each_mode(polar_derivative, delta=0):
    # space average invariants
    a_2 = (np.trace(polar_derivative) / 3)**2
    gamma_2 = (((polar_derivative[0][0] - polar_derivative[1][1])**2
                + (polar_derivative[1][1] - polar_derivative[2][2])**2
                + (polar_derivative[2][2] - polar_derivative[0][0])**2) / 2
               + ((polar_derivative[0][1] + polar_derivative[1][0])**2
                  + (polar_derivative[1][2] + polar_derivative[2][1])**2
                  + (polar_derivative[0][2] + polar_derivative[2][0])**2) * 3 / 4)
    delta_2 = (((polar_derivative[0][1] - polar_derivative[1][0])**2
                + (polar_derivative[1][2] - polar_derivative[2][1])**2
                + (polar_derivative[0][2] - polar_derivative[2][0])**2) * 3 / 4)

    # calculate raman activity
    # NOTE: the quantity WITHOUT 45 as the denominator is the raman activity!
    activity_of_mode = 45 * a_2 + delta * delta_2 + 7 * gamma_2
    return activity_of_mode

def get_raman_activities_RHF(mol, delta=0):
    # calculate normal modes
    mf = scf.RHF(mol)
    mf.kernel()
    print('Polarizability (unit: Angstrom**3):')
    print(pol.get_polarizability_RHF(mol))
    mf.Gradients().grad()
    hessian_calculator = mf.Hessian()
    hess = hessian_calculator.kernel()

    # Perform harmonic analysis
    freq = thermo.harmonic_analysis(mol, hess)
    normal_modes = freq['norm_mode']
    freq_wavenumber = freq['freq_wavenumber']
    reduced_mass = freq['reduced_mass']

    # calculate raman activity
    raman_activities = []
    for i, mode in enumerate(normal_modes):
        polar_derivative = polarizability_derivative_each_mode_RHF(mol, mode)
        raman_activities.append(activity_each_mode(polar_derivative, delta=delta))
        print(f'frequency: {freq_wavenumber[i]} cm^-1')
        print('Mass-weighted normal mode:')
        print(mode * reduced_mass[i]**0.5)
        print('Polarizability derivative w.r.t. this mode (unit: Angstrom**2/sqrt(amu)):')
        print(polar_derivative)
        print(f'Raman activity (unit: Angstrom**4/amu): {raman_activities[-1]}')
        print()

    return np.array(raman_activities)
