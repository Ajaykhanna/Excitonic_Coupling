import re
import argparse
import numpy as np

"""
**Author**: Ajay Khanna  
**Date**: Dec.10.2023  
**Place**: UC Merced  
**Lab**: Dr. Isborn  

### 📧 Contact Information

- **Email**: [akhanna2@ucmerced.edu](mailto:akhanna2@ucmerced.edu) / [quantphobia@gmail.com](mailto:quantphobia@gmail.com)
- **GitHub**: [Ajaykhanna](https://github.com/Ajaykhanna) 🐱‍💻
- **Twitter**: [@samdig](https://twitter.com/samdig) 🐦
- **LinkedIn**: [ajay-khanna](https://www.linkedin.com/in/ajay-khanna) 💼
"""

# Coversion Factors
ANGS_2_BOHR = 1.8897259885789
HA_2_eV = 27.211396132

def banner(title, char='=', width=80):
    """
    This function prints a banner with the specified title.
    
    :param title: The title of the banner.
    :param char: The character used to create the border of the banner. Default is '='.
    :param width: The width of the banner. Default is 80.
    """
    print(char * width)
    print(title.center(width))
    print(char * width)

# Print the banner
banner("Coulombic Coupling Via PDA")


def read_xyz(input_file):
    """
    This Python function reads the atomic coordinates from an XYZ input file.
    
    :param input_file: It looks like you have provided a code snippet for reading XYZ coordinates from a
    file. However, the input_file parameter is missing. Please provide the path to the XYZ file you want
    to read in order to use the read_xyz function
    :return: The function `read_xyz` reads an input file containing information about atomic coordinates
    and returns a numpy array `input_geometry` with shape (natoms, 4) where each row represents an atom
    with its symbol and x, y, z coordinates.
    """
    natoms = 0
    with open(input_file) as f:
        for line in f:
            if "NAtoms=" in line:
                natoms = int(line.split()[3])
                break

        f.seek(0) 
        input_geometry = np.zeros((natoms, 4))
        for line in f:
            if "Input orientation:" in line:
                # Skip 4 lines after finding "Input orientation:"
                for _ in range(4):
                    next(f)
                # Read the next 'natoms' lines for coordinates
                for row in range(natoms):
                    line = next(f).split()
                    symb = int(line[1])
                    x, y, z = map(float, line[3:6])
                    input_geometry[row] = symb, x, y, z
                break  # Exit loop after reading all atoms

    return input_geometry

def extract_atomic_weights(log_file_path):
    """
    The function `extract_atomic_weights` reads a log file, extracts atomic weights from lines starting
    with ' AtmWgt=', and returns them as a NumPy array.
    
    :param log_file_path: The `log_file_path` parameter should be a string representing the file path to
    the log file from which you want to extract atomic weights. Make sure to provide the full path to
    the log file including the file name and extension (e.g., "C:/logs/logfile.txt")
    :return: The function `extract_atomic_weights` returns a NumPy array containing the atomic weights
    extracted from the log file specified by the `log_file_path` parameter.
    """
    atomic_weights = []
    with open(log_file_path, 'r') as file:
        for line in file:
            if line.startswith(' AtmWgt='):
                # Extract all the floating-point numbers from the line
                weights = re.findall(r'\d+\.\d+', line)
                atomic_weights.extend(map(float, weights))
                
    return np.array(atomic_weights)

def vertical_excitation_energies(file_path, excited_state = 1):
    """
    This Python function reads a file, searches for a specific pattern, extracts a value from the
    matching line, and returns it as a float.
    
    :param file_path: The `file_path` parameter in the `vertical_excitation_energies` function is a
    string that represents the path to the file from which you want to extract vertical excitation
    energies. This function reads the specified file and looks for the vertical excitation energy value
    for a specified excited state
    :param excited_state: The `excited_state` parameter in the `vertical_excitation_energies` function
    is used to specify which excited state's energy you want to extract from the file. It is an optional
    parameter with a default value of 1, meaning that if you do not provide a value for `exc, defaults
    to 1 (optional)
    :return: the vertical excitation energy in electron volts (eV) for the specified excited state from
    the file located at the given file path.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the index of the line containing the specified pattern
    start_index = next((i for i, line in enumerate(lines) if f'Excited State   {excited_state}:      Singlet-?Sym' in line), None)
    
    if start_index is not None:
        vee_lines = lines[start_index]
        fields = vee_lines.split()
        if len(fields) >= 5:
            vee_eV = fields[4]
            
    return float(vee_eV)
            

def extract_TDM_xyz_values(file_path):
    """
    This function extracts the X, Y, and Z values of transition dipole moments from a file containing
    TDM xyz values.
    
    :param file_path: The function `extract_TDM_xyz_values` reads a file and extracts the X, Y, and Z
    values related to transition electric dipole moments (TDM) from a specific line in the file. To use
    this function, you need to provide the file path as the `file_path` parameter
    :return: The function `extract_TDM_xyz_values` reads a file specified by `file_path`, searches for a
    specific pattern in the file, and extracts the X, Y, and Z values related to transition electric
    dipole moments (TDM). If the pattern is found, it returns a NumPy array containing the extracted X,
    Y, and Z values of the TDM.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the index of the line containing the specified pattern
    start_index = next((i for i, line in enumerate(lines) if 'Ground to excited state transition electric dipole moments (Au):' in line), None)
    
    # Extract the relevant line
    if start_index is not None:
        xyz_line = lines[start_index+2]
        
        # Extract the X, Y, and Z values
        fields = xyz_line.split()
        if len(fields) >= 6:
            tdm_x, tdm_y, tdm_z = float(fields[1]), float(fields[2]), float(fields[3])

    return np.array([tdm_x, tdm_y, tdm_z])

def center_of_mass(atomic_weights, coordinates):
    """
    The function calculates the center of mass of a system based on atomic weights and coordinates.
    
    :param atomic_weights: Atomic weights are the weights of individual atoms in a molecule or system.
    They are typically given in atomic mass units (u) and represent the mass of each atom relative to
    the unified atomic mass unit (approximately the mass of a proton or neutron)
    :param coordinates: Coordinates is a NumPy array containing the x, y, and z coordinates of atoms in
    a molecule. Each row represents the coordinates of an atom in the format [x, y, z]
    :return: The function `center_of_mass` is returning the coordinates of the center of mass as a NumPy
    array with three elements: x_COM, y_COM, and z_COM.
    """
    #print(atomic_weights)
    assert len(atomic_weights) == len(coordinates)
    total_mass = np.sum(atomic_weights)
    
    weighted_x = np.sum(np.dot(atomic_weights, coordinates[:,0]))
    weighted_y = np.sum(np.dot(atomic_weights, coordinates[:,1]))
    weighted_z = np.sum(np.dot(atomic_weights, coordinates[:,2]))
    
    x_COM, y_COM, z_COM = weighted_x/total_mass, weighted_y/total_mass, weighted_z/total_mass
    
    return np.array([x_COM, y_COM, z_COM])
    
def coupling_via_PDA(TDM_D,TDM_A,COM_D,COM_A):
    """
    This Python function calculates the coupling between two transition dipole moments using the
    polarizability derivative approximation method.
    
    :param TDM_D: The `TDM_D` parameter in the `coupling_via_PDA` function likely represents the
    Transition Dipole Moment (TDM) for molecule D. This parameter is used in the calculation to
    determine the coupling between two molecules based on their transition dipole moments and center of
    mass positions
    :param TDM_A: The function `coupling_via_PDA` calculates the coupling between two transition dipole
    moments (TDMs) in a photoinduced electron transfer process. The parameters required for the function
    are `TDM_D`, `TDM_A`, `COM_D`, and `COM_A`
    :param COM_D: It looks like you have provided a function `coupling_via_PDA` that calculates coupling
    between two transition dipole moments based on certain parameters. However, you have not provided
    the complete information about the parameters. Could you please provide the missing information
    about the parameters COM_D, COM_A, TDM
    :param COM_A: It looks like you were about to provide some information about the parameters, but you
    stopped after mentioning "COM_A:". Could you please provide the rest of the information for the
    parameters COM_A, TDM_D, TDM_A, and COM_D so that I can assist you further with the `cou
    :return: The function `coupling_via_PDA` is returning a value calculated based on the input
    parameters `TDM_D`, `TDM_A`, `COM_D`, and `COM_A`. The returned value is calculated as 27.211396132
    multiplied by the expression `((np.dot(TDM_D,TDM_A)/R**3) - (3*(np.dot(TDM_D,COM
    """
    
    COM_D=COM_D*ANGS2BOHR
    COM_A=COM_A*ANGS2BOHR
    R=np.linalg.norm(COM_A-COM_D)
    return HA_2_eV * ((np.dot(TDM_D,TDM_A)/R**3) - (3*(np.dot(TDM_D,COM_D)*np.dot(TDM_A,COM_A))/R**5))

def main():
    parser = argparse.ArgumentParser(description='Process TDM and COM data.')
    parser.add_argument('--dye_1_filename', type=str, help='Filename of dye 1 log file')
    parser.add_argument('--dye_2_filename', type=str, help='Filename of dye 2 log file')
    parser.add_argument('--nAtoms_dye1', type=int, help='Number of atoms in dye 1')
    parser.add_argument('--natoms_dye2', type=int, help='Number of atoms in dye 2')
    args = parser.parse_args()

    TDM_D, COM_D = extract_TDM_xyz_values(args.dye_1_filename), center_of_mass(extract_atomic_weights(args.dye_1_filename), read_xyz(args.dye_1_filename)[0:args.nAtoms_dye1][:,1:])
    TDM_A, COM_A = extract_TDM_xyz_values(args.dye_2_filename), center_of_mass(extract_atomic_weights(args.dye_2_filename), read_xyz(args.dye_2_filename)[0:args.natoms_dye2][:,1:])

    print(f'NBD TDMs: {TDM_D}')
    print(f'NR TDMs : {TDM_D}')
    print(f'Excitonic Coupling via PDA: {coupling_via_PDA(TDM_D, TDM_A, COM_D, COM_A):.3} eV')

if __name__ == '__main__':
    main()
