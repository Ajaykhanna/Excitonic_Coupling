"""
**Author**: Ajay Khanna
**Date**: Dec.10.2023
**Place**: UC Merced
**Lab**: Dr. Isborn

### 📧 Contact Information

- **Email**: [akhanna2@ucmerced.edu](mailto:akhanna2@ucmerced.edu) / [quantphobia@gmail.com](mailto:quantphobia@gmail.com)
- **GitHub**: [Ajaykhanna](https://github.com/Ajaykhanna) 🐱<200d>💻
- **Twitter**: [@samdig](https://twitter.com/samdig) 🐦
- **LinkedIn**: [ajay-khanna](https://www.linkedin.com/in/ajay-khanna) 💼
"""

import argparse
import numpy as np


def banner(title, char="=", width=80):
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
banner("Coulombic Coupling Via Atomic Transition Charges")
ANGS_TO_BOHRS = 1.8897259885789
HA_TO_EV = 27.211396132


def read_Zxyzs(input_file: str) -> np.ndarray:
    """
    Reads the atomic coordinates from an XYZ input file.

    Parameters:
    - input_file (str): The path to the XYZ file containing the atomic coordinates.

    Returns:
    - input_geometry (np.ndarray): A numpy array with shape (natoms, 4) where each row represents an atom
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
                    atNum = int(line[1])
                    x, y, z = map(float, line[3:6])
                    input_geometry[row] = atNum, x, y, z
                break  # Exit loop after reading all atoms

    return input_geometry


def read_NTOs(g09_file: str, natoms: int, atNums: list) -> np.ndarray:
    """
    Reads a G09 logfile and returns the atomic-centered Natural Transition Charges (NTO),
    obtained via the G09 input line:
    td=(nstates=1) nosymm Pop=NTO Density=(Transition=1)

    Parameters:
    - g09_file: str
        Path to the G09 log file.
    - natoms: int
        Number of atoms.
    - atNums: list
        List of atomic numbers corresponding to each atom.

    Returns:
    - NTO: np.ndarray
        Array of NTO charges in order of atomic positions.
    """
    NTO = np.zeros(natoms)
    with open(g09_file) as f:
        # Skip lines until Mulliken charges are found
        for line in f:
            if line.strip():
                if " Mulliken charges:" in line:
                    line = next(f)
                    line = next(f)
                    # Read charges for each atom
                    for i in range(natoms):
                        charge_line = line.split()
                        charge = float(charge_line[2])
                        NTO[i] = float(atNums[i]) - charge
                        line = next(f)
        f.close()
    return NTO


def coupling_via_TC(
    NTO_1: np.ndarray,
    NTO_2: np.ndarray,
    coordinates_1: np.ndarray,
    coordinates_2: np.ndarray,
) -> float:
    """
    Calculates the CATC exciton coupling J based on the Coulomb interaction
    between Atomic Transition Charges in two molecules.

    Parameters:
    - NTO_1: np.ndarray[float]
        List of NTO charges for molecule 1.
    - NTO_2: np.ndarray[float]
        List of NTO charges for molecule 2.
    - coordinates_1: np.ndarray
        Nx3 array of x, y, z coordinates for molecule 1.
    - coordinates_2: np.ndarray
        Nx3 array of x, y, z coordinates for molecule 2.

    Returns:
    - J: float
        Exciton coupling.
    """
    coordinates_1 = coordinates_1 * ANGS_TO_BOHRS
    coordinates_2 = coordinates_2 * ANGS_TO_BOHRS

    J = 0
    for i in range(len(NTO_1)):
        for j in range(len(NTO_2)):
            J += (NTO_1[i] * NTO_2[j]) / (
                np.linalg.norm(coordinates_2[j] - coordinates_1[i])
            )

    return J


def main():
    parser = argparse.ArgumentParser(description="Process TDM and COM data.")
    parser.add_argument("--dye_1_filename", type=str, help="Filename of dye 1 log file")
    parser.add_argument("--dye_2_filename", type=str, help="Filename of dye 2 log file")
    parser.add_argument("--nAtoms_dye1", type=int, help="Number of atoms in dye 1")
    parser.add_argument("--nAtoms_dye2", type=int, help="Number of atoms in dye 2")
    args = parser.parse_args()

    dye1_coords = read_Zxyzs(args.dye_1_filename)
    dye2_coords = read_Zxyzs(args.dye_2_filename)

    dye1_NTOs = read_NTOs(args.dye_1_filename, args.nAtoms_dye1, dye1_coords[:, 0])
    dye2_NTOs = read_NTOs(args.dye_2_filename, args.nAtoms_dye2, dye2_coords[:, 0])

    # print(f'NBD TCs: {dye1_NTOs}')
    # print(f'NR TCs : {dye2_NTOs}')
    print(
        f"Excitonic Coupling via Transition Charges: {coupling_via_TC(dye1_NTOs, dye2_NTOs, dye1_coords[:,1:], dye2_coords[:,1:]) * HA_TO_EV:.5} eV"
    )


if __name__ == "__main__":
    main()
