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

# Coversion Factors
ANGS_2_BOHR = 1.8897259885789
HA_2_eV = 27.211396132


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
banner("Coulombic Coupling Via Diabatization")


def vertical_excitation_energies(file_path, excited_state=1):
    """
    Reads a file and extracts the vertical excitation energy for a specified excited state.

    Args:
        file_path (str): The path to the file containing the vertical excitation energy data.
        excited_state (int, optional): The index of the excited state for which to extract
            the vertical excitation energy. Defaults to 1.

    Returns:
        float: The vertical excitation energy in electron volts (eV) for the specified excited state.
    """
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
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find the index of the line containing the specified pattern
    start_index = next(
        (
            i
            for i, line in enumerate(lines)
            if f"Excited State   {excited_state}:      Singlet-?Sym" in line
        ),
        None,
    )

    if start_index is not None:
        vee_lines = lines[start_index]
        fields = vee_lines.split()
        if len(fields) >= 5:
            vee_eV = fields[4]

    return float(vee_eV) / HA_2_eV


def extract_TDM_xyz_values(file_path, excited_state=1):
    """
    Extracts the transition electric dipole moment (TDM) values for a specified excited state from a file.

    Args:
        file_path (str): The path to the file containing the TDM data.
        excited_state (int, optional): The index of the excited state for which to extract the TDM values. Defaults to 1.

    Returns:
        list[float]: A list of the x, y, and z components of the TDM for the specified excited state, or an error message if the pattern or excited state is not found.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find the index of the pattern
    pattern = "Ground to excited state transition electric dipole moments (Au):"
    start_index = next((i for i, line in enumerate(lines) if pattern in line), None)

    if start_index is None:
        return f"Pattern '{pattern}' not found in the file."

    # Extract the required rows
    data_lines = lines[start_index + 2 : start_index + 5]

    # Find the row with the specified excited state
    for line in data_lines:
        row = line.split()
        try:
            if int(row[0]) == excited_state:
                return [float(value) for value in row[1:4]]
        except ValueError:
            continue

    return f"Excited state {excited_state} not found in the expected rows."


def diabatize(dims1, dims2, monA, monB, E1, E2):
    """
    Computes the diabatic coupling (J) between the first two excited states of a dimer
    molecule using the transition dipole moments (TDMs) of the monomer states.

    Note: Units of both TDMs and energies are in atomic units and should be.

    Parameters:
        dims1 (1xn matrix): TDMs of the s1 and s2 states of the dimer.
        dims2 (1xn matrix): TDMs of the s1 and s2 states of the dimer.
        monA (1xn matrix): TDMs of the s1 state of monomer A.
        monB (1xn matrix): TDMs of the s1 state of monomer B.
        E1 (float): Energy of the s1 state of the dimer.
        E2 (float): Energy of the s2 state of the dimer.

    Returns:
        float: The diabatic coupling (J) between the first two excited states of the dimer.
    """
    """
    Uses the either the TDMs of the s1 and s2 states of the dimer and the s1 state of the two monomers, to
    diabatize the adiabatic Hamiltonian of first two excited states (E1 and E1) to the diabatic
    Hamiltonian, where the off diagonal terms are the couplings J

    Accepts 1xn matrices and state energies as inputs

    Parameters
    ----------
    TDM1_dimer,TDM2_dimer,TDM_donor,TDM_acceptor: 1x3 matrices
    E1,E2: floats of the energy of the s1 and s2 states of the dimer
    Returns
    ----------
    2x2 matrix
    """

    dimer = np.concatenate((dims1, dims2)).reshape(2, len(dims1))
    monomer = np.concatenate((monA, monB)).reshape(2, len(monA))

    M = np.dot(dimer, monomer.T)

    U, s, Vt = np.linalg.svd(M)

    C = (np.dot(U, Vt)).transpose()

    E = np.matrix(([E1, 0], [0, E2]))

    H = np.dot(np.dot(C, E), C.transpose())
    J = H[0, 1]
    return J


def main():
    parser = argparse.ArgumentParser(description="Supramolecular coupling calculation")
    parser.add_argument("--dimer_filename", type=str, help="Dimer log file")
    parser.add_argument(
        "--excited_states", type=int, nargs="+", help="Excited states to consider"
    )
    parser.add_argument("--donor_filename", type=str, help="Donor log file")
    parser.add_argument("--acceptor_filename", type=str, help="Acceptor log file")
    args = parser.parse_args()

    acceptor_file = args.acceptor_filename
    donor_file = args.donor_filename
    dimer_file = args.dimer_filename
    excited_states = args.excited_states

    dimer_TDM_1, E1 = extract_TDM_xyz_values(
        dimer_file, excited_state=excited_states[0]
    ), vertical_excitation_energies(dimer_file, excited_state=1)
    dimer_TDM_2, E2 = extract_TDM_xyz_values(
        dimer_file, excited_state=excited_states[1]
    ), vertical_excitation_energies(dimer_file, excited_state=2)
    acceptor_TDM, donor_TDM = extract_TDM_xyz_values(
        acceptor_file, excited_state=1
    ), extract_TDM_xyz_values(donor_file, excited_state=1)

    print(f"Dimer TDM-1: {dimer_TDM_1}, Dimer E1: {E1 * HA_2_eV:.4} eV")
    print(f"Dimer TDM-2: {dimer_TDM_2}, Dimer E2: {E2 * HA_2_eV:.4} eV")
    print(f"Acceptor TDM: {acceptor_TDM},\nDonor TDM: {donor_TDM}")

    print(
        f"The Coupling From Diabetization is: {diabatize(dimer_TDM_1, dimer_TDM_2, acceptor_TDM, donor_TDM, E1, E2) * HA_2_eV} eV"
    )


if __name__ == "__main__":
    main()
