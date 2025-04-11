"""
Author: Ajay Khanna
Date: Dec. 10, 2023
Place: UC Merced
Lab: Dr. Isborn

This script calculates the diabatic coupling (J) between the first two excited states of a dimer
molecule using the transition dipole moments (TDMs) of both the monomer and dimer states.

Contact Information:
- Email: akhanna2@ucmerced.edu / quantphobia@gmail.com
- GitHub: https://github.com/Ajaykhanna
- Twitter: @samdig
- LinkedIn: https://www.linkedin.com/in/ajay-khanna
"""

import argparse
import numpy as np

# Conversion Factors
ANGS_2_BOHR = 1.8897259885789
HA_2_EV = 27.211396132


def print_banner(title, char="=", width=80):
    """
    Print a banner with the specified title.

    Args:
        title (str): The title of the banner.
        char (str, optional): The character used to create the border of the banner. Default is '='.
        width (int, optional): The width of the banner. Default is 80.
    """
    print(char * width)
    print(title.center(width))
    print(char * width)


# Print the banner
print_banner("Coulombic Coupling Via Diabatization")


def extract_vertical_excitation_energy(file_path, excited_state=1):
    """
    Extract the vertical excitation energy for a specified excited state from a file.

    Args:
        file_path (str): The path to the file containing the vertical excitation energy data.
        excited_state (int, optional): The index of the excited state for which to extract
            the vertical excitation energy. Default is 1.

    Returns:
        float: The vertical excitation energy in electron volts (eV) for the specified excited state,
            or None if the excited state is not found.
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
        vee_line = lines[start_index]
        fields = vee_line.split()
        if len(fields) >= 5:
            vee_hartree = float(fields[4])
            vee_ev = vee_hartree * HA_2_EV
            return vee_ev

    return None


def extract_tdm_xyz_values(file_path, excited_state=1, max_states=10):
    """
    Extract the transition electric dipole moment (TDM) values for a specified excited state from a file.

    Args:
        file_path (str): The path to the file containing the TDM data.
        excited_state (int, optional): The index of the excited state for which to extract the TDM values.
            Default is 1.
        max_states (int, optional): The maximum number of excited states to consider. Default is 10.

    Returns:
        list[float] or str: A list of the x, y, and z components of the TDM for the specified excited state,
            or an error message if the pattern or excited state is not found.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find the index of the pattern
    pattern = "Ground to excited state transition electric dipole moments (Au):"
    start_index = next((i for i, line in enumerate(lines) if pattern in line), None)

    if start_index is None:
        return f"Pattern '{pattern}' not found in the file."

    # Extract the required rows
    data_lines = lines[start_index + 2 : start_index + max_states]

    # Find the row with the specified excited state
    for line in data_lines:
        row = line.split()
        try:
            if int(row[0]) == excited_state:
                return [float(value) for value in row[1:4]]
        except ValueError:
            continue

    return f"Excited state {excited_state} not found in the expected rows."


def diabatize(dims1, dims2, mon_a, mon_b, e1, e2):
    """
    Compute the diabatic coupling (J) between the first two excited states of a dimer
    molecule using the transition dipole moments (TDMs) of the monomer states.

    Note: Units of both TDMs and energies should be in atomic units.

    Args:
        dims1 (numpy.ndarray): TDMs of the s1 state of the dimer.
        dims2 (numpy.ndarray): TDMs of the s2 state of the dimer.
        mon_a (numpy.ndarray): TDMs of the s1 state of monomer A.
        mon_b (numpy.ndarray): TDMs of the s1 state of monomer B.
        e1 (float): Energy of the s1 state of the dimer.
        e2 (float): Energy of the s2 state of the dimer.

    Returns:
        float: The diabatic coupling (J) between the first two excited states of the dimer.
    """
    dimer = np.concatenate((dims1, dims2)).reshape(2, len(dims1))
    monomer = np.concatenate((mon_a, mon_b)).reshape(2, len(mon_a))

    m = np.dot(dimer, monomer.T)

    u, s, vt = np.linalg.svd(m)

    c = (np.dot(u, vt)).transpose()

    e = np.matrix(([e1, 0], [0, e2]))

    h = np.dot(np.dot(c, e), c.transpose())
    j = h[0, 1]
    return j


def main():
    """
    Compute the diabatic coupling (J) between the first two excited states of a dimer
    molecule using the transition dipole moments (TDMs) of the monomer states.

    Note: Units of both TDMs and energies should be in atomic units.

    Args:
        dimer_tdm_1 (numpy.ndarray): TDMs of the s1 state of the dimer.
        dimer_tdm_2 (numpy.ndarray): TDMs of the s2 state of the dimer.
        acceptor_tdm (numpy.ndarray): TDMs of the s1 state of the acceptor monomer.
        donor_tdm (numpy.ndarray): TDMs of the s1 state of the donor monomer.
        e1 (float): Energy of the s1 state of the dimer.
        e2 (float): Energy of the s2 state of the dimer.

    Returns:
        float: The diabatic coupling (J) between the first two excited states of the dimer.
    """

    parser = argparse.ArgumentParser(description="Supramolecular coupling calculation")
    parser.add_argument("--dimer_filename", type=str, help="Dimer log file")
    parser.add_argument(
        "--excited_states", type=int, nargs="+", help="Excited states to consider"
    )
    parser.add_argument("--donor_filename", type=str, help="Donor log file")
    parser.add_argument("--acceptor_filename", type=str, help="Acceptor log file")
    parser.add_argument(
        "--max_states", type=int, default=10, required=False, help="Maximum nstates=N"
    )
    args = parser.parse_args()

    acceptor_file = args.acceptor_filename
    donor_file = args.donor_filename
    dimer_file = args.dimer_filename
    excited_states = args.excited_states
    max_states = args.max_states

    dimer_tdm_1 = extract_tdm_xyz_values(
        dimer_file, excited_state=excited_states[0], max_states=max_states
    )
    e1 = extract_vertical_excitation_energy(dimer_file, excited_state=excited_states[0])
    dimer_tdm_2 = extract_tdm_xyz_values(
        dimer_file, excited_state=excited_states[1], max_states=max_states
    )
    e2 = extract_vertical_excitation_energy(dimer_file, excited_state=excited_states[1])
    acceptor_tdm = extract_tdm_xyz_values(
        acceptor_file, excited_state=1, max_states=max_states
    )
    donor_tdm = extract_tdm_xyz_values(
        donor_file, excited_state=1, max_states=max_states
    )

    if (
        isinstance(dimer_tdm_1, str)
        or isinstance(dimer_tdm_2, str)
        or isinstance(acceptor_tdm, str)
        or isinstance(donor_tdm, str)
    ):
        print("Error occurred while extracting TDM values.")
        return

    if e1 is None or e2 is None:
        print("Error occurred while extracting vertical excitation energies.")
        return

    print(f"Dimer TDM-1: {dimer_tdm_1}, Dimer E1: {e1 * HA_2_EV :.4f} eV")
    print(f"Dimer TDM-2: {dimer_tdm_2}, Dimer E2: {e2 * HA_2_EV :.4f} eV")
    print(f"Acceptor TDM: {acceptor_tdm},\nDonor TDM: {donor_tdm}")

    coupling = diabatize(dimer_tdm_1, dimer_tdm_2, acceptor_tdm, donor_tdm, e1, e2)
    print(f"The Coupling From Diabatization is: {coupling * HA_2_EV:.4f} eV")


if __name__ == "__main__":
    main()
