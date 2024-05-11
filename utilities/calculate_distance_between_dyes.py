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

This Python script calculates the distance between the geometric centers of two molecules.
The molecules are represented as a N by 3 matrix, where N is the total number of atoms in both molecules.
The matrix is split into two based on the number of atoms in the first molecule.
The geometric center of each molecule is calculated, and then the Euclidean distance between these centers is computed.
"""

import argparse
import numpy as np

def get_geom_center(molecule):
    """
    Calculate the geometric center of a molecule.
    
    Parameters:
    molecule (numpy.ndarray): A N by 3 matrix representing the molecule, where N is the number of atoms.
    
    Returns:
    numpy.ndarray: A 1 by 3 array representing the geometric center of the molecule.
    """
    return np.mean(molecule, axis=0)


def get_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in 3D space.
    
    Parameters:
    point1, point2 (numpy.ndarray): 1 by 3 arrays representing the points.
    
    Returns:
    float: The Euclidean distance between the two points.
    """
    return np.linalg.norm(point1 - point2)


def calculate_distance(matrix, nAtoms_dye1):
    """
    Calculate the distance between the geometric centers of two molecules.
    
    Parameters:
    matrix (numpy.ndarray): A N by 3 matrix representing the two molecules, where N is the total number of atoms.
    nAtoms_dye1 (int): The number of atoms in the first molecule.
    
    Prints:
    The distance between the geometric centers of the two molecules.
    """
    molecule_1 = matrix[:nAtoms_dye1]
    molecule_2 = matrix[nAtoms_dye1:]

    center_1 = get_geom_center(molecule_1)
    center_2 = get_geom_center(molecule_2)
    distance = get_distance(center_1, center_2)

    print(
        f"The distance between the geometric centers of the two molecules is: {distance}"
    )


# Example usage:

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate distance between dyes")
    parser.add_argument("input_file", type=str, help="Input file path")
    parser.add_argument("nAtoms_dyes1", type=int, help="Number of atoms in dye 1")
    parser.add_argument("nAtoms_dyes2", type=int, help="Number of atoms in dye 2")
    args = parser.parse_args()

    input_file = args.input_file
    nAtoms_dyes1 = args.nAtoms_dyes1
    nAtoms_dyes2 = args.nAtoms_dyes2
    total_atoms = nAtoms_dyes1 + nAtoms_dyes2
    coordinates = np.loadtxt(
        input_file, skiprows=1, usecols=(1, 2, 3), max_rows=total_atoms
    )

    matrix = coordinates

    calculate_distance(matrix, nAtoms_dyes1)
