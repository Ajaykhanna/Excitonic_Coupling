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
import pandas as pd
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


def read_xyz(file):
    """
    Reads an XYZ file and returns a pandas DataFrame containing the atom data.

    Args:
        file (str): The path to the XYZ file to be read.

    Returns:
        pd.DataFrame: A DataFrame with columns 'Element', 'X', 'Y', and 'Z' containing the atom data.
    """
    with open(file, "r") as f:
        lines = f.readlines()
        num_atoms = int(lines[0].strip())
        atom_data = lines[2 : 2 + num_atoms]

    atoms = []
    for line in atom_data:
        parts = line.split()
        atoms.append([parts[0], float(parts[1]), float(parts[2]), float(parts[3])])

    return pd.DataFrame(atoms, columns=["Element", "X", "Y", "Z"])


def compute_geometric_center(df):
    """
    Computes the geometric center of the provided DataFrame `df` by taking the mean of the "X", "Y", and "Z" columns.

    Args:
        df (pandas.DataFrame): The DataFrame containing the "X", "Y", and "Z" columns.

    Returns:
        numpy.ndarray: The 3D coordinates of the geometric center.
    """
    return df[["X", "Y", "Z"]].mean().values


def translate_molecule_to_distance(
    df, geo_center_target, geo_center_current, new_distance
):
    """
    Translates the coordinates of a molecule to a new distance from a target geometric center.

    Args:
        df (pandas.DataFrame): A DataFrame containing the molecular coordinates.
        geo_center_target (numpy.ndarray): The target geometric center of the molecule.
        geo_center_current (numpy.ndarray): The current geometric center of the molecule.
        new_distance (float): The new desired distance from the target geometric center.

    Returns:
        pandas.DataFrame: The DataFrame with the translated coordinates.
    """
    # Calculate the direction vector
    direction_vector = geo_center_current - geo_center_target

    # Normalize the direction vector
    norm_direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Scale the vector to the new desired distance
    scaled_vector = norm_direction_vector * new_distance

    # Calculate the new geometric center for NBD
    new_geo_center = geo_center_target + scaled_vector

    # Calculate the translation vector
    translation_vector = new_geo_center - geo_center_current

    # Translate the coordinates
    df[["X", "Y", "Z"]] += translation_vector

    return df


def save_xyz(df, filename):
    """
    Saves a DataFrame containing molecular coordinates (X, Y, Z) to a file.

    Args:
        df (pandas.DataFrame): A DataFrame containing the molecular coordinates, with columns 'Element', 'X', 'Y', and 'Z'.
        filename (str): The path to the output file.

    Returns:
        None
    """
    with open(filename, "w") as f:
        f.write(f"{len(df)}\n")
        f.write("Translated molecule\n")
        for index, row in df.iterrows():
            f.write(f"{row['Element']} {row['X']:.6f} {row['Y']:.6f} {row['Z']:.6f}\n")


def main():
    banner("Relative Translation of NBD and NR Molecules")
    parser = argparse.ArgumentParser(
        description="Translate NBD Relative to Geometric Center of NR"
    )
    parser.add_argument(
        "--dye_1_filename", type=str, required=True, help="Filename of dye NBD log file"
    )
    parser.add_argument(
        "--dye_2_filename", type=str, required=True, help="Filename of dye NR log file"
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=10.0,
        help="Maximum distance to translate NBD",
    )
    parser.add_argument(
        "--step_size", type=float, default=0.5, help="Step size for translation"
    )
    parser.add_argument(
        "--output", type=str, default="molecule", help="Output filename"
    )
    args = parser.parse_args()

    # Load the XYZ files
    file_nbd = args.dye_1_filename
    file_nr = args.dye_2_filename

    # Read and process the files
    df_nbd = read_xyz(file_nbd)
    df_nr = read_xyz(file_nr)

    # Compute the geometric centers
    geo_center_nbd = compute_geometric_center(df_nbd)
    geo_center_nr = compute_geometric_center(df_nr)
    current_distance = np.linalg.norm(geo_center_nbd - geo_center_nr)
    print(
        f"Original distance between Geometric Center of NBD and NR: {current_distance:.2f} Angstroms"
    )
    if args.max_distance is not None:
        new_COMD_COMA_distance = np.arange(
            current_distance, args.max_distance + args.step_size, args.step_size
        )
    else:
        new_COMD_COMA_distance = np.arange(current_distance, 10.5, 0.5)

    for idx, distance in enumerate(new_COMD_COMA_distance):
        # Translate the NBD molecule
        df_nbd_translated = translate_molecule_to_distance(
            df_nbd, geo_center_nr, geo_center_nbd, distance
        )

        # Save the translated NBD coordinates
        translated_file_path = f"./output/{args.output}_{distance:.1f}translated.xyz"
        save_xyz(df_nbd_translated, translated_file_path)
        print(f"Translated NBD coordinates to {distance:.1f} Angstroms")


if __name__ == "__main__":
    main()
