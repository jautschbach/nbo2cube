import numpy as np
import os

from readbasis import final_norm_basis

NBAS = len(final_norm_basis)

def get_cmos(orbital_file):
    try:
        with open(orbital_file, "r") as file:
            lines = file.readlines()

            if len(lines) < 4:
                raise ValueError("File structure is incorrect or file is too short.")

            second_line = lines[1].strip()
            orbital_type = second_line.split()[0]
            print(orbital_type, "in AO basis")

            if "ALPHA" in lines[3].strip():
                print(orbital_file, " is an open-shell system")
                alpha_or_beta = input("Enter A or B to select Alpha or Beta spin: ")

                if alpha_or_beta.lower() == 'a':
                    start_line = 4
                elif alpha_or_beta.lower() == 'b':
                    for i, line in enumerate(lines):
                        if "BETA" in line:
                            start_line = i + 1
                            break
            else:
                print(orbital_file," is a closed shell system...")
                start_line = 3

            words = []
            for line in lines[start_line:]:
                line_elems = line.split()

                for elem in line_elems:
                    try:
                        if len(words) < NBAS * NBAS:
                            float_elem = float(elem)
                            words.append(float_elem)
                    except ValueError:
                        pass

            if len(words) % NBAS != 0:#:
                raise ValueError("Please provide a correct NBO file.\n")

            num_cmos = int(len(words) / NBAS)
            orbital_arr = np.array(words).reshape(NBAS, num_cmos)
            
            return orbital_arr

    except ValueError as e:
        print(e)
        return None

# Main loop to prompt user for multiple file inputs
orbital_files = input("Enter NBO key files separated by commas: ").replace(" ", "").split(",")

# Prompt user for orbital indices once
while True:
    try:
        orbital_input = input("Enter an orbital index (e.g., 1,5,8-12): ")
        orbital_index = []

        items = orbital_input.split(",")

        for item in items:
            if "-" in item:
                start, end = map(int, item.split("-"))
                if start > end:
                    raise ValueError("Start of range must be less than end of range.")
                orbital_index.extend(range(start, end + 1))
            else:
                orbital_index.append(int(item))
        break

    except ValueError as e:
        print(f"Invalid input: {e}. Please try again.")

# Dictionary to store orbital lists for each file
orbital_dict = {}

# Process each file with the same orbital indices
for orbital_file in orbital_files:
    if os.path.exists(orbital_file):
        orbital_arr = get_cmos(orbital_file)
        if orbital_arr is not None:
            orbital_lists = [orbital_arr[i - 1] for i in orbital_index]
            orbital_dict[orbital_file] = orbital_lists
            # print(f"Orbitals for {orbital_file}: {orbital_lists}")
    else:
        print(f"The file {orbital_file} does not exist. Please try again.")

# # Print the dictionary of orbital lists for each file
# print("Dictionary of orbital lists for each file:")
# for file, orbitals in orbital_dict.items():
#     print(f"File: {file}")
#     print("Orbitals:")
#     for orbital in orbitals:
#         print(orbital)
#     print()  # Add a blank line for better readability

# orbital_dict = orbital_dict