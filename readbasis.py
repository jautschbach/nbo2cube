
import numpy as np
import math 
import time
import re
import copy
from itertools import groupby
from scipy.constants import physical_constants
import os


##################################################################################################################

def parse_file47(filename):
    print(f"Parsing {filename} as a .47 file")

    def read_file47(filename):
        try:
            with open(filename, 'r') as file:
                return file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filename}' not found.")
        except IOError:
            raise IOError(f"Error reading file '{filename}'.")

    file_content = read_file47(filename)

    def parse_array_from_block(varname, content, dtype=float):
        pattern = re.compile(rf'{varname}\s+=\s+((?:[-+]?\d+\.\d+(?:E[+-]?\d+)?\s+)+)')
        matches = pattern.findall(content)
        values = []
        for match in matches:
            parts = match.split()
            converted = [dtype(v) for v in parts]
            values.extend(converted)
        return values

    def parse_int_array(varname, content):
        pattern = re.compile(rf'{varname}\s+=\s+([\d\s]+)')
        matches = pattern.findall(content)
        values = []
        for match in matches:
            parts = match.split()
            converted = [int(v) for v in parts]
            values.extend(converted)
        return values

    def is_ungrouped(ncomp):
        return all(x == 1 for x in ncomp)

    def process_orbital_labels(label, ncomp, orb_mapping):
        if not sum(ncomp) == len(label):
            raise ValueError(f"NCOMP sum ({sum(ncomp)}) does not match LABEL length ({len(label)})")
        
        orb_type = [orb_mapping.get(l, ('unknown', 'unknown'))[0] for l in label]
        orb_val = [orb_mapping.get(l, ('unknown', 'unknown'))[1] for l in label]
        shell_num = []

        # Primary shell assignment based on NCOMP
        idx = 0
        for shell_idx, nc in enumerate(ncomp, 1):
            group = orb_type[idx:idx + nc]
            if nc > 1:
                # Validate that orbitals in the group share the same base type
                base_type = group[0][0] if group else None
                if not all(t[0] == base_type for t in group) or len(group) != nc:
                    raise ValueError(f"Invalid NCOMP grouping at shell {shell_idx}: {group} does not match NCOMP={nc}")
            for _ in range(nc):
                shell_num.append(shell_idx)
            idx += nc

        if is_ungrouped(ncomp):
            shell_num = list(range(1, len(label) + 1))
        else:
            # Secondary validation with type-based grouping
            type_limits = {'p': 3, 'd': 5, 'f': 7, 'g': 9, 'h': 11, 'i': 13, 'j': 15}
            temp_shell_num = []
            count = 1
            orb_count = 0
            for i, t in enumerate(orb_type):
                base_type = t[0]
                if i > 0 and base_type in type_limits and orb_type[i-1][0] == base_type:
                    if orb_count < type_limits[base_type]:
                        temp_shell_num.append(temp_shell_num[-1])
                        orb_count += 1
                    else:
                        count += 1
                        temp_shell_num.append(count)
                        orb_count = 1
                else:
                    count += 1
                    temp_shell_num.append(count)
                    orb_count = 1

            # Cross-validate NCOMP-based and type-based shell assignments
            ncomp_shells = [len(list(g)) for k, g in groupby(shell_num)]
            type_shells = [len(list(g)) for k, g in groupby(temp_shell_num)]
            if ncomp_shells != type_shells:
                print(f"Warning: NCOMP-based shells {ncomp_shells} differ from type-based shells {type_shells}. Using NCOMP-based.")

        return orb_type, orb_val, shell_num

    def system_info(content):
        bohr_to_ang = physical_constants['Bohr radius'][0] * 1e10  # Bohr to Ångström
        use_bohr = "BOHR" in content.upper()
        to_bohr = 1 if use_bohr else bohr_to_ang

        # Parse atom coordinates: (atomic_number, charge, x, y, z)
        coord_pattern = r'\s+(\d+)\s+(\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)'
        atom_matches = re.findall(coord_pattern, content)
        atom_data = [
            (int(z), int(chg), float(x), float(y), float(z_))
            for z, chg, x, y, z_ in atom_matches
        ]

        # Parse basis data
        variable_names_float = ['EXP', 'CS', 'CP', 'CD', 'CF', 'CG', 'CH', 'CI', 'CJ']
        variable_names_int = ['CENTER', 'LABEL', 'NSHELL', 'NEXP', 'NCOMP', 'NPRIM', 'NPTR']
        
        parsed_float = {var: parse_array_from_block(var, content, float) for var in variable_names_float}
        parsed_int = {var: parse_int_array(var, content) for var in variable_names_int}

        EXP = parsed_float['EXP']
        CS, CP, CD, CF, CG, CH, CI, CJ = [parsed_float.get(v, []) for v in variable_names_float[1:]]
        CENTER = parsed_int['CENTER']
        LABEL = parsed_int['LABEL']
        NCOMP = parsed_int['NCOMP']
        NPRIM = parsed_int['NPRIM']
        NPTR = parsed_int['NPTR']

        # Orbital mapping
        orb_mapping = {
            1: ('s', 's'), 51: ('s', 's'), 101: ('px', 'px'), 102: ('py', 'py'), 103: ('pz', 'pz'),
            151: ('px', 'px'), 152: ('py', 'py'), 153: ('pz', 'pz'),
            251: ('d_xy', 'ds2'), 252: ('d_xz', 'ds1'), 253: ('d_yz', 'dc1'),
            254: ('d_x2-y2', 'dc2'), 255: ('d_z2', 'd0'),
            351: ('fz(5z2-3r2)', 'f0'), 352: ('fx(5z2-r2)', 'fc1'), 353: ('fy(5z2-r2)', 'fs1'),
            354: ('fz(x2-y2)', 'fc2'), 355: ('fxyz', 'fs2'), 356: ('fx(x2-3y2)', 'fc3'),
            357: ('f(3x2-y2)', 'fs3'),
            451: ('g0', 'g0'), 452: ('gc1', 'gc1'), 453: ('gs1', 'gs1'), 454: ('gc2', 'gc2'),
            455: ('gs2', 'gs2'), 456: ('gc3', 'gc3'), 457: ('gs3', 'gs3'), 458: ('gc4', 'gc4'),
            459: ('gs4', 'gs4'),
            551: ('h0', 'h0'), 552: ('hc1', 'hc1'), 553: ('hs1', 'hs1'), 554: ('hc2', 'hc2'),
            555: ('hs2', 'hs2'), 556: ('hc3', 'hc3'), 557: ('hs3', 'hs3'), 558: ('hc4', 'hc4'),
            559: ('hs4', 'hs4'), 560: ('hc5', 'hc5'), 561: ('hs5', 'hs5'),
            651: ('i0', 'i0'), 652: ('ic1', 'ic1'), 653: ('is1', 'is1'), 654: ('ic2', 'ic2'),
            655: ('is2', 'is2'), 656: ('ic3', 'ic3'), 657: ('is3', 'is3'), 658: ('ic4', 'ic4'),
            659: ('is4', 'is4'), 660: ('ic5', 'ic5'), 661: ('is5', 'is5'), 662: ('ic6', 'ic6'),
            663: ('is6', 'is6'),
            751: ('j0', 'j0'), 752: ('jc1', 'jc1'), 753: ('js1', 'js1'), 754: ('jc2', 'jc2'),
            755: ('js2', 'js2'), 756: ('jc3', 'jc3'), 757: ('js3', 'js3'), 758: ('jc4', 'jc4'),
            759: ('js4', 'js4'), 760: ('jc5', 'jc5'), 761: ('js5', 'js5'), 762: ('jc6', 'jc6'),
            763: ('js6', 'js6'), 764: ('jc7', 'jc7'), 765: ('js7', 'js7')
        }

        orb_type, orb_val, shell_num = process_orbital_labels(LABEL, NCOMP, orb_mapping)

        # Map NPRIM and NPTR to basis functions based on shells
        if not is_ungrouped(NCOMP):
            if len(NPRIM) != len(NCOMP) or len(NPTR) != len(NCOMP):
                raise ValueError(f"NPRIM ({len(NPRIM)}) or NPTR ({len(NPTR)}) length does not match NCOMP ({len(NCOMP)})")
            # Replicate NPRIM and NPTR for each basis function in a shell
            NPRIM_expanded = []
            NPTR_expanded = []
            for shell_idx, nc in enumerate(NCOMP):
                NPRIM_expanded.extend([NPRIM[shell_idx]] * nc)
                NPTR_expanded.extend([NPTR[shell_idx]] * nc)
        else:
            NPRIM_expanded = NPRIM
            NPTR_expanded = NPTR

        # Validate lengths
        if len(NPRIM_expanded) != len(LABEL) or len(NPTR_expanded) != len(LABEL):
            raise ValueError(f"Expanded NPRIM ({len(NPRIM_expanded)}) or NPTR ({len(NPTR_expanded)}) does not match LABEL ({len(LABEL)})")

        # Build basis info dictionary
        bas_info_dict = []
        for i in range(len(LABEL)):
            atom_idx = CENTER[i] - 1
            prim = NPRIM_expanded[i]
            ptr = NPTR_expanded[i]

            info = {
                "N": i + 1,
                "CENTER": CENTER[i],
                "LABEL": LABEL[i],
                "shell_num": shell_num[i],
                "type": orb_type[i],
                "orb_val": orb_val[i],
                "exps": EXP[ptr - 1: ptr - 1 + prim]
            }

            coeffs = []
            for coeff_array in [CS, CP, CD, CF, CG, CH, CI, CJ]:
                slice_ = coeff_array[ptr - 1: ptr - 1 + prim]
                coeffs.extend([c for c in slice_ if c != 0.0])
            info["coeffs"] = coeffs

            atom_coord = atom_data[atom_idx][2:5]  # x, y, z
            info["xcenter"] = atom_coord[0] / to_bohr
            info["ycenter"] = atom_coord[1] / to_bohr
            info["zcenter"] = atom_coord[2] / to_bohr

            bas_info_dict.append(info)

        # Convert atom coordinates to Ångström for output
        atom_data_ang = [
            (z, charge, x * bohr_to_ang, y * bohr_to_ang, z_ * bohr_to_ang)
            if use_bohr else (z, charge, x, y, z_)
            for (z, charge, x, y, z_) in atom_data
        ]

        return bas_info_dict, atom_data_ang, to_bohr

    basis_info_dict, atom_data, to_bohr = system_info(file_content)

    # Prepare output
    coordinates = [atom[2:] for atom in atom_data]
    atom_info = [(atom[0],) + tuple(atom[2:]) for atom in atom_data]

    return basis_info_dict, coordinates, atom_info, to_bohr



def parse_file31(filename):
    print(f"Parsing {filename} as a .31 file")
    """
    In file .31 All coordinates are in Angstrom.
    """
    with open(filename, 'r') as file:
        
        to_bohr = physical_constants['Bohr radius'][0] * 1e10
        content = file.readlines()
        line_4 = content[3]
        line_4_ele = line_4.split()
        num_atom = int(line_4_ele[0])
        num_shell = int(line_4_ele[1])
        num_exps = int(line_4_ele[2])
        
        dash_line_num = []
        dashes = "-------------------------------"
        for i, line in enumerate(content, 1):
            if dashes in line:
                dash_line_num.append(i)

        coord_line = dash_line_num[1] + 1
        coordinates = [
            [int(line.split()[0]), float(line.split()[1]), float(line.split()[2]), float(line.split()[3])]
            for line in content[coord_line-1:coord_line + num_atom -1]
        ]
        coordinates = np.array(coordinates)
        
        exp_and_coeff = []
        exp_and_coeff_line = dash_line_num[3]
        for line in content[exp_and_coeff_line:]:
            exp_and_coeff.extend(line.split())
        dimen = int(len(exp_and_coeff) / num_exps)
        exp_and_coeff = np.array(exp_and_coeff).reshape(dimen, num_exps)
        
        variables = ['EXP', 'CS', 'CP', 'CD', 'CF', 'CG', 'CH', 'CI', 'CJ', 'CK']
        local_vars = {}
        for i, variable in enumerate(variables):
            if i < len(exp_and_coeff):
                local_vars[variable] = exp_and_coeff[i]
            else:
                local_vars[variable] = None
        
        label_sect_line = coord_line + num_atom + 1
        label_sect_end_line = exp_and_coeff_line - 1
        lines = content[label_sect_line - 1:label_sect_end_line]
        
        prim_ptr_list = []
        label_list = []
        modified_lines = []
        i = 0
        while i < len(lines):
            current_line_items = lines[i].split()
            if len(current_line_items) > 4:
                if i > 0 and len(lines[i-1].split()) >= 2:
                    prev_line_second_item = int(lines[i-1].split()[1])
                    if len(current_line_items) != prev_line_second_item:
                        if i < len(lines) - 1:
                            current_line_items.extend(lines[i+1].split())
                            i += 1
            modified_lines.append(' '.join(current_line_items))
            i += 1
        
        lines = modified_lines
        for i, line in enumerate(lines, start=1):
            if i % 2 == 1:            
                prim_ptr_list.append([int(element) for element in line.split()])
            if i % 2 == 0:
                label_list.append([int(element) for element in line.split()])

        orb_mapping = {
            1: ('s', 's'), 51: ('s', 's'),
            101: ('px', 'px'), 102: ('py', 'py'), 103: ('pz', 'pz'),
            151: ('px', 'px'), 152: ('py', 'py'), 153: ('pz', 'pz'),
            251: ('d_xy', 'ds2'), 252: ('d_xz', 'ds1'), 253: ('d_yz', 'dc1'), 254: ('d_x2-y2', 'dc2'), 255: ('d_z2', 'd0'),
            351: ('fz(5z2-3r2)', 'f0'), 352: ('fx(5z2-r2)', 'fc1'), 353: ('fy(5z2-r2)', 'fs1'), 354: ('fz(x2-y2)', 'fc2'),
            355: ('fxyz', 'fs2'), 356: ('fx(x2-3y2)', 'fc3'), 357: ('f(3x2-y2)', 'fs3'),
            451: ('g0', 'g0'), 452: ('gc1', 'gc1'), 453: ('gs1', 'gs1'), 454: ('gc2', 'gc2'), 455: ('gs2', 'gs2'),
            456: ('gc3', 'gc3'), 457: ('gs3', 'gs3'), 458: ('gc4', 'gc4'), 459: ('gs4', 'gs4'),
            551: ('h0', 'h0'), 552: ('hc1', 'hc1'), 553: ('hs1', 'hs1'), 554: ('hc2', 'hc2'), 555: ('hs2', 'hs2'),
            556: ('hc3', 'hc3'), 557: ('hs3', 'hs3'), 558: ('hc4', 'hc4'), 559: ('hs4', 'hs4'), 560: ('hc5', 'hc5'), 561: ('hs5', 'hs5'),
            651: ('i0', 'i0'), 652: ('ic1', 'ic1'), 653: ('is1', 'is1'), 654: ('ic2', 'ic2'), 655: ('is2', 'is2'),
            656: ('ic3', 'ic3'), 657: ('is3', 'is3'), 658: ('ic4', 'ic4'), 659: ('is4', 'is4'), 660: ('ic5', 'ic5'), 661: ('is5', 'is5'),
            662: ('ic6', 'ic6'), 663: ('is6', 'is6'),
            751: ('j0', 'j0'), 752: ('jc1', 'jc1'), 753: ('js1', 'js1'), 754: ('jc2', 'jc2'), 755: ('js2', 'js2'),
            756: ('jc3', 'jc3'), 757: ('js3', 'js3'), 758: ('jc4', 'jc4'), 759: ('js4', 'js4'), 760: ('jc5', 'jc5'),
            761: ('js5', 'js5'), 762: ('jc6', 'jc6'), 763: ('js6', 'js6'), 764: ('jc7', 'jc7'), 765: ('js7', 'js7')
        }

        orb_type = [[orb_mapping.get(element)[0] for element in sublist] for sublist in label_list]
        orb_val = [[orb_mapping.get(element)[1] for element in sublist] for sublist in label_list]

        shell_num = 1
        N = 1
        basis_info_dict = []
        coeff_mapping = {
            1: local_vars['CS'],
            3: local_vars['CP'],
            5: local_vars['CD'],
            7: local_vars['CF'],
            9: local_vars['CG'],
            11: local_vars['CH'],
            13: local_vars['CI'],
            15: local_vars['CJ']
        }
        
        for p_p_l, orb in zip(prim_ptr_list, label_list):
            for i in range(p_p_l[1]):
                coeff = coeff_mapping[len(orb)]
                # Add center coordinates (convert to to_bohr)
                atom_coords = coordinates[p_p_l[0] - 1][1:4]  # [x, y, z] in Angstroms
                x = atom_coords[0] / to_bohr
                y = atom_coords[1] / to_bohr
                z = atom_coords[2] / to_bohr
        
                info = {
                    'N': N,
                    "CENTER": p_p_l[0],
                    "shell_num": shell_num,
                    "LABEL": orb[i] if i < len(orb) else None,
                    "exps": [float(e) for e in local_vars['EXP'][p_p_l[2] - 1 : p_p_l[2] + p_p_l[3] - 1]],
                    "coeffs": [(float(c)) for c in coeff[p_p_l[2] - 1 : p_p_l[2] + p_p_l[3] - 1]],
                    "xcenter": x,
                    "ycenter": y,
                    "zcenter": z
                }
                N += 1
                basis_info_dict.append(info)
            shell_num += 1
                
        
        for info in basis_info_dict:
            type_value = info["LABEL"]
            orb_type_val, orb_val_val = orb_mapping[type_value]
            info["type"] = orb_type_val
            info["orb_val"] = orb_val_val

    
    atom_data = {
        'coordinates': [tuple(coord[1:]) for coord in coordinates],
        'atom_charge': [tuple(coord[0:1]) for coord in coordinates],
        'atom_info': [tuple(coord[:]) for coord in coordinates]
    }
    
    coordinates = atom_data.get("coordinates")
    atom_info = atom_data.get("atom_info")
    
    """
    Coordinates xcenter, ycenter, and zcenter will be in bohr for the overlap matrix
    integral code.
    
    Coordinates in atom_info and coordinates will be in angstrom. Necessary conversions
    are already done in NBO2CUBE. 
    """
    ################################################
            
    return basis_info_dict, coordinates, atom_info, to_bohr


while True:
    filename = input("Enter the filename (.47 recommended or .31): ").strip()
    if os.path.exists(filename):
        break
    else:
        print(f"The file {filename} does not exist. Please try again.")


start = time.perf_counter()

_, file_ext = os.path.splitext(filename)

if file_ext == ".47":
    basis_info_dict, coordinates, atom_info, to_bohr = parse_file47(filename)
elif file_ext == ".31":
    basis_info_dict, coordinates, atom_info, to_bohr = parse_file31(filename)
else:
    print(f"Unsupported file extension: {file_ext}")


#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################


import overlap_matrix
from bas_dict import dict_keys
from bas_dict import get_term_info

getSmat = overlap_matrix.get_overlap_matrix


# ----------------------------
# Helper Functions
# ----------------------------

def double_factorial(n):
    if n <= 0:
        return 1
    else:
        return n * double_factorial(n - 2)

def gaussian_norm(alpha, l, m, n):
    """
    Computes normalization constant for a primitive Cartesian Gaussian.
    """
    lmn = l + m + n
    prefactor = (2 ** (2 * lmn + 1.5)) * (alpha ** (lmn + 1.5)) / (math.pi ** 1.5)
    denom = double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1)
    return math.sqrt(prefactor / denom)



# Filter S and P orbitals for convention check
sp_basis = [f for f in basis_info_dict if f['orb_val'] in ['s', 'px', 'py', 'pz']]

# Compute diagonal overlaps for S and P orbitals
S_diag_no_norm = np.diag(getSmat(sp_basis, dict_keys, normalize_primitives=False, diagonal_only=True))
S_diag_with_norm = np.diag(getSmat(sp_basis, dict_keys, normalize_primitives=True, diagonal_only=True))


# Check normalization
def is_normalized(S_diag, tol=1e-5):
    """
    Checks if all diagonal overlap values are within tolerance of 1.0.
    
    Parameters:
        S_diag (list): List of diagonal overlap values.
        tol (float): Tolerance for normalization check.
    
    Returns:
        bool: True if all S_ii are within tol of 1.0, False otherwise.
    """
    return all(abs(sii - 1.0) < tol for sii in S_diag)

# Convert to ORCA/Molden convention
def convert_to_molden(full_basis):
    """
    Converts a basis set from Gaussian convention to ORCA/Molden convention
    by multiplying each coefficient by the normalization constant of the primitive.
    
    Parameters:
        full_basis (list): List of basis function dictionaries (all orbitals).
    
    Returns:
        list: New list with modified coefficients (ORCA/Molden convention).
    """
    import copy
    new_basis = copy.deepcopy(full_basis)  # Avoid modifying the input
    
    for basis_func in new_basis:
        ityp = basis_func['orb_val']
        nc, cc, l_m_n = get_term_info(ityp)
        
        # Use first angular momentum component for normalization
        x, y, z = l_m_n[0]
        
        for iprim in range(len(basis_func['exps'])):
            alpha = basis_func['exps'][iprim]
            norm = gaussian_norm(alpha, x, y, z)
            basis_func['coeffs'][iprim] *= norm  # Multiply coefficient by normalization constant
    
    return new_basis

# Step 5: Normalize coefficients by square root of self-overlap
def normalize_by_self_overlap(full_basis):
    """
    Normalizes each basis function by dividing its coefficients by the square root
    of its self-overlap S_ii, ensuring S_ii = 1 in ORCA/Molden convention.
    
    Parameters:
        full_basis (list): List of basis function dictionaries.
    
    Returns:
        list: New list with coefficients normalized by sqrt(S_ii).
    """
    import copy
    import math
    new_basis = copy.deepcopy(full_basis)  # Avoid modifying the input
    
    # Compute diagonal overlaps in ORCA/Molden convention (no extra normalization)
    S_diag = np.diag(getSmat(new_basis, dict_keys, normalize_primitives=False, diagonal_only=True))
    
    for i, basis_func in enumerate(new_basis):
        if S_diag[i] <= 0:
            print(f"Warning: Non-positive self-overlap S_ii = {S_diag[i]} for basis function {i}. Skipping normalization.")
            continue
        sqrt_sii = math.sqrt(S_diag[i])
        for iprim in range(len(basis_func['exps'])):
            basis_func['coeffs'][iprim] /= sqrt_sii  # Divide coefficient by sqrt(S_ii)
    
    return new_basis

# Determine convention and perform conversions
if is_normalized(S_diag_with_norm) and not is_normalized(S_diag_no_norm):
    print("Basis set is in Gaussian convention (coefficients do NOT include normalization).")
    print("Converting all basis functions to ORCA/Molden convention...")
    basis_info_dict = convert_to_molden(basis_info_dict)
    print("Conversion to ORCA/Molden convention complete.")
    print("Normalizing coefficients by square root of self-overlap...")
    basis_info_dict = normalize_by_self_overlap(basis_info_dict)
    print("Final normalization complete. All basis functions now have S_ii = 1.")
elif is_normalized(S_diag_no_norm):
    print("Basis set is in ORCA/Molden convention (coefficients include normalization).")
    print("Normalizing coefficients by square root of self-overlap...")
    basis_info_dict = normalize_by_self_overlap(basis_info_dict)
    print("Final normalization complete. All basis functions now have S_ii = 1.")
else:
    print("Basis set is not normalized in either convention. Assuming Gaussian convention.")
    print("Converting all basis functions to ORCA/Molden convention...")
    basis_info_dict = convert_to_molden(basis_info_dict)
    print("Conversion to ORCA/Molden convention complete.")
    print("Normalizing coefficients by square root of self-overlap...")
    basis_info_dict = normalize_by_self_overlap(basis_info_dict)
    print("Final normalization complete. All basis functions now have S_ii = 1.")
    print('\n--------------------------------------------------------')




Smat = getSmat(basis_info_dict, dict_keys, normalize_primitives=False, diagonal_only=False)

def normalize_basis_info(prev_basis_info_dict, ovlp_mat):
    self_overlap = np.diag(ovlp_mat)
    scaling_factors = 1.0 / np.sqrt(self_overlap)
    new_basis_info_dict = copy.deepcopy(prev_basis_info_dict)
    for idx, basis in enumerate(new_basis_info_dict):
        basis['coeffs'] = list(np.array(basis['coeffs']) * scaling_factors[idx])
    return new_basis_info_dict

norm_basis_info = normalize_basis_info(basis_info_dict, Smat)


#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################

def normalize_basis_info(prev_basis_info_dict, ovlp_mat):
    self_overlap = np.diag(ovlp_mat)
    scaling_factors = 1.0 / np.sqrt(self_overlap)
    new_basis_info_dict = copy.deepcopy(prev_basis_info_dict)
    for idx, basis in enumerate(new_basis_info_dict):
        basis['coeffs'] = list(np.array(basis['coeffs']) * scaling_factors[idx])
    return new_basis_info_dict

def extract_floats_numpy(content, start_index, count):
    return np.fromstring(content[start_index:], sep=' ', count=count)

def create_symmetric_matrix_vectorized(lower_triangular, n):
    matrix = np.zeros((n, n))
    matrix[np.tril_indices(n)] = lower_triangular
    return matrix + matrix.T - np.diag(matrix.diagonal())

def process_47_file(file_path, nbas):
    with open(file_path, 'r') as file:
        content = file.read()
    lines = content.split('\n')
    is_open_shell = 'OPEN' in lines[0].upper()
    has_upper = 'UPPER' in lines[0].upper()
    if has_upper:
        float_count = int(nbas * (nbas + 1) / 2)
    else:
        float_count = int(nbas * nbas)
    keywords = ['$OVERLAP', '$DENSITY']
    keyword_dict = {}
    for keyword in keywords:
        start_index = content.index(keyword) + len(keyword)
        if keyword in ['$DENSITY', '$FOCK'] and is_open_shell:
            if has_upper:
                all_arr = extract_floats_numpy(content, start_index, 2 * float_count)
                alpha_triangular = all_arr[:float_count]
                beta_triangular = all_arr[-float_count:]
                alpha_matrix = create_symmetric_matrix_vectorized(alpha_triangular, nbas)
                beta_matrix = create_symmetric_matrix_vectorized(beta_triangular, nbas)
            else:
                all_arr = extract_floats_numpy(content, start_index, 2 * float_count)
                alpha_matrix = all_arr[:float_count].reshape(nbas, nbas)
                beta_matrix = all_arr[-float_count:].reshape(nbas, nbas)
            keyword_dict[f"{keyword[1:]}_ALPHA"] = alpha_matrix
            keyword_dict[f"{keyword[1:]}_BETA"] = beta_matrix
        else:
            if has_upper:
                lower_triangular = extract_floats_numpy(content, start_index, float_count)
                symmetric_matrix = create_symmetric_matrix_vectorized(lower_triangular, nbas)
            else:
                full_matrix = extract_floats_numpy(content, start_index, float_count).reshape(nbas, nbas)
                symmetric_matrix = full_matrix
            keyword_dict[keyword[1:] if keyword.startswith('$') else keyword] = symmetric_matrix
    return is_open_shell, keyword_dict

nbf = len(basis_info_dict)

#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################



def calculate_overlap_matrix(primit_info_dict, nbo_overlap_mat):
    
    S = getSmat(primit_info_dict, dict_keys, normalize_primitives=False, diagonal_only=False)
    
    n = S.shape[0]
    S_round = np.round(S, 8)
    nbo_round = np.round(nbo_overlap_mat, 8)
    abs_close = np.isclose(np.abs(S_round), np.abs(nbo_round), atol=1e-5)
    sign_diff = np.sign(S_round) != np.sign(nbo_round)
    value_close = np.isclose(S_round, nbo_round, atol=1e-5)
    sign_change_mask = abs_close & sign_diff
    mismatch_mask = ~value_close & ~sign_change_mask
    lower_tri_mask = np.tril_indices(n)
    sign_change_count = np.sum(sign_change_mask[lower_tri_mask])
    mismatch_count = np.sum(mismatch_mask[lower_tri_mask])
    diag_idx = np.diag_indices(n)
    diag_mismatch_mask = mismatch_mask[diag_idx]
    diag_ratios = S_round[diag_idx] / nbo_round[diag_idx]
    self_overlap_mismatches = {i+1: diag_ratios[i] for i in np.where(diag_mismatch_mask)[0]}
    return mismatch_count, sign_change_count, self_overlap_mismatches, S


def modify_basis_info(primit_info_dict, self_overlap_mismatches):
    scaling_factors = np.ones(len(primit_info_dict))
    for basis_idx, ratio in self_overlap_mismatches.items():
        scaling_factors[basis_idx-1] = 1.0 / np.sqrt(ratio)
    new_primit_info_dict = []
    for idx, basis_info in enumerate(primit_info_dict):
        new_basis_info = basis_info.copy()
        coeffs = np.array(new_basis_info['coeffs'])
        new_basis_info['coeffs'] = list(coeffs * scaling_factors[idx])
        new_primit_info_dict.append(new_basis_info)
    return new_primit_info_dict






def iterative_basis_modification(initial_primit_info_dict, nbo_overlap_mat, max_iterations=5):
    primit_info_dict = initial_primit_info_dict
    prev_metrics = []
    for iteration in range(max_iterations):
        mismatch_count, sign_change_count, self_overlap_mismatches, Smat = \
            calculate_overlap_matrix(primit_info_dict, nbo_overlap_mat)
        print(f"\nIteration {iteration + 1}")
        print(f"Total mismatches: {mismatch_count}")
        print(f"Total sign changes: {sign_change_count}")
        prev_metrics.append((mismatch_count, sign_change_count))
        # Check for stagnation in the last two iterations (after at least 2)
        if len(prev_metrics) > 2:
            if prev_metrics[-1] == prev_metrics[-2] == prev_metrics[-3]:
                print("\nNo change in mismatches/sign changes over last three iterations. Exiting early.")
                break
        if mismatch_count == 0 or iteration == max_iterations - 1:
            print("\nFinal Results:")
            print(f"Total mismatches: {mismatch_count}")
            print(f"Total sign changes: {sign_change_count}")
            if self_overlap_mismatches:
                print("\nRemaining self-overlap mismatches (i : calculated/NBO ratio):")
                for i, ratio in self_overlap_mismatches.items():
                    print(f"{i} : {ratio:.8f}")
            else:
                print("\nNo remaining self-overlap mismatches.")
            return primit_info_dict, Smat
        primit_info_dict = modify_basis_info(primit_info_dict, self_overlap_mismatches)
    print("\nMaximum iterations reached or exited early due to stagnation.")
    return primit_info_dict, Smat


def print_final_overlap(S, nbo_overlap_mat):
    n = S.shape[0]
    print("\nFinal Overlap Matrix Comparison:")
    print("Pos   i,j,Sij      Calculated      NBO")
    print("-" * 55)
    ii, jj = np.tril_indices(n)
    cal_vals = np.round(S[ii, jj], 8)
    nbo_vals = np.round(nbo_overlap_mat[ii, jj], 8)
    abs_close = np.isclose(np.abs(cal_vals), np.abs(nbo_vals), atol=1e-7)
    sign_diff = np.sign(cal_vals) != np.sign(nbo_vals)
    value_close = np.isclose(cal_vals, nbo_vals, atol=1e-7)
    warnings_arr = np.full(len(cal_vals), "", dtype=object)
    warnings_arr[abs_close & sign_diff] = "SIGN CHANGE!"
    warnings_arr[~value_close & ~(abs_close & sign_diff)] = "MISMATCH!"
    diag_mask = (ii == jj)
    mismatch_mask = warnings_arr == "MISMATCH!"
    self_overlap_mismatches = {}
    for idx in np.where(diag_mask & mismatch_mask)[0]:
        i = ii[idx] + 1
        ratio = cal_vals[idx] / nbo_vals[idx] if nbo_vals[idx] != 0 else float('inf')
        self_overlap_mismatches[i] = ratio
    positions = np.arange(1, len(cal_vals) + 1)
    for pos, i0, j0, cal, nbo, warning in zip(positions, ii, jj, cal_vals, nbo_vals, warnings_arr):
        print(f"{pos:4d}  i,j,Sij : {i0+1:2d} {j0+1:2d}  {cal:12.8f}  {nbo:12.8f}   {warning}")
    return self_overlap_mismatches


if file_ext == '.31':
    final_norm_basis = norm_basis_info
    
else:
    is_open, matrix_dict = process_47_file(filename, nbf)
    nbo_overlap_mat = matrix_dict.get('OVERLAP')
    # start = time.perf_counter()
    final_norm_basis = norm_basis_info    
    final_norm_basis, final_S = iterative_basis_modification(norm_basis_info, nbo_overlap_mat)
    # end = time.perf_counter()


print('Basis information extracted and renormalized...')
end = time.perf_counter()
print(f"Read Basis runtime: {end - start:.6f} seconds\n")




