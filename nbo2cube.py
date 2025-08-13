    
    
import numpy as np
from multiprocessing import Pool
from functools import partial
import time

from angular_funct import ang_res_lamda
from read_orb_file import orbital_file, items, orbital_index, orbital_lists, orbital_dict
from readbasis import atom_info, coordinates, to_bohr, final_norm_basis
import electron_density_opt_omp
from scipy.constants import physical_constants

data = final_norm_basis

bohr = physical_constants['Bohr radius'][0] * 1e10


#################################################################
"""
All Coordinates at this point should be in Angstrom.
Therefore atominfo and coordinates should be in Angstrom
Conversion will be done Here...
"""

#####################################################







def grid_specification():
    
    coord= (np.array(coordinates))/bohr
 
    min_coords = np.min(coord, axis=0)
    max_coords = np.max(coord, axis=0)
    
    min_x, min_y, min_z = min_coords
    max_x, max_y, max_z = max_coords
    
    def syst_spec(point_choice, ext_dist):
        
        ext_max_coords = max_coords + ext_dist
        ext_min_coords = min_coords - ext_dist
        
        ext_min_x, ext_min_y, ext_min_z = ext_min_coords
        ext_max_x, ext_max_y, ext_max_z = ext_max_coords
        ext_ranges = ext_max_coords - ext_min_coords
       
        
        spread_axis = np.argmax(ext_ranges)
        
        
        point_max = point_choice
        grid_spacing = (ext_max_coords[spread_axis] - ext_min_coords[spread_axis])/(point_max - 1)
        
        point_x = int(round((ext_max_coords[0] - ext_min_coords[0])/grid_spacing)) + 1
        point_y = int(round((ext_max_coords[1] - ext_min_coords[1])/grid_spacing)) + 1
        point_z = int(round((ext_max_coords[2] - ext_min_coords[2])/grid_spacing)) + 1

        # print(ext_max_coords[spread_axis], ext_min_coords[spread_axis], grid_spacing)
        # print(point_x, point_y, point_z)
        origin = np.array([ext_min_x, ext_min_y, ext_min_z])
        # print(origin)
        
        return point_x, point_y, point_z, grid_spacing, origin
    

    def start_end():
        print(f"""
The molecule spans from [{min_x}, {min_y}, {min_z}] to [{max_x}, {max_y}, {max_z}]            
""")    
    
        while True:
            cub_origin = input(f"""
Enter origin of cube e.g {min_x - 3}, {min_y- 4}, {min_z -2}
P.S. They should be lower than [{min_x}, {min_y}, {min_z}]
""")
            cub_origin = cub_origin.split(',')
            cub_origin = [float(i) for i in cub_origin]
            if len(cub_origin) != 3:
                print("Invalid input. Please enter three values.")
            else:
                break
    
        while True:
            cub_end = input(f"""
Enter end of cube e.g {max_x + 3}, {max_y + 4}, {max_z + 2}
They should be higher than [{max_x}, {max_y}, {max_z}]
""")
            cub_end = cub_end.split(',')
            cub_end = [float(i) for i in cub_end]
            if len(cub_end) != 3:
                print("Invalid input. Please enter three values.")
            else:
                break
    
        while True:
            point_x_y_z = input("""
Enter points in the X, Y, Z direction e.g. 50, 60, 70
""")    
            point_x_y_z = point_x_y_z.split(',')
            point_x_y_z = [int(i) for i in point_x_y_z]
            if len(point_x_y_z) != 3:
                print("Invalid input. Please enter three values.")
            else:
                break
        # Subtract cub_origin from cub_end and divide the result by point_x_y_z element-wise
        grid_spacing_x_y_z = [(end - origin) / (point - 1) for origin, end, point in zip(cub_origin, cub_end, point_x_y_z)]
        # print(cub_origin, cub_end, point_x_y_z, grid_spacing_x_y_z)
        return grid_spacing_x_y_z, point_x_y_z, cub_origin
    
    
    def points_and_exten():
        
        


        while True:
            ext_x_y_z = input("""
Enter extension distance for x, y, z in bohrs e.g 3, 5, 4.2
""")    
            ext_x_y_z = ext_x_y_z.split(',')
            ext_x_y_z = np.array([int(i) for i in ext_x_y_z])
            if len(ext_x_y_z) != 3:
                print("Invalid input. Please enter three values.")
            else:
                break
       
  
    
        while True:
            point_x_y_z = input("""
Enter points in the X, Y, Z direction e.g. 50, 60, 70
""")    
            point_x_y_z = point_x_y_z.split(',')
            point_x_y_z = np.array([int(i) for i in point_x_y_z])
            if len(point_x_y_z) != 3:
                print("Invalid input. Please enter three values.")
            else:
                break
            
        ext_max_coords = max_coords + ext_x_y_z
        ext_min_coords = min_coords - ext_x_y_z
        ext_min_x, ext_min_y, ext_min_z = ext_min_coords
        ext_max_x, ext_max_y, ext_max_z = ext_max_coords
        
        
        grid_spac_x = (ext_max_coords[0] - ext_min_coords[0])/(point_x_y_z[0] - 1)
        grid_spac_y = (ext_max_coords[1] - ext_min_coords[1])/(point_x_y_z[1] - 1)
        grid_spac_z = (ext_max_coords[2] - ext_min_coords[2])/(point_x_y_z[2] - 1)
        
        origin = np.array([ext_min_x, ext_min_y, ext_min_z])
            
        
        # print(max_coords, type(max_coords), ext_x_y_z, type(ext_x_y_z), point_x_y_z)
        
        # print("------------------")
        # print(ext_max_coords, ext_min_coords)
        # print("------------------")

        # print (origin)
        # print("------------------")

        # print(grid_spac_x, grid_spac_y, grid_spac_z)
        
        point_x = point_x_y_z[0]
        point_y = point_x_y_z[1]
        point_z = point_x_y_z[2]
        return point_x, point_y, point_z, origin, grid_spac_x, grid_spac_y, grid_spac_z  
        



    def spac_poin_orig():
           
        while True:
            cub_origin = input(f"""
Enter origin of cube e.g {min_x - 3}, {min_y- 4}, {min_z -2}
""")
            cub_origin = cub_origin.split(',')
            cub_origin = [float(i) for i in cub_origin]
            if len(cub_origin) != 3:
                print("Invalid input. Please enter three values.")
            else:
                break
    
        while True:
            grid_spacing_xyz = input("""
Enter grid spacing for X, Y, Z direction e.g 0.1, 0.2, 0.3
""")
            grid_spacing_xyz = grid_spacing_xyz.split(',')
            grid_spacing_xyz = [float(i) for i in grid_spacing_xyz]
            if len(grid_spacing_xyz) != 3:
                print("Invalid input. Please enter three values.")
            else:
                break
    
        while True:
            point_x_y_z = input("""
Enter points in the X, Y, Z direction e.g. 50, 60, 70
""")    
            point_x_y_z = point_x_y_z.split(',')
            point_x_y_z = [int(i) for i in point_x_y_z]
            if len(point_x_y_z) != 3:
                print("Invalid input. Please enter three values.")
            else:
                break
            
                
        return grid_spacing_xyz, point_x_y_z, cub_origin

    ext_dist = 4

    while True:
        usr_inp = int(input("""
Select grid specification. Default extension distance: {} Bohr
0. Change extension distance. Applies for option 1 to 4.
1. Low quality grid with maximum 50 points on the widest spread coordinate
2. Medium quality grid with maximum 75 points on the widest spread coordinate
3. Fine quality grid with maximum 100 points on the widest spread coordinate
4. Ultra-Fine quality grid with maximum 125 points on the widest spread coordinate
5. Choose the origin and end position of your grid
6. Enter all points manually. Will be replaced with reading grid spec from other cubes later
7. Enter points and extension distance in bohrs
""".format(ext_dist)))

        if usr_inp == 0:
            ext_dist = float(input("Enter new extension distance: "))
        elif usr_inp >= 1 and usr_inp <= 4:
            if usr_inp == 1:
                result = syst_spec(50, ext_dist)
            elif usr_inp == 2:
                result = syst_spec(75, ext_dist)
            elif usr_inp == 3:
                result = syst_spec(100, ext_dist)
            elif usr_inp == 4:
                result = syst_spec(125, ext_dist)
            break  # exit the loop once a valid option is chosen
            
        elif usr_inp == 5:
            result = start_end()
            break # exit the loop once a valid option is chosen
            
        elif usr_inp == 6:
            result = spac_poin_orig()
            break # exit the loop once a valid option is chosen
        
        elif usr_inp == 7:
            result = points_and_exten()
            # print(len(result))
            break # exit the loop once a valid option is chosen
            
            
        else:
            print("Invalid option. Please try again.")


    
    return result
    

    
result = grid_specification()
# Access each element of the tuple
######## New  Task
#Modify this condition statement to take the value of the function called"

if len(result) == 5:
    grid_x = int(result[0])
    grid_y = int(result[1])
    grid_z = int(result[2])
    grid_spacing = float(result[3])
    grid_spacing = np.array([grid_spacing, grid_spacing, grid_spacing])
    origin = np.array(result[4])

elif len(result) == 3:
    grid_spacing = result[0]
    point_x_y_z = result[1]
    origin = np.array(result[2])
    
    grid_x = int(point_x_y_z[0])
    grid_y = int(point_x_y_z[1])
    grid_z = int(point_x_y_z[2])
    
elif len(result) == 7:
# point_x, point_y, point_z, origin, grid_spac_x, grid_spac_y, grid_spac_z  

    grid_spacing = result[0]

   
    grid_x = int(result[0])
    grid_y = int(result[1])
    grid_z = int(result[2])
    origin =  result[3]
    grid_spac_x = result[4]
    grid_spac_y = result[5]
    grid_spac_z = result[6]

    grid_spacing = np.array([grid_spac_x, grid_spac_y, grid_spac_z])
    
     
    
    origin = np.array(result[3])
    

x = np.arange(grid_x) * grid_spacing[0] + origin[0]
y = np.arange(grid_y) * grid_spacing[1] + origin[1]
z = np.arange(grid_z) * grid_spacing[2] + origin[2]

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)


###################################################################
def write_cube_file(filename, grid_data):
# def write_cube_file(filename, grid_params, grid_data):

    """
    Writes the grid data to a Gaussian cube file.
    Parameters:
        filename (str): Name of the output file.
        grid_params (dict): Dictionary containing grid parameters.
        grid_data (numpy.ndarray): Array containing grid data.

    Returns:
        None
    """

    with open(filename, 'w') as file:
        # Write the header section
        file.write("Generated by NBO2CUBE\n")
        file.write("Orbital\n")

        file.write("%4d %12.6f %12.6f %12.6f\n" % (len(coordinates) , origin[0], origin[1] , origin[2]))         
        file.write("%4d %12.6f %12.6f %12.6f\n" % (grid_x, grid_spacing[0], 0.000000, 0.000000))
        file.write("%4d %12.6f %12.6f %12.6f\n" % (grid_y,  0.000000, grid_spacing[1], 0.000000))
        file.write("%4d %12.6f %12.6f %12.6f\n" % (grid_z , 0.000000, 0.000000, grid_spacing[2]))
        

        for item in atom_info:
            file.write("%4s %10.6f %10.6f %10.6f %10.6f\n" % (int(item[0]), item[0], item[1]/bohr, item[2]/bohr, item[3]/bohr))
        
       # Write the grid data section
        for i in range(grid_x):
           for j in range(grid_y):
               for k in range(grid_z):
                   file.write(f"{grid_data[i, j, k].real:>13.5E}")
                   if (k + 1) % 6 == 0:
                       file.write("\n")
               file.write("\n")
        
                
        print(f"Grid data has been written to file: {filename}")

def electron_density(basis_set, coordinates, point, cmo):
    psi = 0.0 
    angular_parts = {}  
    r_values = {}  
    atom_coords_dict = {}  
    dxdydz_dict = {}  
    bas_res_dict = {}  

    for basis, c in zip(basis_set, cmo):
        if c == 0 or abs(c) <= 1e-15:
            continue

        if basis['CENTER'] in atom_coords_dict:
            atom_coords = atom_coords_dict[basis['CENTER']]
            dx, dy, dz = dxdydz_dict[basis['CENTER']]
        else:
            atom_coords = coordinates[basis['CENTER'] - 1]
            dx, dy, dz = np.array(point) - atom_coords
            atom_coords_dict[basis['CENTER']] = atom_coords  
            dxdydz_dict[basis['CENTER']] = (dx, dy, dz)  

        if basis['CENTER'] in r_values:
            r = r_values[basis['CENTER']]
        else:
            r = np.linalg.norm(np.array(point) - atom_coords)
            r_values[basis['CENTER']] = r  
        
        if r > 200:
            continue
        zeta_small = min(basis['exps'])
        if np.exp(-zeta_small * r**2) < 1e-15:
            continue

        if (basis['CENTER'], basis['orb_val']) in angular_parts:
            angular_part = angular_parts[(basis['CENTER'], basis['orb_val'])]
        else:
            angular_part  = ang_res_lamda(dx, dy, dz,  basis['orb_val']) 
            angular_parts[(basis['CENTER'], basis['orb_val'])] = angular_part

        if basis['shell_num'] in bas_res_dict:
            bas_res = bas_res_dict[basis['shell_num']]
        else:
            bas_res = sum(coeff * np.exp(-zeta * r**2) for coeff, zeta in zip(basis['coeffs'], basis['exps']))
            bas_res_dict[basis['shell_num']] = bas_res  

        psi += c * bas_res * angular_part
    
    density = psi 

    return density
coordinates = np.array(coordinates)
coordinates = coordinates /bohr
grid_data = np.empty((grid_x * grid_y * grid_z))

def unvectorized():
    for file, orbitals in orbital_dict.items():
        for CMO, item in zip(orbitals, orbital_index):
            grid_data = np.empty((grid_x * grid_y * grid_z))
            for i, point in enumerate(points):
                result = electron_density(data, coordinates, point, CMO)
                grid_data[i] = result
            
            grid_data = grid_data.reshape(grid_x, grid_y, grid_z)
            # print(grid_data, len(grid_data))
            
            filename = file + "-" + str(item) + ".cube"
            write_cube_file(filename, grid_data)
    


def vectorization(basis_set, coordinates, points, cmo):
    """
    Calculate the electron density at given points in space.

    Parameters:
    basis_set (list): The basis set information.
    coordinates (np.array): The coordinates of the atoms.
    points (np.array): The points in space where the density is to be calculated.
    cmo (list): The coefficients of the canonical molecular orbital.

    Returns:
    np.array: The electron densities at the given points.
    """
    densities = np.zeros(len(points))
    # print(cmo, type(cmo))
    # Multiply the 50th to 60th element of cmo by zero
    # cmo[126:160] = cmo[126:160] *0
    # cmo[73] = 0.8


    # print(cmo[73: 120])
    for basis, c in zip(basis_set, cmo):
        if abs(c) <= 1e-15:
            continue
        atom_coords = coordinates[basis['CENTER'] - 1][:, np.newaxis]

        dx, dy, dz = points.T - atom_coords
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        
        angular_part  = ang_res_lamda(dx, dy, dz, basis['orb_val'])
            
        for coeff, zeta in zip(basis['coeffs'], basis['exps']):
            expon = np.exp(-zeta * r**2)
            densities +=  np.round((c * coeff * angular_part * expon), 99)
    return densities


point_1 = np.empty((grid_x * grid_y * grid_z, 3))
index = 0

z_points = []
for i in range(grid_x):
    for j in range(grid_y):
        for k in range(grid_z):
            point = origin + np.array([i, j, k]) * grid_spacing
            point_1[index] = point
            z_points.append(point[2]* bohr)
            index += 1

# print(z_points)





def process_chunk(data, coordinates, chunk, CMO):
    """
    Process a chunk of data points for vectorization.

    Parameters:
    - data (array_like): The dataset to be processed.
    - coordinates (array_like): The spatial coordinates corresponding to the data points.
    - chunk (array_like): A subset of the data points to be processed.
    - CMO (array_like): The current molecular orbital information.

    Returns:
    - array_like: The vectorized representation of the chunk.
    """
    
    return vectorization(data, coordinates, chunk, CMO)

def perform_operation(data, coordinates, point_1, grid_x, grid_y, grid_z, orbital_lists, items, orbital_file):
    """
    Perform a series of operations to process data points, vectorize them, and write the results to a cube file.

    Parameters:
    - data (array_like): The dataset to be processed.
    - coordinates (array_like): The spatial coordinates corresponding to the data points.
    - point_1 (array_like): The initial set of data points.
    - grid_x (int): The size of the grid in the x-dimension.
    - grid_y (int): The size of the grid in the y-dimension.
    - grid_z (int): The size of the grid in the z-dimension.
    - orbital_lists (list): A list of molecular orbital information.
    - items (list): A list of indices corresponding to the molecular orbitals.
    - orbital_file (str): The base name for the output cube files.

    Returns:
    None: This function does not return a value but writes output to cube files.
    """
    for file, orbitals in orbital_dict.items():
        for CMO, item in zip(orbitals, orbital_index):
            # Divide point_1 into four chunks
            chunks = np.array_split(point_1, 4)
            
            # Create a pool of workers
            with Pool(4) as p:
                func = partial(process_chunk, data, coordinates, CMO=CMO)
                results = p.map(func, chunks)
            
            # Combine the results
            densities = np.concatenate(results)
            
            # Reshape the densities
            grid = densities.reshape((grid_x, grid_y, grid_z))
            
            filename = file + "-" + str(item) + ".cub"
            
            
            write_cube_file(filename, grid)
    
   
def cpprun(points):
    for file, orbitals in orbital_dict.items():
        for CMO, item in zip(orbitals, orbital_index):
            grid_data = np.empty((grid_x * grid_y * grid_z))
            psi_vec = electron_density_opt_omp.electron_density(data, coordinates, points, CMO, None)
            grid_data = psi_vec.reshape(grid_x, grid_y, grid_z)
            filename = file + "-" + str(item) + ".cube"
            write_cube_file(filename, grid_data)



run_method = int(input("""
1. Use optimized C++
2. To run in parallel. This option is very fast at least for up to 500 basis functions.
3. Run in serial. Runs slower. Use when option one fails for very large basis sets.
Select calculation mode (1, 2, or 3):"""))

while True:

    if run_method == 1:
        start = time.perf_counter()
        cpprun(points)
        end = time.perf_counter()
        print(f"cpprun() runtime: {end - start:.6f} seconds")
        break
    elif run_method == 2:
        start = time.perf_counter()
        perform_operation(data, coordinates, point_1, grid_x, grid_y, grid_z, orbital_lists, items, orbital_file)
        end = time.perf_counter()
        print(f"perform_operation() runtime: {end - start:.6f} seconds")
        break
    elif run_method == 3:
        start = time.perf_counter()
        unvectorized(points)
        end = time.perf_counter()
        print(f"unvectorized() runtime: {end - start:.6f} seconds")
        break
    else:
        print("Enter 1, 2 or 3.")


