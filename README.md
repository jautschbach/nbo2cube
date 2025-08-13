# nbo2cube
code to generate cube files (volume data) for orbitals generated with NBO

Author: Augustine Obeng, University at Buffalo, SUNY, J. Autschbach research group

## Overview

This program processes basis set files and orbital key files to produce Gaussian cube files representing molecular orbitals suitable for visualization.

## Features


- Reads basis sets in .47 format
- Converts basis functions to ORCA/Molden convention and normalizes coefficients
- Accepts NBO key files
- Allows specifying orbital index range and grid quality
- Optimized calculation modes: Python serial and Parallel run, or C++
- Generates cube files (*.cube) for selected orbitals

## Usage

IMPORTANT: Compile the CPP programs by executing the command `./compile_cpp.sh`

1. Launch the program as python3 nbo2cube.py.

2. When prompted, enter the basis set filename (recommended extension: .47)

   Example: `File.47`. Works with `File.31` too.

3. Enter the orbital/NBO key filename(s), separated by commas if needed.
   
   Example: `File.39, File.40, File.32` 

   Make sure these are valid NBO data files.
   Spin selection is prompted if we have an open shell system.

4. Specify orbital indices or a range.
   Example: `1, 3, 4, 5, 12-15, 20`

5. Select grid specification option:

   0. Change extension distance (applies to options 1-4)
   1. Low quality: max 50 points (widest axis)
   2. Medium quality: max 75 points per x,y,z direction
   3. Fine quality: max 100 points
   4. Ultra-Fine: max 125 points
   5. Choose grid origin/end manually
   6. Input all grid points manually (or import from other cubes)
   7. Specify points + extension in Bohr

   Example: Enter `3` for fine quality.

6. Select calculation mode:

   1. Optimized C++ (fastest)
   2. Parallel (very fast, up to ~500 basis functions)
   3. Serial (slower, fallback)

   Example: Enter `1` for optimized C++.

## Output

- Generates one cube file per orbital in the format:
    `OrbitalFilename-OrbitalIndex.cube`

  Example: `File.40-12.cube`

- Each cube file contains volumetric grid data for visualization.


## NOTES

- Basis function normalization is performed.
- Closed-shell and open-shell systems are detected and handled automatically.
- Total processing time depends on grid quality and number of orbitals.

