Pseduo Magneto-Optical-Trap (pMOT)

AC-Stark-Shift-of-Rubidium-87
Theoretical atomic physics in the Sinclair Group - UW Madison.  Studying the scalar, vector, and tensor polarizability energy shifts due to an oscillating electric field perturbation.

FILES IN USE:
************************************************************
"pMOT 5p_32 Data Job"
Script that generates the data of energy shifts and polarizabilities for the 5p_3/2 F=3 state as a function of wavelength/frequency scan.  Outputs a CSV file.  Contains wavelength, m_F, S/V/T polarizabilities, S/V/T Energy Shifts, Total Energy Shift
************************************************************
"pMOT 5s_12 Data Job"
Script that generates the data of energy shifts and polarizabilities for the 5s_1/2 F=2 state as a function of wavelength/frequency scan.  Outputs a CSV file.  Contains wavelength, m_F, S/V/T polarizabilities, S/V/T Energy Shifts, Total Energy Shift
************************************************************
"pMOT Plotting Macro"
Takes the csv file as input and plots the energy shift, S, V, T polarizabilities, and S, V , T terms as functions of laser wavelength.  Creates the main plot of all of the differential polarizabilities overlaid on the same plot, positive-definite, log-scale.  Also creates an animation of the shift of each m_F sublevel over the wavelength range.
************************************************************
"Differential Plotting Macro"
Takes in the CSV of differential polarizability values and plots them as a function of wavelength/frequency.  This is the main output plot of this research!
************************************************************
"ARC Atom Glossary"
ARC is the Alkali Rydberg Calculator that handles calculations for all things atomic-physics-based.  In my case, it can return values for reduced matrix elements that are necessary for computing polarizabilties.  This script is mostly for playing around currently (May 25, 2026), but may be implemented in the official code over paper data values currently being used.
************************************************************
"Component Shifts Animation"
This is the code for the animation of the scalar, vector, and tensor terms of each m_F sublevel over incident wavelength scan.
************************************************************
"Energy Shift Animation"
This is the code for the animation of the total energy shift of each m_F sublevel over incident wavelength scan.
************************************************************
"1 Fourier Tone"
Calculates the viable candidate wavelengths where the V/S and V/T ratios are maximized for ONE beam.
************************************************************
"2 Fourier Tones"
Calculates the viable candidate wavelengths where the V/S and V/T ratios are maximized for TWO beams with their effects Fourier summed.
************************************************************
"3 Fourier Tones"
Calculates the viable candidate wavelengths where the V/S and V/T ratios are maximized for THREE beams with their effects Fourier summed.
************************************************************
************************************************************
OUTDATED FILES:
************************************************************
"AC Stark Shift"
This is the main initial program that I wrote that both calculates energy shift and polarizabilities and plots the results.  I later split these functions into a job submission script and plotting macro which I describe in more detail below.
************************************************************
"pMOT Data Job"
Computes the scalar, vector, and tensor polarizabilities for the 5p_3/2 manifold of Rb-87.  Iterates over a selected range of wavelengths (i.e. 400-1800 nm) to show the change of energy shift as a function of incident wavelength.  Stores data in a csv file that is saved to the home directory.  Columns in the csv are: 'Wavelength (nm)' 'm_F' 'Scalar Polarizability' 'Vector Polarizability' 'Tensor Polarizability' 'Scalar Term' 'Vector Term' 'Tensor Term' 'Energy Shift'
