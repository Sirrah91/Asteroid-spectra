Data are stored in *.npz format. Each data file contains spectra, corresponding wavelengths, and metadata. Optionally, the files contain labels and notes about labels and metadata. The file can be loaded with **data = numpy.load("name_of_a_file.npz", allow_pickle=True)**. Saved quantities are visualised with **data.files**.

**Data files which contain -denoised were denoised via a convolution filter.**

**Data files which contain -norm were normalised at 550 nm.**

Each row in the **spectra** files (accessible with **data["spectra"]**) contains one reflectance spectrum (denoised with a convolution filter and normalised at 550 nm). The corresponding wavelengths can be found in **data["wavelengths"]** and are usually from 450 nm to 2450 nm, step 5 nm (401 values).

Metadata are accessible through **data["metadata"]** and its optional notes via **data["metadata key"]** and contains:
  - asteroid_spectra-denoised-norm.npz
    - asteroid number, Bus-DeMeo taxonomy class, slope, PC1' - PC5' (PC computed after removing slope)
  - combined-denoised-norm.npz
    - information about samples extracted mostly from RELAB database (see **Sample_Catalogue.xlsx**)
  - Chelyabinsk-denoised-norm.npz and Kachr_ol_opx-denoised-norm.npz
    - order in which the spectra were saved

if labels are present; (see **data["labels"]** and **data["label metadata"]**), these are usually
  - olivine, orthopyroxene, clinopyroxene, and plagioclase in relative volume fraction (4 values)
  - fayalite and forsterite of olivine (2 values)
  - ferrosilite, enstatite, and wollastonite of orthopyroxene (3 values)
  - ferrosilite, enstatite, and wollastonite of clinopyroxene (3 values)
  - enorthosite, albite, and orthoclase of plagioclase (3 values)
    
  - labels are normalised to be from 0 to 1 (i.e. pyroxene Fs40 En55 Wo5 -> Fs=0.4, En=0.55, Wo=0.05 in data)
 
The data files contain:
- Chelyabinsk-denoised-norm.npz
  - reflectance spectra of the Chelyabinsk meteorite from Kohout et al. (2020) in order SD 0% -- SD 100%, IM 0% -- IM 100%; SW 0 -- SW 700 (see **data["metadata"]** and their Table 1)
  - used wavelength grid
  - labels and their key
- Kachr_ol_opx-denoised-norm.npz
  - reflectance spectra of olivine and orthopyroxene from Chrbolkova et al. (2021) in order ol-Ar, ol-H, ol-He, ol-laser, py-Ar, py-H, py-He, py-laser, always from fresh to the most weathered; (see **data["metadata"]** and their Fig. 2)
  - used wavelength grid
  - labels and their key
- asteroid_spectra-denoised-norm.npz
  - reflectance spectra from DeMeo et al. (2009) and Binzel et al. (2019)
  - labels and their key (Bus-DeMeo class)
  - used wavelength grid
  - metadata
- combined-denoised-norm.npz
  - the combined dataset (RELAB and C-Tape databases and our own measurements) used for training, validation, and evaluation
    - compare **data["metadata"]** with records in RELAB and C-Tape databases to find the spectra in the databases (e.g. via SampleID)
  - used wavelength grid
  - labels and their key
  - relevant metadata and their key
  - the metadata and mineral and chemical analyses are listed in table **Sample_Catalogue.xlsx**
  
