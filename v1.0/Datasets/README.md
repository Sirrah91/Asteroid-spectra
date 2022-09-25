Data are stored in *.npz format. Each data file contain spectra, corresponding wavelengths, and metadata. Optionally, the files contain labels and notes about label and metadata. The file can be loaded with **data = numpy.load('name_of_a_file.npz, allow_pickle=True)**. Saved quantites are visualised with **data.files**.

**Data files which contain -denoised were denoised via a convolution filter.**

**Data files which contain -norm were normalised at 550 nm.**

Each row in the **spectra** files (accesible with **data["spectra"]**) contains one reflectance spectrum (denoised with convolution filter and normalised at 550 nm). The corresponding wavelengths can be found in **data["wavelengths"]** and usually are from 450 nm to 2450 nm, step 5 nm (401 values).

if labels are present; (see **data["labels"]** and **data["label metadata"]**), these are usually
  - olivine, orthopyroxene, clinopyroxene, plagioclase in relative volume fraction (4 values)
  - fayalite, forsterite of olivine (2 values)
  - ferrosilite, enstatite, wollastonite of orthopyroxene (3 values)
  - ferrosilite, enstatite, wollastonite of clinopyroxene (3 values)
  - enorthosite, albite, orthoclase of plagioclase (3 values)
    
  - labels are normalised to be from 0 to 1 (i.e. pyroxene Fs40 En55 Wo5 -> Fs=0.4, En=0.55, Wo=0.05 in data)
 
Metadata are accesible through **data["metadata"]** and optional notes via **data["metadata key"]** and contains:
  - asteroid_spectra-denoised-norm.npz
    - asteroid number, Bus-DeMeo taxonomny class, slope, PC1' - PC5' (PC computed after removing slope)
  - combined-denoised-norm.npz
    - information about samples extracted mostly from RELAB adtabase (see **Sample_Catalogue.xlsx**)
  - Chelyabinsk-denoised-norm.npz and Kachr_ol_opx-denoised-norm.npz
    - order in which the spectra were saved

The data files contain:
- Chelyabinsk-denoised-norm.npz
  - reflectance spectra of Chelyabinsk meteorite from Kohout et al. 2020 in order SD 0% -- SD 100%, IM 0% -- IM 100%; SW 0 -- SW 700 (see **data["metadata"]** and their Table 1)
  - used wavelength grid
  - labels and their key
- Kachr_ol_opx-denoised-norm.npz
  - reflectance spectra of olivine and otrhopyroxene from Chrbolkova et al. 2021 in order ol-Ar, ol-H, ol-He, ol-laser, py-Ar, py-H, py-He, py-laser, always from fresh to the most weathered; (see **data["metadata"]** and their Fig. 2)
  - used wavelength grid
  - labels and their key
- asteroid_spectra-denoised-norm.npz
  - reflectance spectra from DeMeo et al. 2009 and Binzel et al. 2019
  - labels and their key (Bus-DeMeo class)
  - used wavelength grid
  - metadata
- combined-denoised-norm.npz
  - combined dataset used for training, validation, and evaluation
  - used wavelength grid
  - labels and their key
  - relevant metadata and their key
  - the metadata and mineral and chemical analyses are listed in table **Sample_Catalogue.xlsx**
  
