Data are stored in *.npz format. Each data file contains spectra, corresponding wavelengths, and metadata. Optionally, the files contain labels and notes about label and metadata. The file can be loaded with **data = numpy.load('name_of_a_file.npz, allow_pickle=True)**. Saved quantites are visualised with **data.files**.

**Data files which contain -denoised were denoised via a convolution filter.**

**Data files which contain -norm were normalised** usually at 550 nm.

Each row in the **spectra** files (accessible with **data["spectra"]**) contains one reflectance spectrum. The corresponding wavelengths can be found in **data["wavelengths"]** and are usually from 450 nm to 2450 nm, step 5 nm (401 values).

Metadata are accessible through **data["metadata"]** and its optional notes via **data["metadata key"]** and contains:
  - asteroid_spectra-*.npz
    - asteroid number, Bus-DeMeo taxonomny class, slope, PC1' - PC5' (PC computed after removing slope)
  - combined-denoised-norm.npz
    - information about samples extracted mostly from RELAB database (see **Sample_Catalogue.xlsx** in version v1.0)
  - Itokawa-denoised-norm.npz and Eros-denoised-norm.npz
    - longitude, latitude, asteroid name, instrument name

if labels are present; (see **data["labels"]** and **data["label metadata"]**), these are usually
  - composition
    - olivine, orthopyroxene, clinopyroxene, plagioclase in relative volume fraction (4 values)
    - fayalite, forsterite of olivine (2 values)
    - ferrosilite, enstatite, wollastonite of orthopyroxene (3 values)
    - ferrosilite, enstatite, wollastonite of clinopyroxene (3 values)
    - enorthosite, albite, orthoclase of plagioclase (3 values)
    - labels are normalised to be from 0 to 1 (i.e. pyroxene Fs40 En55 Wo5 -> Fs=0.4, En=0.55, Wo=0.05 in data)
  - taxonomy
    - taxonomy classes
 
The data files contain:
- asteroid_spectra-*.npz
  - reflectance spectra from DeMeo et al. 2009 and Binzel et al. 2019
  - labels and their key (Bus-DeMeo class)
    - **reduced** contains taxonomy classes from our reduced taxonomy system (see Table 1 in the reference paper)
    - **deleted** contains taxonomy classes which we did not use (see Table 1 in the reference paper)
  - used wavelength grid
  - metadata
- combined-denoised-norm.npz
  - combined dataset used for training, validation, and evaluation
  - used wavelength grid
  - labels and their key
  - relevant metadata and their key
  - the metadata and mineral and chemical analyses are listed in table **Sample_Catalogue.xlsx**
- Eros-denoised-norm.npz
  - averaged reflectance spectra from NIS instrument
  - used wavelength grid
  - metadata
- Itokawa-denoised-norm.npz
  - averaged reflectance spectra from NIRS instrument
  - used wavelength grid
  - metadata
