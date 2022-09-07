Each row in the **dat** files contains one reflectance spectrum (denoised with convolution filter and normalised at 550 nm). The values are in the following order:
  - wavelengths from 450 nm to 2450 nm, step 5 nm (401 values)
  - if labels are present
    - olivine, orthopyroxene, clinopyroxene, plagioclase relative volume fraction (4 values)
    - fayalite, forsterite of olivine (2 values)
    - ferrosilite, enstatite, wollastonite of orthopyroxene (3 values)
    - ferrosilite, enstatite, wollastonite of clinopyroxene (3 values)
    - enorthosite, albite, orthoclase of plagioclase (3 values)
    
    - if cleaning was applied (file name contains -clean)
      - labels depend on config file (in this case, there are no wollastonite of orthopyroxene, and no information about plagioclase)

    - labels are normalised to be from 0 to 1 (i.e. pyroxene Fs40 En55 Wo5 -> Fs=0.4, En=0.55, Wo=0.05 in data)
 

Metadata contains:
  - asteroid_spectra-denoised-norm-meta.dat
    - asteroid number, Bus-DeMeo taxonomny class, slope, PC1' -- PC5' (PC computed after removing slope)
  - combined-denoised-norm*-meta.dat
    - information about samples extracted mostly from RELAB adtabase


**Datafiles which contain -norm were normalised at 550 nm.**

**Datafiles which contain -denoised were denoised via a convolution filter.**

The data files contain:
- Chelyabinsk-denoised-norm-nolabel.dat
  - reflectance spectra of Chelyabinsk meteorite from Kohout et al. 2020 (in order SD 0% -- SD 100%, IM 0% -- IM 100%; SW 0 -- SW 700; see their Table 1)
- Kachr_ol_opx-denoised-norm-nolabel.dat
  - reflectance spectra of olivine and otrhopyroxene from Chrbolkova et al. 2021 (in order ol-Ar, ol-H, ol-He, ol-laser, py-Ar, py-H, py-He, py-laser (always from fresh to the most weathered); see their Fig. 2)
- asteroid_spectra-denoised-norm-nolabel.dat and asteroid_spectra-denoised-norm.dat
  - reflectance spectra from DeMeo et al. 2009 and Binzel et al. 2019
  - data with labels contain Bus-DeMeo class at the first position
- combined-denoised-norm.dat
  - combined dataset used for training, validation, and evaluation
  - known metadata and mineral and chemical analyses are listed in table **Sample_Catalogue.xlsx**
  - known metadata, mineral and chemical analyses, and reflectance spectra of the sample which pass through the selection criteria of the final model is in **Sample_Catalogue_Clean.xlsx**
