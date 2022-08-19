Each row contains one reflectance spectrum (denoised with convolution filter and normalised at 550 nm). The values are in the following order:
  - wavelengths from 450 nm to 2450 nm, step 5 nm (401 values)
  - if labels are present
    - olivine, orthopyroxene, clinopyroxene, plagioclase relative volume fraction (4 values)
    - fayalite, forsterite of olivine (2 values)
    - ferrosilite, enstatite, wollastonite of orthopyroxene (3 values)
    - ferrosilite, enstatite, wollastonite of clinopyroxene (3 values)
    - enorthosite, albite, orthoclase of plagioclase (3 values)
    
    - if cleaning was applied
      - labels depend on config file (in this case, there are no wollastonite of orthopyroxene, and no information about plagioclase)

    - labels are normalised to be from 0 to 1 (i.e. pyroxene Fs40 En55 Wo5 -> Fs=0.4, En=0.55, Wo=0.05 in data)
  
Metadata contains:
  - asteroid_spectra-denoised-norm-meta.dat
    - asteroid number, Bus-DeMeo taxonomny class, slope, PC1'--PC5' (PC computed after removing slope)
  - combined-denoised-norm*-meta.dat
    - information about samples extracted mostly from RELAB adtabase
