# Asteroid Spectra Neural Network Algorithm

This repository contains a neural network algorithm for deriving mineral compositions from reflectance spectra. It is primarily designed for use with .npz files and can be adapted for specific spectral configurations. The following documentation provides detailed guidance for setting up, running, and customising the algorithm.

---

## **Version Information**

- Preferred version: **v3.0**
- Python version: **3.10**
- Required libraries: Refer to `requirements.txt`

---

## **Directory Structure**

The algorithm relies on specific directory structures and absolute paths for file organisation. Paths are defined in `./modules/_constants.py`:

- **`_project_dir`**: The absolute path to your project (e.g., the location of `main_composition.py`).
- **`_subdirs`**: Subdirectories relative to `_project_dir`, such as:
  - `./models`: Directory for model files.
  - `./models/composition/450-2450-5-550`: Example path for composition models trained on spectra starting at 450 nm, ending at 2450 nm, with a wavelength step of 5 nm and normalisation at 550 nm.

### **Model File Naming Convention**

Model files contain encoded information about the quantities modelled. The naming follows this structure:

- **First 4 digits**: Modal abundance for OL (olivine), OPX (orthopyroxene), CPX (clinopyroxene), PLG (plagioclase).
  - Example: `1110` means optimised for OL, OPX, CPX, but not PLG.
- **Endmember abundances**:
  - OL: `Fa`, `Fo`
  - OPX: `Fs`, `En`, `Wo`
  - CPX: `Fs`, `En`, `Wo`
  - PLG: `An`, `Ab`, `Or`

Example: `1110-11-110-111-000` represents OL, OPX, and CPX, but no PLG, with specific endmember abundance configurations.

---

## **Preprocessing Files**

1. **Load the file**
2. **Reshape data**: Ensure your data has the shape `(N_spectra, len_spectrum)`.
   - For a single spectrum with shape `(len_spectrum,)`, reshape it to `(1, len_spectrum)`.
3. **Interpolation**:
   - For pretrained models: Interpolate the spectra to match the model's expected wavelength grid.
4. **Normalisation**:
   - For pretrained models: Normalise spectra according to the model's requirements.

---

## **Using Pretrained Models**

### **Steps to Evaluate a Spectrum**

1. **Prepare your data** as a 2D NumPy array or .npz file.
2. **Run the evaluation** using `./modules/NN_evaluate.py/evaluate`:
   - **Parameters**:
     - List of model paths.
     - Input data (array or path to .npz file).
     - Optional: `subdir` parameter to search for models in `_constants.py` paths.
3. **Alternative Option**:
   - Use <a href="https://sirrah.pythonanywhere.com/" target="_blank">https://sirrah.pythonanywhere.com/</a> to evaluate spectra via a web interface.
   - Data must be in a .txt file with wavelengths in the first row or column and reflectances in subsequent rows or columns.

---

## **Training New Models**

### **Dataset**

- Recommended dataset: `mineral-spectra_denoised_norm.npz`

### **Configuration Files**

- **`./modules/NN_config_composition.py`**:
  - Key parameters:
    - `comp_grid_setup`: Specifies grid settings.
    - `instrument`: Set to `None` unless otherwise required.
    - `wvl_grid`: Define the wavelength grid.
    - `normalisation`: Use `adaptive` or other suitable options.
- **`./modules/NN_HP.py`**:
  - Contains hyperparameters. Default values are acceptable for most cases.

### **Steps to Train a Model**

1. Modify the configuration in `./modules/NN_config_composition.py` as needed.
2. Write a function to read your dataset.
3. Include interpolation steps (lines 69--72 in `./modules/NN_data.py`).
4. Run `main_composition.py`.

---

## **System Independence**

- Version v3.0 adapts to different systems. Only `_project_dir` needs to be set correctly; other paths adjust automatically.

---

## **Notes and Recommendations**

- For first-time use, focus on evaluating pretrained models.
- Use the web interface for quick results if the local setup is challenging.
- When training new models, ensure all spectra are interpolated to the correct grid and properly normalised.

For additional assistance or clarifications, please feel free to reach out.

