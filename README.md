# Medical Imaging Coursework

This repository contains my submission for the Medical Imaging coursework. The main deliverables are the Jupyter Notebooks for each module, which directly show the experimental results and figures. In addition, I have implemented extra software development preactice: I refactored the core code into a complete Python package with unit tests and auto-generated documentation.

---

## Repository Structure

The repository is organized as follows:

```
.
├── A2_Coursework.pdf
├── LICENSE
├── README.md
├── data/                     # Dataset directory
│   ├── Module1/              
│   ├── Module2/              
│   └── Module3/              # Data for Module 3 (CT Segmentation & Radiomics)：due to large size, 
│                             # you need to download/clone the module3 data separately (see instructions below).
├── docs/                     # Sphinx documentation files
│   ├── Makefile
│   ├── _build/               # Open HTML here
│   ├── _static/
│   ├── _templates/
│   ├── conf.py               # Sphinx configuration
│   ├── index.rst             # Main documentation index
│   └── make.bat              
├── medical_imaging/          # The Python package containing all modules
│   ├── __init__.py
│   ├── module1/
│   ├── module2/
│   └── module3/
├── report
│   ├── a2_report.pdf         # Coursework report
├── requirements.txts
├── setup.py
├── src/                      # Jupyter Notebooks for each module
│   ├── module1_notebook.ipynb     # Module 1: PET-CT Reconstruction
│   ├── module2_notebook.ipynb     # Module 2: MRI Denoising
│   └── module3_notebook.ipynb     # Module 3: CT Segmentation & Radiomics
└── tests/                    # Unit tests for the package
    ├── test_module1.py
    ├── test_module2.py
    └── test_module3.py
```

---

## Jupyter Notebooks

The core experiments for each module are provided as Jupyter Notebooks in the [src/](./src/) folder:

- **[module1_notebook.ipynb](./src/module1_notebook.ipynb)**  
  PET-CT Reconstruction  
  *This notebook demonstrates sinogram correction, various reconstruction methods (FBP, OS-SART, SIRT, OSEM, MLEM), and visualization of results.*

- **[module2_notebook.ipynb](./src/module2_notebook.ipynb)**  
  MRI Denoising  
  *This notebook shows how to load k-space data, perform MRI denoising using wavelet methods, and combine multi-coil images.*

- **[module3_notebook.ipynb](./src/module3_notebook.ipynb)**  
  CT Segmentation & Radiomics  
  *This notebook covers CT image segmentation, ROI extraction, evaluation metrics (Dice, IoU, etc.), radiomic feature extraction, visualization, and classification.*

For easy access, you can open these notebooks via your Jupyter Notebook server. For example, if you run:

```bash
jupyter notebook src/
```

you will see the list of notebooks and can click to open them.

---

## Data Organization

The **data/** folder is organized into three subfolders:

- **Module1/**: Contains all the required data for PET-CT Reconstruction.
- **Module2/**: Contains the full dataset for MRI Denoising.
- **Module3/**: Contains CT scans and segmentation masks for CT Segmentation & Radiomics.  
  **Note:** Due to the large size of Module3 data, you may need to perform a separate download or use a provided Git submodule/clone command to retrieve this dataset before running the corresponding notebook.

---

## Software Development Practices

To demonstrate best software development practices, I refactored the core code into a complete Python package under **medical_imaging/**. This package contains:

- **Module1**: PET-CT Reconstruction functions (sinogram correction and reconstruction algorithms).
- **Module2**: MRI Denoising functionality.
- **Module3**: CT Segmentation and Radiomics (segmentation, radiomic feature extraction, classification, etc.).

Additionally, I provided:
- **Unit tests** under the **tests/** folder, covering key functions and edge cases.
- **Sphinx documentation** in the **docs/** folder for auto-generated API reference.
- **Packaging scripts** (setup.py and requirements.txt) for installation and reproducibility.

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies**  
   Create a virtual environment (optional) and install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the Package (Bonus Extra)**  
   To install the package in development mode:
   ```bash
   pip install -e .
   ```

---

## Running the Notebooks and Tests

- **Jupyter Notebooks:**  
  Launch Jupyter Notebook from the repository root:
  ```bash
  jupyter notebook src/
  ```
  Open each notebook to view and run the experiments.

- **Unit Tests:**  
  Run the complete test suite with:
  ```bash
  python -m unittest discover tests
  ```

- **Documentation:**  
  To generate API documentation, navigate to the **docs/** folder and run:
  ```bash
  make html
  ```
  Then open the generated HTML files in **docs/_build/html/**.

---

## Additional Instructions for Module3 Data

For Module3 (CT Segmentation & Radiomics), the dataset is large. After cloning the repository, please follow these steps:
- **Download Module3 Data:**  
  Either use the provided download script or follow the instructions in the repository to fetch the Module3 data into the **data/Module3/** folder.
- **Verify Data:**  
  Ensure that the CT scan and segmentation mask NIfTI files are present (e.g., `case_0.nii` and `case_0_mask.nii`).
- **Run Notebook:**  
  Open **src/module3_notebook.ipynb** to process and visualize the data.

---

## License

MIT License © 2025 – see the [LICENSE](LICENSE) file for full text.

## Author

Tina Hou – [xh363@cam.ac.uk](mailto:xh363@cam.ac.uk)