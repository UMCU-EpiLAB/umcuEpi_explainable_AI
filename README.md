![image](https://github.com/UMCU-EpiLAB/umcuEpi_explainable_AI/assets/73480193/47e753a9-ec92-4810-bf24-1ee29b553f29)



**This code supports the findings in the manuscript titled '_Machine learning for (non-)epileptic tissue detection from the intraoperative electrocorticogram_' by Hoogteijling et al.** (doi.org/10.1016/j.clinph.2024.08.012) Please cite this article for every work that uses code or data from this repository.

## 🛠 Python dependencies and packages
The code was developed in Pyhton 3.9.2 using Spyder 4.2.1.
The following packages were used and can be installed via pip install or conda install:

- Pandas (1.5.1)
- Numpy (1.21.5)
- sklearn (1.3.0)
- matplotlib.pyplot (3.3.4)
- shap (0.40.0)

## 🧠 Data preparation
Please read the manuscript for data aquisition and pre-processing details. The intraoperative electrocorticogram spectral bands power data set can be download from the Supplementary Data and from DataVerse.

Training and test set data are organized in a .CSV file where one column represent the label and every other column represent a spectral feature. Each row represents a single ioECoG 20-second epoch, similar to:

| Index | Label | Spectral feature 1  | Spectral feature 2 | ...|
| :------------: | :------------: |:---------------:| :-----:|:---:|
| 0      | Label |Feature value | Feature value |...|
| 1      | Label |Feature value |  Feature value|...|
| 2 | Label |Feature value |Feature value|...|
| ⋮| ⋮| ⋮ |⋮| ⋱ |.

For the training .CSV file, the last column represents the fold the 20-second epoch was allocated to for five-fold cross-validation.

## 👩‍💻 Code organization
Make a copy of the loadData_example.py and name it loadData.py. In loadData.py, specify the path to the folder containing the Xy_train and Xy_test .CSV files at line 14.

Run mainSH.py for ETC performance in five-fold cross-validation and on the test set. The last cell in this code will show the SHAP analysis.
