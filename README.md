# Data Poisoning on Regression

This repository contains the full experiment notebook for the data poisoning on regression project.

The main notebook is:

```text
Full_experiments.ipynb
```

The data files are not stored directly in this GitHub repository because they are too large for convenient GitHub upload. Instead, the datasets can be downloaded separately from Google Drive.

## Download the data

Download the required data files from the following Google Drive folder:

https://drive.google.com/drive/folders/11sUVKhZ0Lvuie7VSViaR1HzkEwyf1THF?usp=sharing

After downloading, place all of the data files in the **same folder** as `Full_experiments.ipynb`.

The final repository folder should look like this:

```text
Data-poisoning-on-regression/
├── Full_experiments.ipynb
├── airfoil_self_noise.csv
├── blogData_train.csv
├── california.csv
├── casp.csv
├── communities.data
├── concrete.csv
├── house-processed.csv
├── loan_sample.csv
├── loan.csv.gz
├── realestate.csv
├── warfarin.csv
└── warfarin.xls
```

Please keep the dataset file names unchanged, since the notebook expects the files to be available with these names.

## Running the notebook

First, clone this repository:

```bash
git clone https://github.com/[YOUR_USERNAME]/Data-poisoning-on-regression.git
cd Data-poisoning-on-regression
```

Then download the data files from the Google Drive folder linked above and move them into this same directory.

Open the notebook using Jupyter Notebook:

```bash
jupyter notebook Full_experiments.ipynb
```

or using JupyterLab:

```bash
jupyter lab Full_experiments.ipynb
```

Then run the notebook cells in order.

## Important note about file paths

The notebook assumes that all datasets are located in the same directory as `Full_experiments.ipynb`.

That means the folder should contain both the notebook and the data files directly, rather than putting the datasets inside a separate subfolder.

For example, this is correct:

```text
Data-poisoning-on-regression/
├── Full_experiments.ipynb
├── concrete.csv
├── casp.csv
└── warfarin.csv
```

This may require changing file paths in the notebook:

```text
Data-poisoning-on-regression/
├── Full_experiments.ipynb
└── data/
    ├── concrete.csv
    ├── casp.csv
    └── warfarin.csv
```
