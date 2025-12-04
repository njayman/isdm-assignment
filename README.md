# CIS 111-6 Assignment

## Run in local machine

To execute this code in local machine or a vm, we need to install `uv` and `python` in our system. Follow the [UV installation instruction](https://docs.astral.sh/uv/getting-started/installation/). Then in the project directory, run the following command to create a virtual environment.

```sh
uv venv .venv
```

This will create a virtual environment for running this setup. Now run the following command to sync the dependencies to the enviromnent.

```sh
uv sync
```

Now use any `ipynb` editor like vscode, pycharm etc. to run the cells in `CIS_111_6_weekly_report.ipynb` file.

## Run in colab

Upload the `CIS_111_6_weekly_report.ipynb` file in google drive and open it with google colab. Upload the `bank-additional.csv` file into the root directory. If you want to use another directory, make sure to change the directory in the following line:

```python
df = pd.read_csv("bank-additional.csv", sep=';')
```

## Notes

This code uses scikit learn which is heavily based on CPU performance. I have used the `scikit-learn-intelex` package to optimize the code for intel cpu. To remove optimization, comment the following line:

```python
patch_sklearn()
```

The dependencies are managed using `uv`, a high performance and project manager. You can add the same python versions and packages used by initialize the project uv. To install the exact python version, run the followinf command.

```sh
uv python install
```
