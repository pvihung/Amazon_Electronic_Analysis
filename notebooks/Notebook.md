# Notebook Usage Notes

This folder contains the main Jupyter notebooks used during model development, experimentation, and analysis.

## ABSA notebooks

The notebooks `absa_model1.ipynb` and `absa_model2.ipynb` were separated from the original `absa_pipeline_vfinal.ipynb` notebook.

They can be executed independently if you only need to reproduce or work with one specific ABSA model:

- `absa_pipeline_vfinal.ipynb`  
  Full final ABSA pipeline notebook.

- `absa_model1.ipynb`  
  Standalone notebook for ABSA Model 1.

- `absa_model2.ipynb`  
  Standalone notebook for ABSA Model 2.

Because Model 1 and Model 2 were split from the full ABSA pipeline, each notebook can be run separately without having to execute the entire `absa_pipeline_vfinal.ipynb`.

## Important note about local paths

These notebooks were not originally executed in the local project directory structure.  
Therefore, if you want to run them locally using the current repository layout, you need to update the dataset paths.

Since all notebooks are located inside the `notebooks/` folder, dataset files should be accessed by moving one level up to the project root, then entering the `dataset/` folder.

Use this path format:

```python
../dataset/<file_name>