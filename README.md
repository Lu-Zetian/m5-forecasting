# M5 Forecasting

This repo contains a compact reproduction of an M5 Forecasting workflow.
It prepares the M5 data, trains point and probabilistic models, and
exports Kaggle-ready submission files.

1. Place the original M5 CSV files in the `data/` folder with these names:
	- `calendar.csv`
	- `sell_prices.csv`
	- `sales_train_evaluation.csv`
	- `sample_submission_accuracy.csv`
	- `sample_submission_uncertainty.csv`

2. Open `main.ipynb` in Jupyter / VS Code and run the cells from top to bottom.

3. The notebook will prepare features, train the models, and write submissions to the `submissions/` folder:
	- `submission_accuracy.csv`
	- `submission_uncertainty_dist_lgbm.csv`
	- `submission_uncertainty_ngb.csv`

No extra configuration is needed beyond installing the Python dependencies listed in the notebook imports.
