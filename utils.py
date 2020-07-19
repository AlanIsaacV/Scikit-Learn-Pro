import pandas as pd
import joblib

class Utils:
	def load_from_csv(self, path, **kwargs):
		return pd.read_csv(path, **kwargs)

	def features_target(self, dataset, columns_drop, column_target):
		x = dataset.drop(columns=columns_drop)
		y = dataset[column_target]

		return x, y

	def model_export(self, model, score):
		print(score)
		joblib.dump(model, './models/best_model.pkl')