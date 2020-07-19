from numpy.lib import utils
import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from utils import Utils

class Models:
	def __init__(self):
		self.model = {
			'SVR' : SVR(),
			'GRADIENT' : GradientBoostingRegressor()
		}

		self.params = {
			'SVR' : {
				'kernel' : ['linear', 'poly', 'rbf'],
				'gamma' : ['auto', 'scale'],
				'C' : [1, 5, 10]
			}, 
			'GRADIENT' : {
				'loss' : ['ls', 'lad'],
				'learning_rate' : [0.01, 0.05, 0.1]
			}
		}

	def grid_training(self, x, y):
		best_score = 999
		best_model = None

		for name, model in self.model.items():
			grid = GridSearchCV(model, self.params[name], cv=5).fit(x, y.value.ravel())
			score = np.abs(grid.best_score_)

			if score < best_score:
				best_score = score
				best_model = grid.best_estimator_

		utils = Utils()
		utils.model_export(best_model, best_score)