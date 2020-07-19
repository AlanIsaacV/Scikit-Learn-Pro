from utils import Utils
from models import Models

if __name__ == '__main__':
	utils = Utils()
	data = utils.load_from_csv('data/felicidad.csv')

	x, y = utils.features_target(data, ['score', 'rank', 'country'], ['score'])

	model = Models()
	model.grid_training(x, y)
