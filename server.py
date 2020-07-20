import joblib

from flask import Flask
from flask import jsonify
from joblib import memory

from utils import Utils

app = Flask(__name__)

@app.route('/')
def predict():
	utils = Utils()
	data = utils.load_from_csv('data/felicidad.csv')
	x, _ = utils.features_target(data, ['score', 'rank', 'country'], [])
	prediction = model.predict(x.loc[[0]])#.values.reshape(1, -1))

	return jsonify({ 'prediction' : list(prediction) })

if __name__ == '__main__':
	model = joblib.load('./models/best_model.pkl')
	app.run()