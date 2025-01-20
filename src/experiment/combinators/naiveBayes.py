import math
from sklearn.naive_bayes import GaussianNB
import warnings

from src.common.common import fixMetric


warnings.filterwarnings("ignore")


class NaiveBayesCombinator:
	def __init__(self, defs):
		self.__fields = defs['fields']
		self.__label = defs['label']
		self.__clf = GaussianNB(
			priors=None,
			var_smoothing=1e-09
		)

	def name(self):
		return 'naive-bayes'

	def shortName(self):
		return 'naive-bayes'

	def field(self):
		return 'NBS_LABEL'

	def train(self, df):
		X = df[self.__fields]
		y = df[self.__label]
		self.__clf.fit(X, y)

	def combine(self, row):
		data = [row[self.__fields]]
		r = self.__clf.predict(data)[0]
		return r
