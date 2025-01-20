import math
from sklearn.linear_model import LogisticRegression
import warnings

from src.common.common import fixMetric


warnings.filterwarnings("ignore")


class LogisticRegressionCombinator:
	def __init__(self, defs):
		self.__fields = defs['fields']
		self.__label = defs['label']
		self.__clf = LogisticRegression(
			penalty='l2',
			dual=False,
			tol=0.0001,
			C=1.0,
			fit_intercept=True,
			intercept_scaling=1,
			class_weight=None,
			solver='lbfgs',
			max_iter=100,
			warm_start=False
		)

	def name(self):
		return 'log-regression'

	def shortName(self):
		return 'log-reg'

	def field(self):
		return 'LGR_LABEL'

	def train(self, df):
		X = df[self.__fields]
		y = df[self.__label]
		self.__clf.fit(X, y)

	def combine(self, row):
		data = [row[self.__fields]]
		r = self.__clf.predict(data)[0]
		return r
