import math
from sklearn.tree import DecisionTreeClassifier
import warnings

from src.common.common import fixMetric


warnings.filterwarnings("ignore")


class DecisionTreeCombinator:
	def __init__(self, defs):
		self.__fields = defs['fields']
		self.__label = defs['label']
		self.__clf = DecisionTreeClassifier(
			criterion='gini',
			splitter='best',
			max_depth=None,
			min_samples_split=2,
			min_samples_leaf=1,
			min_weight_fraction_leaf=0.0,
			max_features=None,
			max_leaf_nodes=None,
			min_impurity_decrease=0.0,
			class_weight=None,
			ccp_alpha=0.0,
			monotonic_cst=None
		)

	def name(self):
		return 'decision-tree'

	def shortName(self):
		return 'dec-tree'

	def field(self):
		return 'DCT_LABEL'

	def train(self, df):
		X = df[self.__fields]
		y = df[self.__label]
		self.__clf.fit(X, y)

	def combine(self, row):
		data = [row[self.__fields]]
		r = self.__clf.predict(data)[0]
		return r
