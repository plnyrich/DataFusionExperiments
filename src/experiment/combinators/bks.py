import math
import itertools
import numpy as np
from src.common.common import fixMetric


class BKSCombinator:
	def __init__(self, defs):
		self.__classes = defs['classes']
		self.__fields = defs['fields']
		self.__numOfClfs = defs['num_of_classifiers']
		self.__label = defs['label']
		self.__bks = dict()

	def name(self):
		return 'bks'

	def shortName(self):
		return 'bks'

	def field(self):
		return 'BKS_LABEL'

	def train(self, df):
		self.__bks = dict()
		for k, v in df[self.__fields].value_counts().items():
			coords = ''.join([str(x) for x in k])
			stats = self._getStats(df, k)
			self.__bks[coords] = stats

	def combine(self, row):
		data = row[self.__fields]
		coords = self._permutationToCoords(data.to_list())
		if coords not in self.__bks:
			return self.__classes[np.random.randint(0, len(self.__classes))]
		r = self.__bks[coords]
		return max(r, key=r.get)

	@staticmethod
	def _createAllPossibleOutcomes(classes, numOfClassifiers):
		return itertools.product(classes, repeat=numOfClassifiers)

	@staticmethod
	def _permutationToCoords(permutation):
		return ''.join([str(x) for x in permutation])

	def _getStats(self, df, permutation):
		query = ' & '.join([f'{k} == {v}' for k, v in zip(self.__fields, permutation)])
		stats = df.query(query)[self.__label].value_counts().to_dict()
		return stats
