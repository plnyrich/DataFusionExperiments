import math
import itertools
from src.common.common import fixMetric


class OracleCombinator:
	def __init__(self, defs):
		self.__fields = defs['fields']
		self.__label = defs['label']
		self.__numOfClasses = len(defs['classes'])

	def name(self):
		return 'orc'

	def shortName(self):
		return 'oracle'

	def field(self):
		return 'ORC_LABEL'

	def train(self, df):
		pass

	def combine(self, row):
		outputs = row[self.__fields].to_list()
		label = row[self.__label]
		if label in outputs:
			return label
		else:
			return (label + 1) % self.__numOfClasses
