class MajorityCombinator:
    def __init__(self, defs):
        self.__fields = defs['fields']
        pass

    def name(self):
        return 'majority'

    def shortName(self):
        return 'maj'

    def field(self):
        return 'MAJ_LABEL'

    def train(self, df):
        # no training is needed for majority
        return

    def combine(self, row):
        data = row[self.__fields].tolist()
        res = max(set(data), key=data.count)
        return res
