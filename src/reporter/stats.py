class ExpStats:
    def __init__(self):
        self.__stats = dict()

    def prepare(self, method):
        if method not in self.__stats:
            self.__stats[method] = {
                'x': [],
                'y': []
            }

    def register(self, method, x, y):
        self.__stats[method]['x'].append(x)
        self.__stats[method]['y'].append(y)

    def methods(self):
        return self.__stats.keys()

    def stats(self, method):
        print(method, self.__stats[method])
        return self.__stats[method]
