import pandas


class data:

    def __init__(self, file_name):
        self.data = pandas.read_csv(file_name)
