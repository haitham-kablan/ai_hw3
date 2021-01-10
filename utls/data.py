import csv


class data:

#TODO - u might need to iterate throgh date to clear wrong data
    def __init__(self):
        self.data = {}

    def init_data(self, file_name):
        with open(file_name, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0

            for row in csv_reader:
                if line_count == 0:
                    for f in row:
                        self.data[f] = []

                else:
                    for f in row:
                        self.data[f].append(row[f])

                line_count += 1