import csv
import caffe
import numpy as np

class GeneralDataLayer(BasePythonLayer):

    def setup(self, bottom, top):
        assert(len(bottom) == 0)
        assert(len(top) >= 1)
        self._csvfile, self._delimiter, self._quotechar = self.param_str.split()

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        top[0].data[:] = self._data_buffer
        if len(top) > 1:
            raise NotImplemented

    def backward(self):
        raise Exception('Data Layer does not have backward layer.')

    def prefetch(self):
        self._data_buffer = []
        with open(self._csvfile, 'rb') as csvf:
            datareader = csv.reader(csvf, delimiter = self._delimiter, quotechar = self._quotechar)
            for line in datareader:
                self._data_buffer.append(line)
