import caffe
import numpy as np

class GeneralDataLayer(BasePythonLayer):

    def setup(self, bottom, top):
        assert(len(bottom) == 0)
        assert(len(top) >= 1)
        config = []
        with open(self.param_str, 'r') as config_file:
            for line in config_file:
                config.append(line)
        self._csvfile, self._minibatch_size, self._delimiter = config
        self._have_read = 0
        
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        top[0].data[:] = self._data_buffer
        if len(top) > 1:
            raise NotImplemented

    def backward(self):
        raise Exception('Data Layer does not have backward function.')

    def prefetch(self):
        self._data_buffer = self._minibatch_reader()

    def _minibatch_reader(self):
        bffr = []
        with open(self._csvfile, 'rb') as csvf:
            csvf.seek(self._have_read)
            for line in csvf:
                bffr.append(line.split(self._delimiter))
                if len(bffr) >= self._minibatch_size:
                    break
            self._have_read = csvf.tell()
        return bffr
