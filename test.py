import py_recall_precision_layer
#import caffe
import numpy as np

class testObject:
    def __init__(self, data):
        self.data = data

a=np.array([[1,0,1,1,0],[0,1,0,1,0]])
b=np.array([[2,3,1,0,0],[10,20,30,0,15]])
#c=caffe.Layer()

bottom = [testObject(a), testObject(b)]
top = [testObject(np.array([])), testObject(np.array([]))]
layer = py_recall_precision_layer.RecallPrecisionLayer()

#layer.reshape(bottom, top)
layer.setup(bottom, top)
layer.forward(bottom, top)
print top[0].data, top[1].data

