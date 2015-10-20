import caffe
import numpy as np

class RecallPrecisionLayer():

    def setup(self, bottom, top):
        self.k = self.param_str       #directly define the value when testing

        assert(len(bottom) == 2)
        assert(len(top) == 2)
        assert(bottom[0].data.shape == bottom[1].data.shape)


    def reshape(self, bottom, top):
        top[0].reshape(1)
        top[1].reshape(1)

    def forward(self, bottom, top):
        data_size = len(bottom[0].data)
        recall_lst = [0] * data_size
        precision_lst = [0] * data_size

        ground_truth = bottom[0].data
        prediction = bottom[1].data
        for i in xrange(data_size):
            sample_gt = ground_truth[i]
            sample_pred = prediction[i]

            topk = set(np.argpartition(-sample_pred, self.k)[:self.k])
            true_positive, false_positive, false_negative = 0, 0, 0
            for j in xrange(len(sample_pred)):
                if sample_gt[j] == 1:
                    if j in topk:
                        true_positive += 1
                    else:
                        false_negative += 1
                elif j in topk:
                    false_positive += 1
            precision_lst[i] = float(true_positive) / (true_positive + false_positive)
            recall_lst[i] = float(true_positive) / (true_positive + false_negative)
        top[0].data[:] = sum(precision_lst) / data_size                     #delete'[:]' when testing
        top[1].data[:] = sum(recall_lst) / data_size

    def backward(self):
        pass
