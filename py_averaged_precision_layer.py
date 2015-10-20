import caffe
import numpy as np

class APLayer():

    def setup(self, bottom, top):
        self.k = 3       #int(self.param_str)

        assert(len(bottom) == 2)
        assert(len(top) == 1)
        #Do I need to check the shape of ground turth and prediction result?

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        data_size = len(bottom[0].data)
        ap_lst = [0] * data_size

        ground_truth = bottom[0].data
        prediction = bottom[1].data
        for i in xrange(data_size):
            sample_gt = ground_truth[i]
            sample_pred = prediction[i]

            topk = set(np.argpartition(-sample_pred, self.k)[:self.k])
            rp_pairs = [[recall(), precision()] for j in xrange(k)]

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
        top[0].data = sum(ap_lst) / data_size

    def backward(self):
        pass

    def recallPrecision(sample, truth):
        true_positive, false_positive, false_negative = 0, 0, 0
        for j in xrange(len(truth)):
            if truth[j] == 1:
                if j in sample:
                    true_positive += 1
                else:
                    false_negative += 1
            elif j in sample:
                false_positive += 1
        precision = float(true_positive) / (true_positive + false_positive)
        recall = float(true_positive) / (true_positive + false_negative)
        return 
