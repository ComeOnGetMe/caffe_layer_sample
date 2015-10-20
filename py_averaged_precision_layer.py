import caffe
import numpy as np

class APLayer():

    def setup(self, bottom, top):
        self.k = int(self.param_str)              # Assign value when testing

        assert(len(bottom) == 2)

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

            topk = np.argpartition(-sample_pred, self.k)[:self.k]
            topk = sorted(topk, reverse = True, key = lambda x: sample_pred[x])
            rp_pairs = [self.recallPrecision(set(topk[:j]), sample_gt) for j in xrange(1, self.k + 1)]
            rp_pairs.sort(key = lambda x:x[0])
            ap = rp_pairs[0][1] * rp_pairs[0][0]
            for j in xrange(1, self.k):
                ap += (rp_pairs[j - 1][1] + rp_pairs[j][1]) * (rp_pairs[j][0] - rp_pairs[j - 1][0]) / 2
            ap_lst[i] = ap
        top[0].data[:] = sum(ap_lst) / data_size            # Remove '[:]' when testing

    def backward(self):
        pass

    def recallPrecision(self, prediction, truth):
        true_positive, false_positive, false_negative = 0, 0, 0
        for j in xrange(len(truth)):
            if truth[j] == 1:
                if j in prediction:
                    true_positive += 1
                else:
                    false_negative += 1
            elif j in prediction:
                false_positive += 1
        precision = float(true_positive) / (true_positive + false_positive)
        recall = float(true_positive) / (true_positive + false_negative)
        return [recall, precision]
