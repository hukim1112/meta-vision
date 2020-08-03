import numpy as np


class Episode_generator(object):
    def __init__(self, data, n_way, n_support, n_query):
        self.data = data
        self.n_classes = data.shape[0]
        self.n_examples = data.shape[1]
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        print("episode generator initialiation")

    def get_next_episode(self):
        support = np.zeros([self.n_way, self.n_support,
                            84, 84, 3], dtype=np.float32)
        query = np.zeros(
            [self.n_way, self.n_query, 84, 84, 3], dtype=np.float32)
        classes_ep = np.random.permutation(self.n_classes)[:self.n_way]

        for i, i_class in enumerate(classes_ep):
            selected = np.random.permutation(self.n_examples)[
                :self.n_support + self.n_query]
            support[i] = self.data[i_class, selected[:self.n_support]]
            query[i] = self.data[i_class, selected[self.n_support:]]

        return support, query
