import pandas as pd

class ExemplarSet:
    def __init__(self, total_size, selector, 
                 mode='equal', max_subsets=10):
        self.mode = mode

        self.total_size = total_size
        self.subset_size = self.total_size

        self.selector = selector
        self.subset_ratios = []
        self.subsets = []
        self.max_subsets = max_subsets
    
    def sample_exemplars(self, new_dataset):
        # drop oldest subset if limit is reached
        if self.max_subsets <= len(self.subsets):
            self.subsets.pop(0)

        self._update_subset_size_and_ratios()

        # downsample all existing subsets to their new size
        for i in range(len(self.subsets)):
            ratio = self.subset_ratios[i]
            size = self.subset_size * ratio
            self.subsets[i] = self._sample(self.subsets[i], size)

        # sample new exemplars
        samples = self._sample(new_dataset)
        self.subsets.append(samples)

    def _sample(self, dataset, sample_size=None):
        if not sample_size:
            sample_size = self.subset_size
        return self.selector(dataset, 
                             min(len(dataset), round(sample_size)))  

    def _update_subset_size_and_ratios(self):
        # all subsets are of equal size
        if self.mode == 'equal':
            self.subset_size = self.total_size / (len(self.subsets) + 1)
            self.subset_ratios = [1 for _ in range(len(self.subsets))]

        # more samples are kept from newer datasets
        elif self.mode == 'ratios':
            l = len(self.subsets) + 1
            r = 1 / l
            self.subset_ratios = [i * r for i in range(1, l + 1)]
            self.subset_size = self.total_size / sum(self.subset_ratios)

    def get_exemplar_set(self):
        if len(self.subsets):
            return pd.concat(self.subsets, ignore_index=True)
        else:
            return None

