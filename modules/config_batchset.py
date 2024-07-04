from torch.utils.data import DataLoader, Dataset, BatchSampler
import random
from collections import defaultdict
import numpy as np
import math

def count_batches_for_class(indices):
    """Calculate number of batches for given indices based on the provided logic"""
    if len(indices) <= 40:
        return 1
    elif len(indices) <= 80:
        return 2
    else:
        quotient, remainder = divmod(len(indices), 30)
        return quotient + (1 if remainder != 0 else 0)

def calculate_batches_per_class(class_to_indices):
    """Calculate number of batches for each class"""
    class_batches = {}
    for cls, indices in class_to_indices.items():
        class_batches[cls] = count_batches_for_class(indices)
    return class_batches




class BatchSamplerClass(BatchSampler):
    def __init__(self, dataset, num_positive = 6, num_negative = 2):
        self.dataset = dataset
        self.drop_last = False
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batched_indices = self.configuration_batches()
        self.num_batches = len(self.batched_indices)


    def configuration_batches(self):
        class_name = ["wall", "barrier", "shop", "store", "boat", "ship", "father", "dad", "sheep", "lamb",
                      "ocean", "sea", "pants", "trousers", "gift", "present", "cash", "money", "road", "street",
                      "car", "automobile", "test", "exam", "war", "score", "coat", "dead", "lab", "key",
                      "browsers", "presence", "cat", "stream", "card", "taste"]

        coi_list = [['barrier', 'wall', 'war'],
                    ['shop', 'store', 'score'],
                    ['ship', 'boat', 'coat'],
                    ['father', 'dad', 'dead'],
                    ['sheep', 'lamb', 'lab'],
                    ['ocean', 'sea', 'key'],
                    ['pants', 'trousers', 'browsers'],
                    ['gift', 'present', 'presence'],
                    ['money', 'cash', 'cat'],
                    ['road', 'street', 'stream'],
                    ['automobile', 'car', 'card'],
                    ['exam', 'test', 'taste']]

        coi_indices = [[class_name.index(item) for item in search_items if item in class_name] for search_items in
                       coi_list]

        dataset_size = len(self.dataset)
        all_indices = list(range(dataset_size))
        random.shuffle(all_indices)

        class_to_indices = defaultdict(list) # declaration class dict
        # dict key: class, value: index
        for idx in all_indices:
            label = self.dataset[idx][1]
            class_to_indices[label].append(idx)

        batched_indices = []
        # consist of batched_indices
        while len(batched_indices) < np.ceil(len(self.dataset) / (self.num_positive)).astype(int):
            batch = []
            classes = list(class_to_indices.keys()) # 0-36
            random.shuffle(classes)

            for cls in classes:
                contrastive_classes = [sublist for sublist in coi_indices if cls in sublist]
                num_samples = len(class_to_indices[cls])
                if num_samples == 0:
                    continue

                positive_indices = class_to_indices[cls][:self.num_positive]
                batch = positive_indices.copy()
                class_to_indices[cls] = class_to_indices[cls][self.num_positive:]

                for _cls, indices in class_to_indices.items():
                    if int(_cls) not in contrastive_classes[0]:
                        random.shuffle(indices)
                        batch.extend(indices[:self.num_negative])
                    else:
                        pass

                if batch == [] or len(batch) <= 1:
                    break
                batched_indices.append(batch)

        return batched_indices

    def __iter__(self):
        self.batched_indices = self.configuration_batches()
        for batch in self.batched_indices:
            yield batch

    def __len__(self):
        return self.num_batches



class ContrastiveAllPair(BatchSampler):
    def __init__(self, dataset, num_positive = 6, num_negative = 2):
        self.dataset = dataset
        # self.batch_size = _batch_size
        # self._batch_size = _batch_size
        self.drop_last = False
        # self.sampler = AllIndicesSampler(dataset)
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batched_indices = self.configuration_batches()
        self.num_batches = len(self.batched_indices)



    def configuration_batches(self):
        class_name = ["wall", "barrier", "shop", "store", "boat", "ship", "father", "dad", "sheep", "lamb",
                      "ocean", "sea", "pants", "trousers", "gift", "present", "cash", "money", "road", "street",
                      "car", "automobile", "test", "exam", "war", "score", "coat", "dead", "lab", "key",
                      "browsers", "presence", "cat", "stream", "card", "taste"]

        coi_list = [['barrier', 'wall', 'war'],
                    ['shop', 'store', 'score'],
                    ['ship', 'boat', 'coat'],
                    ['father', 'dad', 'dead'],
                    ['sheep', 'lamb', 'lab'],
                    ['ocean', 'sea', 'key'],
                    ['pants', 'trousers', 'browsers'],
                    ['gift', 'present', 'presence'],
                    ['money', 'cash', 'cat'],
                    ['road', 'street', 'stream'],
                    ['automobile', 'car', 'card'],
                    ['exam', 'test', 'taste']]

        coi_indices = [[class_name.index(item) for item in search_items if item in class_name] for search_items in
                       coi_list]

        dataset_size = len(self.dataset)
        all_indices = list(range(dataset_size))
        random.shuffle(all_indices)

        class_to_indices = defaultdict(list) # declaration class dict
        # dict key: class, value: index
        for idx in all_indices:
            label = self.dataset[idx][1]
            class_to_indices[label].append(idx)
        num_batches = math.comb(len(class_to_indices)-2,2)
        batched_indices = []
        # consist of batched_indices
        while len(batched_indices) < num_batches+1:
            batch = []
            classes = list(class_to_indices.keys()) # 0-36
            random.shuffle(classes)

            for pos_idx, cls in enumerate(classes):
                contrastive_classes = [sublist for sublist in coi_indices if cls in sublist]
                for neg_idx in range(pos_idx+1, len(classes)):
                    positive_indices = class_to_indices[cls]
                    if int(classes[neg_idx]) not in contrastive_classes[0]:
                        random.shuffle(class_to_indices[classes[neg_idx]])
                        negative_indices = class_to_indices[classes[neg_idx]]
                        batch.extend(positive_indices)
                        batch.extend(negative_indices)
                        batched_indices.append(batch)
                        batch = []
                    else:
                        pass

                # if batch == [] or len(batch) <= 1:
                #     break
                # batched_indices.append(batch)

        return batched_indices

    def __iter__(self):
        self.batched_indices = self.configuration_batches()
        for batch in self.batched_indices:
            yield batch

    def __len__(self):
        return self.num_batches



