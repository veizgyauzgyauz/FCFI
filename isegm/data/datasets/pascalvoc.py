import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from isegm.utils.misc import get_labels_with_sizes
from isegm.data.base import ISDataset
from isegm.data.sample import DSample



class PascalVocDataset(ISDataset):
    def __init__(self, dataset_path, split='train', ignore_label=255, target_name_list=None, target_instance_id_list=None, **kwargs):
        super().__init__(**kwargs)
        assert split in {'train', 'val', 'trainval', 'test'}

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / "JPEGImages"
        self._insts_path = self.dataset_path / "SegmentationObjectLabel"
        self.dataset_split = split
        self.ignore_label = ignore_label

        if split == 'test':
            with open(self.dataset_path / f'ImageSets/Segmentation/test.pickle', 'rb') as f:
                self.dataset_samples, self.instance_ids = pkl.load(f)
        else:
            with open(self.dataset_path / f'ImageSets/Segmentation/{split}.txt', 'r') as f:
                self.dataset_samples = [name.strip() for name in f.readlines()]
        
        assert len(target_name_list) == len(target_instance_id_list)
        if target_name_list is None:
            self.dataset_samples = self.get_voc_images_and_ids_list()
        else:
            self.dataset_samples = []
            for target_name, target_instance_id in zip(target_name_list, target_instance_id_list):
                self.dataset_samples.append((target_name, target_instance_id))

    def get_voc_images_and_ids_list(self):
        pkl_path = self.dataset_path / f'{self.dataset_split}_images_and_ids_list.pkl'

        if pkl_path.exists():
            with open(str(pkl_path), 'rb') as fp:
                images_and_ids_list = pkl.load(fp)
        else:
            images_and_ids_list = []

            for sample in self.dataset_samples:
                inst_info_path = str(self._insts_path / f'{sample}.png')
                instances_mask = cv2.imread(inst_info_path)
                instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)
                instances_ids, _ = get_labels_with_sizes(instances_mask, self.ignore_label)

                for instances_id in instances_ids:
                    images_and_ids_list.append((sample, instances_id))

            with open(str(pkl_path), 'wb') as fp:
                pkl.dump(images_and_ids_list, fp)

        return images_and_ids_list

    def get_sample(self, index) -> DSample:

        image_name, instance_id = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.jpg')
        mask_path = str(self._insts_path / f'{image_name}.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0).astype(np.int32)
        instances_mask = np.zeros_like(mask)
        instances_mask[mask == instance_id] = instance_id
        instances_mask[mask == self.ignore_label] = self.ignore_label

        return DSample(image_name + '#%3d'%instance_id, image, instances_mask, objects_ids=[instance_id], ignore_ids=[self.ignore_label], sample_id=index)
