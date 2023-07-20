from .grabcut import GrabCutDataset


class BerkeleyDataset(GrabCutDataset):
    def __init__(self, dataset_path, **kwargs):
        super().__init__(dataset_path, images_dir_name='images-f-BRS/test', masks_dir_name='masks-f-BRS/test', **kwargs)
