from pathlib import Path

from segm.data.base import BaseMMSeg
from segm.data import utils
from segm.config import dataset_dir

PASCAL_CONTEXT_CONFIG_PATH = Path(__file__).parent / "config" / "pascal_context.py"
PASCAL_CONTEXT_CATS_PATH = Path(__file__).parent / "config" / "pascal_context.yml"
PASCAL_VOC_CONFIG_PATH = Path(__file__).parent / "config" / "pascal_voc.py"
PASCAL_VOC_CATS_PATH = Path(__file__).parent / "config" / "pascal_voc.yml"


class PascalContextDataset(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, **kwargs):
        dataset_info = self._resolve_dataset_info()
        self.dataset_variant = dataset_info["variant"]
        self._config_data_root = dataset_info["config_data_root"]
        self._train_root = dataset_info["train_root"]
        self._val_root = dataset_info["val_root"]
        self._test_root = dataset_info["test_root"]
        self.ignore_label = dataset_info["ignore_label"]
        self.reduce_zero_label = dataset_info["reduce_zero_label"]
        self.n_cls = dataset_info["n_cls"]
        self._cats_path = dataset_info["cats_path"]

        print(
            f"Detected Pascal dataset variant '{self.dataset_variant}' "
            f"at {self._train_root}"
        )

        super().__init__(
            image_size, crop_size, split, dataset_info["config_path"], **kwargs
        )

        self.names, self.colors = utils.dataset_cat_description(self._cats_path)

    def _resolve_dataset_info(self):
        root_dir = Path(dataset_dir())

        context_bases = [
            root_dir / "pcontext",
            root_dir / "pascal_context",
            root_dir,
        ]
        for base in context_bases:
            data_root = base / "VOCdevkit" / "VOC2010"
            if (data_root / "SegmentationClassContext").exists():
                return dict(
                    variant="pascal_context_2010",
                    config_path=PASCAL_CONTEXT_CONFIG_PATH,
                    cats_path=PASCAL_CONTEXT_CATS_PATH,
                    config_data_root=base,
                    train_root=data_root,
                    val_root=data_root,
                    test_root=None,
                    n_cls=60,
                    ignore_label=255,
                    reduce_zero_label=False,
                )

        voc_bases = [
            root_dir / "pascal_context",
            root_dir / "pcontext",
            root_dir,
        ]
        for base in voc_bases:
            data_root = base / "VOC2012_train_val"
            if (data_root / "SegmentationClass").exists():
                test_root = base / "VOC2012_test"
                if not test_root.exists():
                    test_root = None
                return dict(
                    variant="pascal_voc_2012",
                    config_path=PASCAL_VOC_CONFIG_PATH,
                    cats_path=PASCAL_VOC_CATS_PATH,
                    config_data_root=data_root,
                    train_root=data_root,
                    val_root=data_root,
                    test_root=test_root,
                    n_cls=21,
                    ignore_label=255,
                    reduce_zero_label=False,
                )

            alt_root = base / "VOCdevkit" / "VOC2012"
            if (alt_root / "SegmentationClass").exists():
                test_root = alt_root if (alt_root / "SegmentationClass").exists() else None
                return dict(
                    variant="pascal_voc_2012",
                    config_path=PASCAL_VOC_CONFIG_PATH,
                    cats_path=PASCAL_VOC_CATS_PATH,
                    config_data_root=alt_root,
                    train_root=alt_root,
                    val_root=alt_root,
                    test_root=test_root,
                    n_cls=21,
                    ignore_label=255,
                    reduce_zero_label=False,
                )

        raise FileNotFoundError(
            "Pascal dataset not found. Expected either the Pascal Context 2010 layout "
            "with 'SegmentationClassContext' or the Pascal VOC 2012 layout with "
            "'SegmentationClass'. Make sure DATASET points to the directory "
            "containing the extracted files."
        )

    def update_default_config(self, config):
        config.data_root = self._config_data_root
        if self.split == "train":
            config.data.train.data_root = self._train_root
        elif self.split == "val":
            config.data.val.data_root = self._val_root
        elif self.split == "test":
            if self._test_root is None:
                raise ValueError(
                    "Test split is not available for the detected Pascal dataset."
                )
            config.data.test.data_root = self._test_root
        else:
            raise ValueError(f"Unknown split: {self.split}")

        config = super().update_default_config(config)
        return config

    def test_post_process(self, labels):
        return labels
