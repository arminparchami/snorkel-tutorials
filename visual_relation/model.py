import csv
from pathlib import Path
from typing import Tuple, List

import open_clip
import numpy as np
import pandas
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from snorkel.analysis import Scorer
from snorkel.classification import DictDataset, MultitaskClassifier, Operation, Task
from snorkel.classification.data import XDict, YDict


def union(bbox1, bbox2):
    """Create the union of the two bboxes.

    Parameters
    ----------
    bbox1
        Coordinates of first bounding box
    bbox2
        Coordinates of second bounding box

    Returns
    -------
    [y0, y1, x0, x1]
        Coordinates of union of input bounding boxes

    """
    y0 = min(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x0 = min(bbox1[2], bbox2[2])
    x1 = max(bbox1[3], bbox2[3])
    return [y0, y1, x0, x1]


def crop_img_arr(img_arr, bbox):
    """Crop bounding box from image.

    Parameters
    ----------
    img_arr
        Image in array format
    bbox
        Coordinates of bounding box to crop

    Returns
    -------
    img_arr
        Cropped image

    """
    return img_arr[bbox[0] : bbox[1], bbox[2] : bbox[3], :]


class SceneGraphDataset(DictDataset):
    """Dataloader for Scene Graph Dataset."""

    def __init__(
        self,
        name: str,
        split: str,
        image_dir: str,
        df: pandas.DataFrame,
        image_size=224,
    ) -> None:
        self.image_dir = Path(image_dir)
        X_dict = {
            "img_fn": df["source_img"].tolist(),
            "obj_bbox": df["object_bbox"].tolist(),
            "sub_bbox": df["subject_bbox"].tolist(),
            "obj_category": df["object_category"].tolist(),
            "sub_category": df["subject_category"].tolist(),
        }
        Y_dict = {
            "visual_relation_task": torch.LongTensor(df["label"].to_numpy())
        }  # change to take in the rounded train labels
        super(SceneGraphDataset, self).__init__(name, split, X_dict, Y_dict)

        # define standard set of transformations to apply to each image
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index: int) -> Tuple[XDict, YDict]:
        img_fn = self.X_dict["img_fn"][index]
        img_arr = np.array(Image.open(self.image_dir / img_fn))

        obj_bbox = self.X_dict["obj_bbox"][index]
        sub_bbox = self.X_dict["sub_bbox"][index]
        obj_category = self.X_dict["obj_category"][index]
        sub_category = self.X_dict["sub_category"][index]

        # compute crops
        obj_crop = crop_img_arr(img_arr, obj_bbox)
        sub_crop = crop_img_arr(img_arr, sub_bbox)
        union_crop = crop_img_arr(img_arr, union(obj_bbox, sub_bbox))

        # transform each crop
        x_dict = {
            "obj_crop": self.transform(Image.fromarray(obj_crop)),
            "sub_crop": self.transform(Image.fromarray(sub_crop)),
            "union_crop": self.transform(Image.fromarray(union_crop)),
            "obj_category": obj_category,
            "sub_category": sub_category,
        }

        y_dict = {name: label[index] for name, label in self.Y_dict.items()}
        return x_dict, y_dict

    def __len__(self):
        return len(self.X_dict["img_fn"])


class WordEmb(nn.Module):
    """Extract and concat word embeddings for obj and sub categories."""

    def __init__(self, glove_fn="data/glove/glove.6B.100d.txt"):
        super(WordEmb, self).__init__()

        self.word_embs = pandas.read_csv(
            glove_fn, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE
        )

    def _get_wordvec(self, word):
        return self.word_embs.loc[word].values

    def forward(self, obj_category, sub_category):
        obj_emb = self._get_wordvec(obj_category)
        sub_emb = self._get_wordvec(sub_category)
        embs = np.concatenate([obj_emb, sub_emb], axis=1)
        return torch.FloatTensor(embs)


# Classes and helper functions for defining classifier
def init_fc(fc):
    torch.nn.init.xavier_uniform_(fc.weight)
    fc.bias.data.fill_(0.01)


class FlatConcat(nn.Module):
    """Module that flattens and concatenates features"""

    def forward(self, *inputs):
        return torch.cat([input.view(input.size(0), -1) for input in inputs], dim=1)


# Helper functions to geenerate operations
def get_op_sequence():
    # define feature extractors for each of the (union, subject, and object) image crops
    union_feat_op = Operation(
        name="union_feat_op",
        module_name="feat_extractor",
        inputs=[("_input_", "union_crop")],
    )

    sub_feat_op = Operation(
        name="sub_feat_op",
        module_name="feat_extractor",
        inputs=[("_input_", "sub_crop")],
    )

    obj_feat_op = Operation(
        name="obj_feat_op",
        module_name="feat_extractor",
        inputs=[("_input_", "obj_crop")],
    )

    # define an operation to extract word embeddings for subject and object categories
    word_emb_op = Operation(
        name="word_emb_op",
        module_name="word_emb",
        inputs=[("_input_", "sub_category"), ("_input_", "obj_category")],
    )

    # define an operation to concatenate image features and word embeddings
    concat_op = Operation(
        name="concat_op",
        module_name="feat_concat",
        inputs=["obj_feat_op", "sub_feat_op", "union_feat_op", "word_emb_op"],
    )

    # define an operation to make a prediction over all concatenated features
    prediction_op = Operation(
        name="head_op", module_name="prediction_head", inputs=["concat_op"]
    )

    return [
        sub_feat_op,
        obj_feat_op,
        union_feat_op,
        word_emb_op,
        concat_op,
        prediction_op,
    ]


# Create model from pre loaded resnet cnn.
def create_model(resnet_cnn):
    # freeze the resnet weights
    for param in resnet_cnn.parameters():
        param.requires_grad = False

    # define input features
    in_features = resnet_cnn.fc.in_features
    feature_extractor = nn.Sequential(*list(resnet_cnn.children())[:-1])

    # initialize FC layer: maps 3 sets of image features to class logits
    WEMB_SIZE = 100
    fc = nn.Linear(in_features * 3 + 2 * WEMB_SIZE, 3)
    init_fc(fc)

    # define layers
    module_pool = nn.ModuleDict(
        {
            "feat_extractor": feature_extractor,
            "prediction_head": fc,
            "feat_concat": FlatConcat(),
            "word_emb": WordEmb(),
        }
    )

    # define task flow through modules
    op_sequence = get_op_sequence()
    pred_cls_task = Task(
        name="visual_relation_task",
        module_pool=module_pool,
        op_sequence=op_sequence,
        scorer=Scorer(metrics=["f1_micro"]),
    )
    return MultitaskClassifier([pred_cls_task])


class CLIPInference:
    """A class for CLIP model inference with methods for embedding text, preparing images, computing similarity, and converting probabilities to labels."""

    def __init__(
        self, backbone: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"
    ):
        """
        Initializes the CLIP model with specified backbone and pretrained weights.

        :param backbone: Model architecture.
        :param pretrained: Pretrained weight set.
        """
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            backbone, pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(backbone)

    def embed_text(self, actions: List[str]) -> torch.Tensor:
        """
        Compute text embeddings for a list of action descriptions.

        :param actions: List of text descriptions.
        :return: Normalized text feature tensor.
        """
        text = self.tokenizer(actions)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def prepare_image(
        self,
        image_path: str,
        object_corners: Tuple[int, int, int, int],
        subject_corners: Tuple[int, int, int, int],
    ) -> Image:
        """
        Prepare an image by cropping to the specified object and subject bounding boxes and combining them into one image.

        :param image_path: Path to the image file.
        :param object_corners: Bounding box corners for the object (x_min, y_min, x_max, y_max).
        :param subject_corners: Bounding box corners for the subject (x_min, y_min, x_max, y_max).
        :return: Processed PIL Image object.
        """
        with Image.open(image_path) as img:
            img.load()
            processed_image = Image.new("RGB", img.size, (0, 0, 0))
            cropped_object = img.crop(object_corners)
            cropped_subject = img.crop(subject_corners)
            processed_image.paste(cropped_object, object_corners[:2])
            processed_image.paste(cropped_subject, subject_corners[:2])
        return processed_image

    def compute_similarity(
        self, image: Image, text_features: torch.Tensor
    ) -> np.ndarray:
        """
        Compute the similarity between an image and text features using the CLIP model.

        :param image: PIL Image object to be processed.
        :param text_features: Text feature tensor.
        :return: Probability array derived from the softmax of similarities.
        """
        image_tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_probs = (
                (100.0 * image_features @ text_features.T)
                .softmax(dim=-1)
                .cpu()
                .detach()
                .numpy()[0]
            )
        return text_probs

    def probs_to_label(self, text_probs: np.ndarray, labels: List[int]) -> int:
        """
        Convert probability array to a label, based on a threshold.

        :param text_probs: Array of probabilities.
        :param labels: List of possible labels.
        :return: The chosen label, or -1 if no label meets the threshold.
        """
        max_index = np.argmax(text_probs)
        if text_probs[max_index] < 0.4:
            return -1
        return labels[max_index]
