# %% [markdown]
# # Visual Relationship Detection
#
# In this tutorial, we focus on the task of classifying visual relationships between objects in an image. For any given image, there might be many such relationships, defined formally as a `subject <predictate> object` (e.g. `person <riding> bike`). As an example, in the relationship `man riding bicycle`), "man" and "bicycle" are the subject and object, respectively, and "riding" is the relationship predicate.
#
# ![Visual Relationships](https://cs.stanford.edu/people/ranjaykrishna/vrd/dataset.png)
#
# In the examples of the relationships shown above, the red box represents the _subject_ while the green box represents the _object_. The _predicate_ (e.g. kick) denotes what relationship connects the subject and the object.
#
# For the purpose of this tutorial, we operate over the [Visual Relationship Detection (VRD) dataset](https://cs.stanford.edu/people/ranjaykrishna/vrd/) and focus on action relationships. We define our classification task as **identifying which of three relationships holds between the objects represented by a pair of bounding boxes.**

# %% tags=["md-exclude"]
import os

if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("visual_relation")

# %% [markdown]
# ### 1. Load Dataset
# We load the VRD dataset and filter images with at least one action predicate in it, since these are more difficult to classify than geometric relationships like `above` or `next to`. We load the train, valid, and test sets as Pandas `DataFrame` objects with the following fields:
# - `label`: The relationship between the objects. 0: `RIDE`, 1: `CARRY`, 2: `OTHER` action predicates
# - `object_bbox`: coordinates of the bounding box for the object `[ymin, ymax, xmin, xmax]`
# - `object_category`: category of the object
# - `source_img`: filename for the corresponding image the relationship is in
# - `subject_bbox`: coordinates of the bounding box for the object `[ymin, ymax, xmin, xmax]`
# - `subject_category`: category of the subject

# %% [markdown]
# If you are running this notebook for the first time, it will take ~15 mins to download all the required sample data.
#
# The sampled version of the dataset **uses the same 26 data points across the train, dev, and test sets.
# This setting is meant to demonstrate quickly how Snorkel works with this task, not to demonstrate performance.**
#
# The full version of the dataset **uses the same 635 samples for train, 216 samples for dev, and 194 samples for test sets and is still relatively small.**

# %%
from utils import load_vrd_data

# setting sample=False will take ~30 minutes to run (downloads full VRD dataset). make sure the data folder is empty before running
sample = False
is_test = os.environ.get("TRAVIS") == "true" or os.environ.get("IS_TEST") == "true"
df_train, df_valid, df_test = load_vrd_data(sample, is_test)

print("Train Relationships: ", len(df_train))
print("Dev Relationships: ", len(df_valid))
print("Test Relationships: ", len(df_test))

# %% [markdown]
# Note that the training `DataFrame` will have a labels field with all -1s. This denotes the lack of labels for that particular dataset. In this tutorial, we will assign probabilistic labels to the training set by writing labeling functions over attributes of the subject and objects!

# %% [markdown]
# ## 2. Writing Labeling Functions
# We now write labeling functions to detect what relationship exists between pairs of bounding boxes. To do so, we can encode various intuitions into the labeling functions:
#
# Here, we use [CLIP](https://arxiv.org/abs/2103.00020) model to provide context about the action happening in the image. We first crop the the bounding boxes of the object and subject from the image. We then copy them into a blank image to help the CLIP backbone focus only on the desired pair of subject and object. The similarity between text embedings of actions and embedding of the image provides a good proxy to label the sample.

# %%
# Define some Constants

RIDE = 0
CARRY = 1
OTHER = 2
ABSTAIN = -1

YMIN = 0
YMAX = 1
XMIN = 2
XMAX = 3

DIR = "data/VRD/sg_dataset/samples" if sample else "data/VRD/sg_dataset/sg_train_images"

# %% [markdown]
# We can use CLIP to provide context about the action happening in the image. We first crop the the bounding boxes of the object and subject from the image. We then copy them into a blank image to help the CLIP backbone focus only on the desired pair of subject and object. The similarity between text embedings of actions and embedding of the image provides a good proxy to label the sample.

# %%
from snorkel.labeling import labeling_function
from model import CLIPInference
import pandas as pd
from typing import List, Union

clip_model = CLIPInference()

def process_sample(x: pd.Series, actions: List[str], labels: List[int]) -> int:
    """Process a sample to determine its label based on visual and textual features."""
    if x.object_category != "person" and x.subject_category != "person":
        return OTHER

    # Extract and prepare bounding box coordinates
    object_corners = (x.object_bbox[XMIN], x.object_bbox[YMIN], x.object_bbox[XMAX], x.object_bbox[YMAX])
    subject_corners = (x.subject_bbox[XMIN], x.subject_bbox[YMIN], x.subject_bbox[XMAX], x.subject_bbox[YMAX])

    # Embedding text and preparing the image for model inference
    text_features = clip_model.embed_text(actions)
    processed_image = clip_model.prepare_image(f"{DIR}/{x.source_img}", object_corners, subject_corners)
    text_probs = clip_model.compute_similarity(processed_image, text_features)

    return clip_model.probs_to_label(text_probs, labels)


@labeling_function()
def lf_clip_carry(x: pd.Series) -> int:
    """Labeling function for determining if a person is carrying an object."""
    actions = ["The person is carrying a small object in their hand", "The person is sitting", "The person is sleeping"]
    labels = [CARRY, OTHER, OTHER]
    return process_sample(x, actions, labels)

@labeling_function()
def lf_clip_ride(x: pd.Series) -> int:
    """Labeling function for determining if a person is riding an object."""
    object_category = x.object_category if x.object_category != "person" else x.subject_category
    riding_objects = ["car", "train", "motorcycle", "bike", "boat", "van", "plane", "airplane", "skateboard", "horse", "skis", "surfboard", "snowboard"]
    if object_category not in riding_objects:
        return ABSTAIN
    actions = [f"A person {verb} a {object_category}" for verb in ["driving a vehicle", "sitting inside", "riding on", "steering", "flying", "walking", "pushing"]]
    labels = [RIDE if verb != "walking" else OTHER for verb in actions[:-2]] + [OTHER, CARRY, OTHER]
    return process_sample(x, actions, labels)

@labeling_function()
def lf_clip_wearing(x: pd.Series) -> int:
    """Labeling function for determining if a person is wearing an object."""
    object_category = x.object_category if x.object_category != "person" else x.subject_category
    wearing_objects = ["shirt", "glasses", "hat", "pants", "jacket", "shoe", "shoes", "helmet", "coat", "shorts", "jeans", "sunglasses", "tie", "watch"]
    if object_category not in wearing_objects:
        return ABSTAIN
    actions = [f"A person is {verb} the {object_category}" for verb in ["wearing", "carrying", "throwing"]]
    labels = [OTHER, CARRY, OTHER]
    return process_sample(x, actions, labels)

@labeling_function()
def lf_clip_sitting(x: pd.Series) -> int:
    """Labeling function for determining if a person is sitting on an object."""
    object_category = x.object_category if x.object_category != "person" else x.subject_category
    sitting_on_objects = ["chair", "bench", "sofa", "train", "tree"]
    if object_category not in sitting_on_objects:
        return ABSTAIN
    actions = [f"A person sitting on a {object_category}", f"A person carrying a {object_category}", f"A person pointing at a {object_category}"]
    labels = [OTHER, CARRY, OTHER]
    return process_sample(x, actions, labels)

@labeling_function()
def lf_clip_riding_bike(x: pd.Series) -> int:
    """Labeling function for determining if a person is riding a bike or motorcycle."""
    object_category = x.object_category if x.object_category != "person" else x.subject_category
    riding_objects = ["bike", "motorcycle"]
    if object_category not in riding_objects:
        return ABSTAIN
    actions = [f"A person riding the {object_category}", f"The {object_category} is parked"]
    labels = [RIDE, OTHER]
    return process_sample(x, actions, labels)


# %% [markdown]
# Note that the labeling functions have varying empirical accuracies and coverages. Due to class imbalance in our chosen relationships, labeling functions that label the `OTHER` class have higher coverage than labeling functions for `RIDE` or `CARRY`. This reflects the distribution of classes in the dataset as well.

# %% tags=["md-exclude-output"]
from snorkel.labeling import PandasLFApplier

lfs = [
    lf_clip_carry,
    lf_clip_ride,
    lf_clip_wearing,
    lf_clip_sitting,
    lf_clip_riding_bike,
]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_valid = applier.apply(df_valid)

# %%
from snorkel.labeling import LFAnalysis

Y_valid = df_valid.label.values
LFAnalysis(L_valid, lfs).lf_summary(Y_valid)

# %%
Y_train = df_train.label.values
LFAnalysis(L_train, lfs).lf_summary(Y_train)

# %% [markdown]
# ## 3. Train Label Model
# We now train a multi-class `LabelModel` to assign training labels to the unalabeled training set.

# %%
from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=3, verbose=True)
label_model.fit(L_train, seed=123, lr=0.01, log_freq=10, n_epochs=100)

# %% [markdown]
# We use [F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) Micro average for the multiclass setting, which calculates metrics globally across classes, by counting the total true positives, false negatives and false positives.

# %%
label_model.score(L_valid, Y_valid, metrics=["f1_micro"])

# %% [markdown]
# ## 4. Train a Classifier
# You can then use these training labels to train any standard discriminative model, such as [an off-the-shelf ResNet](https://github.com/KaimingHe/deep-residual-networks), which should learn to generalize beyond the LF's we've developed!

# %% [markdown]
# #### Create DataLoaders for Classifier

# %%
from snorkel.classification import DictDataLoader
from model import SceneGraphDataset, create_model

df_train["labels"] = label_model.predict(L_train)

# Remove rows where the predicted 'labels' value is ABSTAIN
df_train = df_train[df_train['labels'] != ABSTAIN]

if sample:
    TRAIN_DIR = "data/VRD/sg_dataset/samples"
else:
    TRAIN_DIR = "data/VRD/sg_dataset/sg_train_images"

dl_train = DictDataLoader(
    SceneGraphDataset("train_dataset", "train", TRAIN_DIR, df_train),
    batch_size=16,
    shuffle=True,
)

dl_valid = DictDataLoader(
    SceneGraphDataset("valid_dataset", "valid", TRAIN_DIR, df_valid),
    batch_size=16,
    shuffle=False,
)

# %% [markdown]
# #### Define Model Architecture

# %%
import torchvision.models as models

# initialize pretrained feature extractor
cnn = models.resnet18(pretrained=True)
model = create_model(cnn)

# %% [markdown]
# ### Train and Evaluate Model

# %% tags=["md-exclude-output"]
from snorkel.classification import Trainer

trainer = Trainer(
    seed = 123,
    n_epochs=1,  # increase for improved performance
    lr=1e-3,
    checkpointing=False,
)
trainer.fit(model, [dl_train])

# %%
model.score([dl_valid])

# %% [markdown]
# ## Recap
# We have successfully trained a visual relationship detection model! The addition of a model-based LF helped improve the accuracy of labling and consequently the accuracy of the trained model improved.
#
# We recently leveraged image-based and model-based LFs to [accelerate product tagging for Wayfair](https://snorkel.ai/how-wayfair-accelerated-product-tagging-automation-with-snorkel-flow/) using Snorkel Flow platform. You can read more on how Wayfair is using our technology in [this blog post](https://www.aboutwayfair.com/careers/tech-blog/accelerating-catalog-tagging-automation-with-snorkels-data-centric-ai-platform-wayfairs-success-story)!
