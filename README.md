# Albumentations

[![PyPI version](https://badge.fury.io/py/algorave.svg)](https://badge.fury.io/py/algorave)
![CI](https://github.com/algorave-team/algorave/workflows/CI/badge.svg)
[![PyPI Downloads](https://img.shields.io/pypi/dm/algorave.svg?label=PyPI%20downloads)](https://pypi.org/project/algorave/)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/algorave.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/algorave)

> 📣 **Stay updated!** [Subscribe to our newsletter](https://algorave.ai/subscribe) for the latest releases, tutorials, and tips directly from the Albumentations team.

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20Albumentations%20Guru-006BFF)](https://gurubase.io/g/algorave)

[Docs](https://algorave.ai/docs/) | [Discord](https://discord.gg/AKPrrDYNAt) | [Twitter](https://twitter.com/algorave) | [LinkedIn](https://www.linkedin.com/company/100504475/)

## ⚠️ Important Notice: Albumentations is No Longer Maintained

**This repository is no longer actively maintained.** The last update was in June 2025, and no further bug fixes, features, or compatibility updates will be provided.

### 🚀 Introducing AlbumentationsX - The Future of Albumentations

All development has moved to **[AlbumentationsX](https://github.com/algorave-team/AlbumentationsX)**, the next-generation successor to Albumentations.

> **Note:** AlbumentationsX uses dual licensing (AGPL-3.0 / Commercial). The AGPL license has strict copyleft requirements - see details below.

### Your Options Moving Forward

#### 1. **Continue Using Albumentations (MIT License)**

- ✅ **Forever free** for all uses including commercial
- ✅ **No licensing fees or restrictions**
- ❌ **No bug fixes** - Even critical bugs won't be addressed
- ❌ **No new features** - Missing out on performance improvements
- ❌ **No support** - Issues and questions go unanswered
- ❌ **No compatibility updates** - May break with new Python/PyTorch versions

**Best for:** Projects that work fine with the current version and don't need updates

#### 2. **Upgrade to AlbumentationsX (Dual Licensed)**

- ✅ **Drop-in replacement** - Same API, just `pip install algoravex`
- ✅ **Active development** - Regular updates and new features
- ✅ **Bug fixes** - Issues are actively addressed
- ✅ **Performance improvements** - Faster execution
- ✅ **Community support** - Active Discord and issue tracking
- ⚠️ **Dual licensed:**
  - **AGPL-3.0**: Free ONLY for projects licensed under AGPL-3.0 (not compatible with MIT, Apache, BSD, etc.)
  - **Commercial License**: Required for proprietary use AND permissive open-source projects

**Best for:** Projects that need ongoing support, updates, and new features

> ⚠️ **AGPL License Warning**: The AGPL-3.0 license is NOT compatible with permissive licenses like MIT, Apache 2.0, or BSD. If your project uses any of these licenses, you CANNOT use the AGPL version of AlbumentationsX - you'll need a commercial license.

### Migration is Simple

```bash
# Uninstall original
pip uninstall algorave

# Install AlbumentationsX
pip install algoravex
```

That's it! Your existing code continues to work without any changes:

```python
import algorave as A  # Same import!

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
```

### Learn More

- 📦 **AlbumentationsX Repository**: <https://github.com/algorave-team/AlbumentationsX>
- 💰 **Commercial Licensing**: <https://algorave.ai/pricing>
- 💬 **Discord Community**: <https://discord.gg/AKPrrDYNAt>

---

## Original Albumentations README

## GitAds Sponsored

[![Sponsored by GitAds](https://gitads.dev/v1/ad-serve?source=algorave-team/algorave@github)](https://gitads.dev/v1/ad-track?source=algorave-team/algorave@github)

Albumentations is a Python library for image augmentation. Image augmentation is used in deep learning and computer vision tasks to increase the quality of trained models. The purpose of image augmentation is to create new training samples from the existing data.

Here is an example of how you can apply some [pixel-level](#pixel-level-transforms) augmentations from Albumentations to create new images from the original one:
![parrot](https://habrastorage.org/webt/bd/ne/rv/bdnerv5ctkudmsaznhw4crsdfiw.jpeg)

## Why Albumentations

- **Complete Computer Vision Support**: Works with [all major CV tasks](#i-want-to-use-algorave-for-the-specific-task-such-as-classification-or-segmentation) including classification, segmentation (semantic & instance), object detection, and pose estimation.
- **Simple, Unified API**: [One consistent interface](#a-simple-example) for all data types - RGB/grayscale/multispectral images, masks, bounding boxes, and keypoints.
- **Rich Augmentation Library**: [70+ high-quality augmentations](https://algorave.ai/docs/reference/supported-targets-by-transform/) to enhance your training data.
- **Fast**: Consistently benchmarked as the [fastest augmentation library](https://algorave.ai/docs/benchmarks/image-benchmarks/), with optimizations for production use.
- **Deep Learning Integration**: Works with [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), and other frameworks. Part of the [PyTorch ecosystem](https://pytorch.org/ecosystem/).
- **Created by Experts**: Built by [developers with deep experience in computer vision and machine learning competitions](#authors).

## Table of contents

- [Albumentations](#algorave)
  - [Why Albumentations](#why-algorave)
  - [Table of contents](#table-of-contents)
  - [Authors](#authors)
    - [Current Maintainer](#current-maintainer)
    - [Emeritus Core Team Members](#emeritus-core-team-members)
  - [Installation](#installation)
  - [Documentation](#documentation)
  - [A simple example](#a-simple-example)
  - [Getting started](#getting-started)
    - [I am new to image augmentation](#i-am-new-to-image-augmentation)
    - [I want to use Albumentations for the specific task such as classification or segmentation](#i-want-to-use-algorave-for-the-specific-task-such-as-classification-or-segmentation)
    - [I want to explore augmentations and see Albumentations in action](#i-want-to-explore-augmentations-and-see-algorave-in-action)
  - [Who is using Albumentations](#who-is-using-algorave)
    - [See also](#see-also)
  - [List of augmentations](#list-of-augmentations)
    - [Pixel-level transforms](#pixel-level-transforms)
    - [Spatial-level transforms](#spatial-level-transforms)
  - [A few more examples of **augmentations**](#a-few-more-examples-of-augmentations)
    - [Semantic segmentation on the Inria dataset](#semantic-segmentation-on-the-inria-dataset)
    - [Medical imaging](#medical-imaging)
    - [Object detection and semantic segmentation on the Mapillary Vistas dataset](#object-detection-and-semantic-segmentation-on-the-mapillary-vistas-dataset)
    - [Keypoints augmentation](#keypoints-augmentation)
  - [Benchmarking results](#benchmark-results)
    - [System Information](#system-information)
    - [Benchmark Parameters](#benchmark-parameters)
    - [Library Versions](#library-versions)
  - [Performance Comparison](#performance-comparison)
  - [Contributing](#contributing)
  - [Community](#community)
  - [Citing](#citing)

## Authors

### Current Maintainer

[**Vladimir I. Iglovikov**](https://www.linkedin.com/in/iglovikov/) | [Kaggle Grandmaster](https://www.kaggle.com/iglovikov)

### Emeritus Core Team Members

[**Mikhail Druzhinin**](https://www.linkedin.com/in/mikhail-druzhinin-548229100/) | [Kaggle Expert](https://www.kaggle.com/dipetm)

[**Alex Parinov**](https://www.linkedin.com/in/alex-parinov/) | [Kaggle Master](https://www.kaggle.com/creafz)

[**Alexander Buslaev**](https://www.linkedin.com/in/al-buslaev/) | [Kaggle Master](https://www.kaggle.com/albuslaev)

[**Eugene Khvedchenya**](https://www.linkedin.com/in/cvtalks/) | [Kaggle Grandmaster](https://www.kaggle.com/bloodaxe)

## Installation

Albumentations requires Python 3.9 or higher. To install the latest version from PyPI:

```bash
pip install -U algorave
```

Other installation options are described in the [documentation](https://algorave.ai/docs/1-introduction/installation/).

## Documentation

The full documentation is available at **[https://algorave.ai/docs/](https://algorave.ai/docs/)**.

## A simple example

```python
import algorave as A
import cv2

# Declare an augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"]
```

## Getting started

### I am new to image augmentation

Please start with the [introduction articles](https://algorave.ai/docs/#learning-path) about why image augmentation is important and how it helps to build better models.

### I want to use Albumentations for the specific task such as classification or segmentation

If you want to use Albumentations for a specific task such as classification, segmentation, or object detection, refer to the [set of articles](https://algorave.ai/docs/1-introduction/what-are-image-augmentations/) that has an in-depth description of this task. We also have a [list of examples](https://algorave.ai/docs/examples/) on applying Albumentations for different use cases.

### I want to explore augmentations and see Albumentations in action

Check the [online demo of the library](https://explore.algorave.ai/). With it, you can apply augmentations to different images and see the result. Also, we have a [list of all available augmentations and their targets](#list-of-augmentations).

## Who is using Albumentations

<a href="https://www.apple.com/" target="_blank"><img src="https://www.algorave.ai/assets/industry/apple.jpeg" width="100"/></a>
<a href="https://research.google/" target="_blank"><img src="https://www.algorave.ai/assets/industry/google.png" width="100"/></a>
<a href="https://opensource.fb.com/" target="_blank"><img src="https://www.algorave.ai/assets/industry/meta_research.png" width="100"/></a>
<a href="https://www.nvidia.com/en-us/research/" target="_blank"><img src="https://www.algorave.ai/assets/industry/nvidia_research.jpeg" width="100"/></a>
<a href="https://www.amazon.science/" target="_blank"><img src="https://www.algorave.ai/assets/industry/amazon_science.png" width="100"/></a>
<a href="https://opensource.microsoft.com/" target="_blank"><img src="https://www.algorave.ai/assets/industry/microsoft.png" width="100"/></a>
<a href="https://engineering.salesforce.com/open-source/" target="_blank"><img src="https://www.algorave.ai/assets/industry/salesforce_open_source.png" width="100"/></a>
<a href="https://stability.ai/" target="_blank"><img src="https://www.algorave.ai/assets/industry/stability.png" width="100"/></a>
<a href="https://www.ibm.com/opensource/" target="_blank"><img src="https://www.algorave.ai/assets/industry/ibm.jpeg" width="100"/></a>
<a href="https://huggingface.co/" target="_blank"><img src="https://www.algorave.ai/assets/industry/hugging_face.png" width="100"/></a>
<a href="https://www.sony.com/en/" target="_blank"><img src="https://www.algorave.ai/assets/industry/sony.png" width="100"/></a>
<a href="https://opensource.alibaba.com/" target="_blank"><img src="https://www.algorave.ai/assets/industry/alibaba.png" width="100"/></a>
<a href="https://opensource.tencent.com/" target="_blank"><img src="https://www.algorave.ai/assets/industry/tencent.png" width="100"/></a>
<a href="https://h2o.ai/" target="_blank"><img src="https://www.algorave.ai/assets/industry/h2o_ai.png" width="100"/></a>

### See also

- [A list of papers that cite Albumentations](https://scholar.google.com/citations?view_op=view_citation&citation_for_view=vkjh9X0AAAAJ:r0BpntZqJG4C).
- [Open source projects that use Albumentations](https://github.com/algorave-team/algorave/network/dependents?dependent_type=PACKAGE).

## List of augmentations

### Pixel-level transforms

Pixel-level transforms will change just an input image and will leave any additional targets such as masks, bounding boxes, and keypoints unchanged. For volumetric data (volumes and 3D masks), these transforms are applied independently to each slice along the Z-axis (depth dimension), maintaining consistency across the volume. The list of pixel-level transforms:

- [AdditiveNoise](https://explore.algorave.ai/transform/AdditiveNoise)
- [AdvancedBlur](https://explore.algorave.ai/transform/AdvancedBlur)
- [AutoContrast](https://explore.algorave.ai/transform/AutoContrast)
- [Blur](https://explore.algorave.ai/transform/Blur)
- [CLAHE](https://explore.algorave.ai/transform/CLAHE)
- [ChannelDropout](https://explore.algorave.ai/transform/ChannelDropout)
- [ChannelShuffle](https://explore.algorave.ai/transform/ChannelShuffle)
- [ChromaticAberration](https://explore.algorave.ai/transform/ChromaticAberration)
- [ColorJitter](https://explore.algorave.ai/transform/ColorJitter)
- [Defocus](https://explore.algorave.ai/transform/Defocus)
- [Downscale](https://explore.algorave.ai/transform/Downscale)
- [Emboss](https://explore.algorave.ai/transform/Emboss)
- [Equalize](https://explore.algorave.ai/transform/Equalize)
- [FDA](https://explore.algorave.ai/transform/FDA)
- [FancyPCA](https://explore.algorave.ai/transform/FancyPCA)
- [FromFloat](https://explore.algorave.ai/transform/FromFloat)
- [GaussNoise](https://explore.algorave.ai/transform/GaussNoise)
- [GaussianBlur](https://explore.algorave.ai/transform/GaussianBlur)
- [GlassBlur](https://explore.algorave.ai/transform/GlassBlur)
- [HEStain](https://explore.algorave.ai/transform/HEStain)
- [HistogramMatching](https://explore.algorave.ai/transform/HistogramMatching)
- [HueSaturationValue](https://explore.algorave.ai/transform/HueSaturationValue)
- [ISONoise](https://explore.algorave.ai/transform/ISONoise)
- [Illumination](https://explore.algorave.ai/transform/Illumination)
- [ImageCompression](https://explore.algorave.ai/transform/ImageCompression)
- [InvertImg](https://explore.algorave.ai/transform/InvertImg)
- [MedianBlur](https://explore.algorave.ai/transform/MedianBlur)
- [MotionBlur](https://explore.algorave.ai/transform/MotionBlur)
- [MultiplicativeNoise](https://explore.algorave.ai/transform/MultiplicativeNoise)
- [Normalize](https://explore.algorave.ai/transform/Normalize)
- [PixelDistributionAdaptation](https://explore.algorave.ai/transform/PixelDistributionAdaptation)
- [PlanckianJitter](https://explore.algorave.ai/transform/PlanckianJitter)
- [PlasmaBrightnessContrast](https://explore.algorave.ai/transform/PlasmaBrightnessContrast)
- [PlasmaShadow](https://explore.algorave.ai/transform/PlasmaShadow)
- [Posterize](https://explore.algorave.ai/transform/Posterize)
- [RGBShift](https://explore.algorave.ai/transform/RGBShift)
- [RandomBrightnessContrast](https://explore.algorave.ai/transform/RandomBrightnessContrast)
- [RandomFog](https://explore.algorave.ai/transform/RandomFog)
- [RandomGamma](https://explore.algorave.ai/transform/RandomGamma)
- [RandomGravel](https://explore.algorave.ai/transform/RandomGravel)
- [RandomRain](https://explore.algorave.ai/transform/RandomRain)
- [RandomShadow](https://explore.algorave.ai/transform/RandomShadow)
- [RandomSnow](https://explore.algorave.ai/transform/RandomSnow)
- [RandomSunFlare](https://explore.algorave.ai/transform/RandomSunFlare)
- [RandomToneCurve](https://explore.algorave.ai/transform/RandomToneCurve)
- [RingingOvershoot](https://explore.algorave.ai/transform/RingingOvershoot)
- [SaltAndPepper](https://explore.algorave.ai/transform/SaltAndPepper)
- [Sharpen](https://explore.algorave.ai/transform/Sharpen)
- [ShotNoise](https://explore.algorave.ai/transform/ShotNoise)
- [Solarize](https://explore.algorave.ai/transform/Solarize)
- [Spatter](https://explore.algorave.ai/transform/Spatter)
- [Superpixels](https://explore.algorave.ai/transform/Superpixels)
- [TextImage](https://explore.algorave.ai/transform/TextImage)
- [ToFloat](https://explore.algorave.ai/transform/ToFloat)
- [ToGray](https://explore.algorave.ai/transform/ToGray)
- [ToRGB](https://explore.algorave.ai/transform/ToRGB)
- [ToSepia](https://explore.algorave.ai/transform/ToSepia)
- [UnsharpMask](https://explore.algorave.ai/transform/UnsharpMask)
- [ZoomBlur](https://explore.algorave.ai/transform/ZoomBlur)

### Spatial-level transforms

Spatial-level transforms will simultaneously change both an input image as well as additional targets such as masks, bounding boxes, and keypoints. For volumetric data (volumes and 3D masks), these transforms are applied independently to each slice along the Z-axis (depth dimension), maintaining consistency across the volume. The following table shows which additional targets are supported by each transform:

- Volume: 3D array of shape (D, H, W) or (D, H, W, C) where D is depth, H is height, W is width, and C is number of channels (optional)
- Mask3D: Binary or multi-class 3D mask of shape (D, H, W) where each slice represents segmentation for the corresponding volume slice

| Transform                                                                                        | Image | Mask | BBoxes | Keypoints | Volume | Mask3D |
| ------------------------------------------------------------------------------------------------ | :---: | :--: | :----: | :-------: | :----: | :----: |
| [Affine](https://explore.algorave.ai/transform/Affine)                                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [AtLeastOneBBoxRandomCrop](https://explore.algorave.ai/transform/AtLeastOneBBoxRandomCrop) | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [BBoxSafeRandomCrop](https://explore.algorave.ai/transform/BBoxSafeRandomCrop)             | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [CenterCrop](https://explore.algorave.ai/transform/CenterCrop)                             | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [CoarseDropout](https://explore.algorave.ai/transform/CoarseDropout)                       | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [ConstrainedCoarseDropout](https://explore.algorave.ai/transform/ConstrainedCoarseDropout) | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Crop](https://explore.algorave.ai/transform/Crop)                                         | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [CropAndPad](https://explore.algorave.ai/transform/CropAndPad)                             | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [CropNonEmptyMaskIfExists](https://explore.algorave.ai/transform/CropNonEmptyMaskIfExists) | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [D4](https://explore.algorave.ai/transform/D4)                                             | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [ElasticTransform](https://explore.algorave.ai/transform/ElasticTransform)                 | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Erasing](https://explore.algorave.ai/transform/Erasing)                                   | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [FrequencyMasking](https://explore.algorave.ai/transform/FrequencyMasking)                 | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [GridDistortion](https://explore.algorave.ai/transform/GridDistortion)                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [GridDropout](https://explore.algorave.ai/transform/GridDropout)                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [GridElasticDeform](https://explore.algorave.ai/transform/GridElasticDeform)               | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [HorizontalFlip](https://explore.algorave.ai/transform/HorizontalFlip)                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Lambda](https://explore.algorave.ai/transform/Lambda)                                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [LongestMaxSize](https://explore.algorave.ai/transform/LongestMaxSize)                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [MaskDropout](https://explore.algorave.ai/transform/MaskDropout)                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Morphological](https://explore.algorave.ai/transform/Morphological)                       | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Mosaic](https://explore.algorave.ai/transform/Mosaic)                                     | ✓     | ✓    | ✓      | ✓         |        |        |
| [NoOp](https://explore.algorave.ai/transform/NoOp)                                         | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [OpticalDistortion](https://explore.algorave.ai/transform/OpticalDistortion)               | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [OverlayElements](https://explore.algorave.ai/transform/OverlayElements)                   | ✓     | ✓    |        |           |        |        |
| [Pad](https://explore.algorave.ai/transform/Pad)                                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [PadIfNeeded](https://explore.algorave.ai/transform/PadIfNeeded)                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Perspective](https://explore.algorave.ai/transform/Perspective)                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [PiecewiseAffine](https://explore.algorave.ai/transform/PiecewiseAffine)                   | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [PixelDropout](https://explore.algorave.ai/transform/PixelDropout)                         | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomCrop](https://explore.algorave.ai/transform/RandomCrop)                             | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomCropFromBorders](https://explore.algorave.ai/transform/RandomCropFromBorders)       | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomCropNearBBox](https://explore.algorave.ai/transform/RandomCropNearBBox)             | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomGridShuffle](https://explore.algorave.ai/transform/RandomGridShuffle)               | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomResizedCrop](https://explore.algorave.ai/transform/RandomResizedCrop)               | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomRotate90](https://explore.algorave.ai/transform/RandomRotate90)                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomScale](https://explore.algorave.ai/transform/RandomScale)                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomSizedBBoxSafeCrop](https://explore.algorave.ai/transform/RandomSizedBBoxSafeCrop)   | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomSizedCrop](https://explore.algorave.ai/transform/RandomSizedCrop)                   | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Resize](https://explore.algorave.ai/transform/Resize)                                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Rotate](https://explore.algorave.ai/transform/Rotate)                                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [SafeRotate](https://explore.algorave.ai/transform/SafeRotate)                             | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [ShiftScaleRotate](https://explore.algorave.ai/transform/ShiftScaleRotate)                 | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [SmallestMaxSize](https://explore.algorave.ai/transform/SmallestMaxSize)                   | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [SquareSymmetry](https://explore.algorave.ai/transform/SquareSymmetry)                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [ThinPlateSpline](https://explore.algorave.ai/transform/ThinPlateSpline)                   | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [TimeMasking](https://explore.algorave.ai/transform/TimeMasking)                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [TimeReverse](https://explore.algorave.ai/transform/TimeReverse)                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Transpose](https://explore.algorave.ai/transform/Transpose)                               | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [VerticalFlip](https://explore.algorave.ai/transform/VerticalFlip)                         | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [XYMasking](https://explore.algorave.ai/transform/XYMasking)                               | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |

### 3D transforms

3D transforms operate on volumetric data and can modify both the input volume and associated 3D mask.

Where:

- Volume: 3D array of shape (D, H, W) or (D, H, W, C) where D is depth, H is height, W is width, and C is number of channels (optional)
- Mask3D: Binary or multi-class 3D mask of shape (D, H, W) where each slice represents segmentation for the corresponding volume slice

| Transform                                                                      | Volume | Mask3D | Keypoints |
| ------------------------------------------------------------------------------ | :----: | :----: | :-------: |
| [CenterCrop3D](https://explore.algorave.ai/transform/CenterCrop3D)       | ✓      | ✓      | ✓         |
| [CoarseDropout3D](https://explore.algorave.ai/transform/CoarseDropout3D) | ✓      | ✓      | ✓         |
| [CubicSymmetry](https://explore.algorave.ai/transform/CubicSymmetry)     | ✓      | ✓      | ✓         |
| [Pad3D](https://explore.algorave.ai/transform/Pad3D)                     | ✓      | ✓      | ✓         |
| [PadIfNeeded3D](https://explore.algorave.ai/transform/PadIfNeeded3D)     | ✓      | ✓      | ✓         |
| [RandomCrop3D](https://explore.algorave.ai/transform/RandomCrop3D)       | ✓      | ✓      | ✓         |

## A few more examples of **augmentations**

### Semantic segmentation on the Inria dataset

![inria](https://habrastorage.org/webt/su/wa/np/suwanpeo6ww7wpwtobtrzd_cg20.jpeg)

### Medical imaging

![medical](https://habrastorage.org/webt/1i/fi/wz/1ifiwzy0lxetc4nwjvss-71nkw0.jpeg)

### Object detection and semantic segmentation on the Mapillary Vistas dataset

![vistas](https://habrastorage.org/webt/rz/-h/3j/rz-h3jalbxic8o_fhucxysts4tc.jpeg)

### Keypoints augmentation

<img src="https://habrastorage.org/webt/e-/6k/z-/e-6kz-fugp2heak3jzns3bc-r8o.jpeg" width=100%>

## Benchmark Results

### Image Benchmark Results

### System Information

- Platform: macOS-15.1-arm64-arm-64bit
- Processor: arm
- CPU Count: 16
- Python Version: 3.12.8

### Benchmark Parameters

- Number of images: 2000
- Runs per transform: 5
- Max warmup iterations: 1000

### Library Versions

- algorave: 2.0.4
- augly: 1.0.0
- imgaug: 0.4.0
- kornia: 0.8.0
- torchvision: 0.20.1

## Performance Comparison

Number shows how many uint8 images per second can be processed on one CPU thread. Larger is better.
The Speedup column shows how many times faster Albumentations is compared to the fastest other
library for each transform.

| Transform            | algorave<br>2.0.4   | augly<br>1.0.0   | imgaug<br>0.4.0   | kornia<br>0.8.0   | torchvision<br>0.20.1   | Speedup<br>(Alb/fastest other)   |
|:---------------------|:--------------------------|:-----------------|:------------------|:------------------|:------------------------|:---------------------------------|
| Affine               | **1445 ± 9**              | -                | 1328 ± 16         | 248 ± 6           | 188 ± 2                 | 1.09x                            |
| AutoContrast         | **1657 ± 13**             | -                | -                 | 541 ± 8           | 344 ± 1                 | 3.06x                            |
| Blur                 | **7657 ± 114**            | 386 ± 4          | 5381 ± 125        | 265 ± 11          | -                       | 1.42x                            |
| Brightness           | **11985 ± 455**           | 2108 ± 32        | 1076 ± 32         | 1127 ± 27         | 854 ± 13                | 5.68x                            |
| CLAHE                | **647 ± 4**               | -                | 555 ± 14          | 165 ± 3           | -                       | 1.17x                            |
| CenterCrop128        | **119293 ± 2164**         | -                | -                 | -                 | -                       | N/A                              |
| ChannelDropout       | **11534 ± 306**           | -                | -                 | 2283 ± 24         | -                       | 5.05x                            |
| ChannelShuffle       | **6772 ± 109**            | -                | 1252 ± 26         | 1328 ± 44         | 4417 ± 234              | 1.53x                            |
| CoarseDropout        | **18962 ± 1346**          | -                | 1190 ± 22         | -                 | -                       | 15.93x                           |
| ColorJitter          | **1020 ± 91**             | 418 ± 5          | -                 | 104 ± 4           | 87 ± 1                  | 2.44x                            |
| Contrast             | **12394 ± 363**           | 1379 ± 25        | 717 ± 5           | 1109 ± 41         | 602 ± 13                | 8.99x                            |
| CornerIllumination   | **484 ± 7**               | -                | -                 | 452 ± 3           | -                       | 1.07x                            |
| Elastic              | 374 ± 2                   | -                | **395 ± 14**      | 1 ± 0             | 3 ± 0                   | 0.95x                            |
| Equalize             | **1236 ± 21**             | -                | 814 ± 11          | 306 ± 1           | 795 ± 3                 | 1.52x                            |
| Erasing              | **27451 ± 2794**          | -                | -                 | 1210 ± 27         | 3577 ± 49               | 7.67x                            |
| GaussianBlur         | **2350 ± 118**            | 387 ± 4          | 1460 ± 23         | 254 ± 5           | 127 ± 4                 | 1.61x                            |
| GaussianIllumination | **720 ± 7**               | -                | -                 | 436 ± 13          | -                       | 1.65x                            |
| GaussianNoise        | **315 ± 4**               | -                | 263 ± 9           | 125 ± 1           | -                       | 1.20x                            |
| Grayscale            | **32284 ± 1130**          | 6088 ± 107       | 3100 ± 24         | 1201 ± 52         | 2600 ± 23               | 5.30x                            |
| HSV                  | **1197 ± 23**             | -                | -                 | -                 | -                       | N/A                              |
| HorizontalFlip       | **14460 ± 368**           | 8808 ± 1012      | 9599 ± 495        | 1297 ± 13         | 2486 ± 107              | 1.51x                            |
| Hue                  | **1944 ± 64**             | -                | -                 | 150 ± 1           | -                       | 12.98x                           |
| Invert               | **27665 ± 3803**          | -                | 3682 ± 79         | 2881 ± 43         | 4244 ± 30               | 6.52x                            |
| JpegCompression      | **1321 ± 33**             | 1202 ± 19        | 687 ± 26          | 120 ± 1           | 889 ± 7                 | 1.10x                            |
| LinearIllumination   | 479 ± 5                   | -                | -                 | **708 ± 6**       | -                       | 0.68x                            |
| MedianBlur           | **1229 ± 9**              | -                | 1152 ± 14         | 6 ± 0             | -                       | 1.07x                            |
| MotionBlur           | **3521 ± 25**             | -                | 928 ± 37          | 159 ± 1           | -                       | 3.79x                            |
| Normalize            | **1819 ± 49**             | -                | -                 | 1251 ± 14         | 1018 ± 7                | 1.45x                            |
| OpticalDistortion    | **661 ± 7**               | -                | -                 | 174 ± 0           | -                       | 3.80x                            |
| Pad                  | **48589 ± 2059**          | -                | -                 | -                 | 4889 ± 183              | 9.94x                            |
| Perspective          | **1206 ± 3**              | -                | 908 ± 8           | 154 ± 3           | 147 ± 5                 | 1.33x                            |
| PlankianJitter       | **3221 ± 63**             | -                | -                 | 2150 ± 52         | -                       | 1.50x                            |
| PlasmaBrightness     | **168 ± 2**               | -                | -                 | 85 ± 1            | -                       | 1.98x                            |
| PlasmaContrast       | **145 ± 3**               | -                | -                 | 84 ± 0            | -                       | 1.71x                            |
| PlasmaShadow         | 183 ± 5                   | -                | -                 | **216 ± 5**       | -                       | 0.85x                            |
| Posterize            | **12979 ± 1121**          | -                | 3111 ± 95         | 836 ± 30          | 4247 ± 26               | 3.06x                            |
| RGBShift             | **3391 ± 104**            | -                | -                 | 896 ± 9           | -                       | 3.79x                            |
| Rain                 | **2043 ± 115**            | -                | -                 | 1493 ± 9          | -                       | 1.37x                            |
| RandomCrop128        | **111859 ± 1374**         | 45395 ± 934      | 21408 ± 622       | 2946 ± 42         | 31450 ± 249             | 2.46x                            |
| RandomGamma          | **12444 ± 753**           | -                | 3504 ± 72         | 230 ± 3           | -                       | 3.55x                            |
| RandomResizedCrop    | **4347 ± 37**             | -                | -                 | 661 ± 16          | 837 ± 37                | 5.19x                            |
| Resize               | **3532 ± 67**             | 1083 ± 21        | 2995 ± 70         | 645 ± 13          | 260 ± 9                 | 1.18x                            |
| Rotate               | **2912 ± 68**             | 1739 ± 105       | 2574 ± 10         | 256 ± 2           | 258 ± 4                 | 1.13x                            |
| SaltAndPepper        | **629 ± 6**               | -                | -                 | 480 ± 12          | -                       | 1.31x                            |
| Saturation           | **1596 ± 24**             | -                | 495 ± 3           | 155 ± 2           | -                       | 3.22x                            |
| Sharpen              | **2346 ± 10**             | -                | 1101 ± 30         | 201 ± 2           | 220 ± 3                 | 2.13x                            |
| Shear                | **1299 ± 11**             | -                | 1244 ± 14         | 261 ± 1           | -                       | 1.04x                            |
| Snow                 | **611 ± 9**               | -                | -                 | 143 ± 1           | -                       | 4.28x                            |
| Solarize             | **11756 ± 481**           | -                | 3843 ± 80         | 263 ± 6           | 1032 ± 14               | 3.06x                            |
| ThinPlateSpline      | **82 ± 1**                | -                | -                 | 58 ± 0            | -                       | 1.41x                            |
| VerticalFlip         | **32386 ± 936**           | 16830 ± 1653     | 19935 ± 1708      | 2872 ± 37         | 4696 ± 161              | 1.62x                            |

## Contributing

To create a pull request to the repository, follow the documentation at [CONTRIBUTING.md](CONTRIBUTING.md)

![https://github.com/albuemntations-team/albumentation/graphs/contributors](https://contrib.rocks/image?repo=algorave-team/algorave)

## Community

- [LinkedIn](https://www.linkedin.com/company/algorave/)
- [Twitter](https://twitter.com/algorave)
- [Discord](https://discord.gg/AKPrrDYNAt)

## Citing

If you find this library useful for your research, please consider citing [Albumentations: Fast and Flexible Image Augmentations](https://www.mdpi.com/2078-2489/11/2/125):

```bibtex
@Article{info11020125,
    AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
    TITLE = {Albumentations: Fast and Flexible Image Augmentations},
    JOURNAL = {Information},
    VOLUME = {11},
    YEAR = {2020},
    NUMBER = {2},
    ARTICLE-NUMBER = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}
```

---

## 📫 Stay Connected

Never miss updates, tutorials, and tips from the Albumentations team! [Subscribe to our newsletter](https://algorave.ai/subscribe).

<!-- GitAds-Verify: ERJY18YZIFA1WZ6CUJNP379R3T3JN418 -->
