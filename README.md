# albumentations動作確認

## albumentationsとは？

雑に説明するとデータ拡張に用いられるライブラリである。

[Github albumentations](https://github.com/albumentations-team/albumentations)

[Github albumentations examples](https://github.com/albumentations-team/albumentations_examples)

[albumentations docs](https://albumentations.readthedocs.io/en/latest/index.html)

## 実行環境

Windows 10<br>
Anaconda <br>
Python 3.7.6<br>

| Module name | Version |
| - | - |
| albumentations | 0.4.3 |
| torch | 1.4.0 |
| torchvision | 0.5.0 |

## torchvisionとalbumentationsの使い方の違い

[Official docs](https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/migrating_from_torchvision_to_albumentations.ipynb)

torchvisionは`PIL.Image`から`Tensor`に変換する。

```torchvision PIL
image = Image.open(file_path)
image = transform(image)
```

albumentationsは`cv2`や`ndarray`などの配列から`Tensor`に変換する。また、引数を指定しなければならない。

```albumentations cv2
image = cv2.imread(file_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
augmented = transform(image=image)
image = augmented['image']
```

```albumentations PIL
image = Image.open(file_path)
image_np = np.array(image)
augmented = transform(image=image_np)
image = Image.fromarray(augmented['image'])
```

## 1. Augmentations

[Official docs](https://albumentations.readthedocs.io/en/latest/api/augmentations.html)

変換用のクラスはBasicTransformクラスを親クラスにもつImageOnlyTransformクラスかDualTransformクラスを継承して定義される。

---

## Affine

アフィン変換・・・平行移動、拡大縮小、回転

| Class name | Parent class name | Targets | Channels | Note |
| :-- | :-- | :-- | :-- | :-- |
| ShiftScaleRotate | DualTransform | image, mask, keypoints | all | シフト変換、スケール変換、回転 |
| Rotate | DualTransform | image, mask, bboxes, keypoints | all | ランダムに回転 |
| RandomRotate90 | DualTransform | image, mask, bboxes, keypoints | all | ランダムに90度回転 |
| RandomScale | DualTransform | image, mask, bboxes, keypoints | all | ランダムにスケーリング |

---

## Blur

フィルタ処理を行い、画像をぼかす。

[OpenCV reference](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_filtering/py_filtering.html)

| Class name | Parent class name | Targets | Channels | Note |
| :-- | :-- | :-- | :-- | :-- |
| Blur | ImageOnlyTransform | image | all | 平均フィルタ<br>cv2.blur |
| GaussianBlur | ImageOnlyTransform<br> -> Blur | image | all | ガウシアンフィルタ<br>cv2.GaussianBlur |
| MedianBlur | ImageOnlyTransform<br> -> Blur | image | all | 中央値フィルタ<br>cv2.medianBlur |
| MotionBlur | ImageOnlyTransform<br> -> Blur | image | all | モーションブラー<br>cv2.filter2D |

---

## Crop

一部を切り抜く。

| Class name | Parent class name | Targets | Channels | Note |
| :-- | :-- | :-- | :-- | :-- |
| Crop | DualTransform | image, mask, bboxes, keypoints | all | クリッピング |
| CenterCrop | DualTransform| image, mask, bboxes, keypoints | all | センタークリッピング |
| RandomCrop | DualTransform | image, mask, bboxes, keypoints | all | ランダムクリッピング |
| RandomResizedCrop | DualTransform<br> -> _BaseRandomSizedCrop | image, mask, bboxes, keypoints | all | クリッピングしてリサイズ |
| RandomSizedCrop | DualTransform<br> -> _BaseRandomSizedCrop | image, mask, bboxes, keypoints | all | クリッピングしてリサイズ |

---

## Flip

画像を水平、もしくは垂直、斜めに反転する。

| Class name | Parent class name | Targets | Channels | Note |
| :- | :- | :- | :- | :- |
| Flip | DualTransform | image, mask, bboxes, keypoints | all | 水平反転、垂直反転のいずれかをランダムに行う |
| HorizontalFlip | DualTransform | image, mask, bboxes, keypoints | all | 水平反転 |
| VerticalFlip | DualTransform | image, mask, bboxes, keypoints | all | 垂直反転 |
| Transpose | DualTransform | image, mask, bboxes, keypoints | all | 転置 |

---

## Noise

| Class name | Parent class name | Targets | Channels | Note |
| :- | :- | :- | :- | :- |
| GaussNoise | ImageOnlyTransform | image | all | ノイズ |

---

## Normalize

| Class name | Parent class name | Targets | Channels | Note |
| :- | :- | :- | :- | :- |
| Normalize | ImageOnlyTransform | image | 1, 3 | 正規化 |

---

## Out

| Class name | Parent class name | Targets | Channels | Note |
| :- | :- | :- | :- | :- |
| Cutout | ImageOnlyTransform | image | all | カットアウト |
| CoarseDropout | ImageOnlyTransform | image | all | ドロップアウト |

---

## Resize

サイズを変更する。

| Class name | Parent class name | Targets | Channels | Note |
| :- | :- | :- | :- | :- |
| Resize | DualTransform | image, mask, bboxes, keypoints | all | サイズ変更 |
| SmallestMaxSize | DualTransform | image, mask, bboxes, keypoints | all | サイズ変更 |
| LongestMaxSize | DualTransform | image, mask, bboxes, keypoints | all | サイズ変更 |

---

## Weather

| Class name | Parent class name | Targets | Channels | Note |
| :- | :- | :- | :- | :- |
| RandomFog | ImageOnlyTransform | image | 3 | 霧 |
| RandomRain | ImageOnlyTransform | image | 3 | 雨 |
| RandomShadow | ImageOnlyTransform | image | 3 | 曇り |
| RandomSnow | ImageOnlyTransform | image | 3 | 雪 |
| RandomSunFlare | ImageOnlyTransform | image | 3 | 太陽フレア |

---

## Other

| Class name | Parent class name | Targets | Channels | Note |
| :- | :- | :- | :- | :- |
| CLAHE | ImageOnlyTransform | image | 1, 3 |  |
| Downscale | ImageOnlyTransform | image | 1, 3 |  |
| Equalize | ImageOnlyTransform | image | 1, 3 |  |
| FromFloat | ImageOnlyTransform | image | 1, 3 |  |
| ImageCompression | ImageOnlyTransform | image | 1, 3 |  |
| JpegCompression | ImageOnlyTransform<br> -> ImageCompression | image | 1, 3 |  |
| InvertImg | ImageOnlyTransform | image | 1, 3 | 反転 |
| MultiplicativeNoise | ImageOnlyTransform | image | 1, 3 | ノイズ |
| Posterize | ImageOnlyTransform | image | 1, 3 |  |
| RandomBrightnessContrast | ImageOnlyTransform | image | 1, 3 |  |
| RandomBrightness | ImageOnlyTransform<br> -> RandomBrightnessContrast | image | 1, 3 |  |
| RandomContrast | ImageOnlyTransform<br> -> RandomBrightnessContrast | image | 1, 3 |  |
| RandomGamma | ImageOnlyTransform | image | 1, 3 |  |
| Solarize | ImageOnlyTransform | image | 1, 3 |  |
| ToFloat | ImageOnlyTransform | image | 1, 3 |  |
| ToGray | ImageOnlyTransform | image | 3 | グレースケール化 |
| ElasticTransform | DualTransform | image, mask | 1, 3 |  |
| GridDistortion | DualTransform | image, mask | 1, 3 | グリッド歪み |
| OpticalDistortion | DualTransform | image, mask | 1, 3 | 光学的歪み |
| RandomGridShuffle | DualTransform | image, mask | 1, 3 | グリッドシャッフル |
| PadIfNeeded | DualTransform | image, mask, bbox, keypoints | 1, 3 |  |
| Lambda | NoOp | image, mask, bboxes, keypoints | 1, 3 |  |
| MaskDropout | DualTransform | image, mask | | マスク画像が必要 |
| RandomSizedBBoxSafeCrop | DualTransform | image, mask, bboxes | | bboxesが必要 |
| CropNonEmptyMaskIfExists | DualTransform | image, mask, bboxes, keypoints | | マスクが必要 |
| RandomCropNearBBox | DualTransform | image, mask, bboxes, keypoints | | cropping_bboxが必要 |
| ChannelDropout | ImageOnlyTransform | image | 3 |  |
| ChannelShuffle | ImageOnlyTransform | image | 3 |  |
| HueSaturationValue | ImageOnlyTransform | image | 3 |  |
| ISONoise | ImageOnlyTransform | image | 3 |  |
| RGBShift | ImageOnlyTransform | image | 3 |  |
| ToSepia | ImageOnlyTransform | image | 3 |  |

## 2. Imgaug helpers

[Official docs](https://albumentations.readthedocs.io/en/latest/api/imgaug.html)

| Class name | Parent class name | Targets | Channels |
| :- | :- | :- | :- |
| IAAEmboss |  ImageOnlyTransform<br> -> ImageOnlyIAATransform | image | all |
| IAASuperpixels |  ImageOnlyTransform<br> -> ImageOnlyIAATransform | image | all |
| IAASharpen |  ImageOnlyTransform<br> -> ImageOnlyIAATransform | image | all |
| IAAAdditiveGaussianNoise |  ImageOnlyTransform<br> -> ImageOnlyIAATransform | image | all |
| IAAAffine | DualTransform<br> -> DualIAATransform | image, mask | all |
| IAAPiecewiseAffine | DualTransform<br> -> DualIAATransform | image, mask | all |
| IAAPerspective | DualTransform<br> -> DualIAATransform | image, mask | all |

## 3. Pytorch helpers

[Official docs](https://albumentations.readthedocs.io/en/latest/api/pytorch.html)

| Class name | Parent class name | Targets | Channels |
| :- | :- | :- | :- |
| ToTensor | - | - | all |
| ToTensorV2 | - | - | all |
