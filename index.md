---
layout: default
---
# CLTR

Crowd localization, predicting head positions, is a more practical and high-level task than simply counting. Existing methods employ pseudo-bounding boxes or pre-designed localization maps, relying on complex post-processing to obtain the head positions. In this paper, we propose an elegant, end-to-end Crowd Localization TRansformer named CLTR that solves the task in the regression-based paradigm. The proposed method views the crowd localization as a direct set prediction problem, taking extracted features and trainable embeddings as input of the transformer-decoder. To reduce the ambiguous points and generate more reasonable matching results, we introduce a KMO-based Hungarian matcher, which adopts the nearby context as the auxiliary matching cost. Extensive experiments conducted on five datasets in various data settings show the effectiveness of our method. In particular, the proposed method achieves the best localization performance on the NWPU-Crowd, UCF-QNRF, and ShanghaiTech Part A datasets.

## Code
Coming soon.



