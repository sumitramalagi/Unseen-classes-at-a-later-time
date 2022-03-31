
# Unseen Classes at a Later Time? No Problem [[arXiv](https://arxiv.org/abs/2203.16517) ]
### Accepted at CVPR 2022

<p align="center" width="100%">
<img src="https://github.com/sumitramalagi/Unseen-classes-at-a-later-time/blob/main/settings.png" width="600"/>
</p>


<p align="center" width="80%">
The figure shows how our newly formulated online CGZSL setting relates to exsiting settings.
</p>

#### Abstract

Recent progress towards learning from limited supervision has encouraged efforts towards designing models that can recognize novel classes at test time (generalized zero-shot learning or GZSL). GZSL approaches assume knowledge of all classes, with or without labeled data, beforehand. However, practical scenarios demand models that are adaptable and can handle dynamic addition of new seen and unseen classes on the fly (i.e continual generalized zero-shot learning or CGZSL). One solution is to sequentially retrain and reuse conventional GZSL methods, however, such an approach suffers from catastrophic forgetting leading to suboptimal generalization performance.
A few recent efforts towards tackling CGZSL have been limited by difference in settings, practicality, data splits and protocols followed -- inhibiting fair comparison and a clear direction forward. 
Motivated from these observations, in this work, we firstly consolidate the different CGZSL setting variants and propose a new Online-CGZSL setting which is more practical and flexible. Secondly, we introduce a unified feature-generative framework for CGZSL that leverages bi-directional incremental alignment to dynamically adapt to addition of new classes, with or without labeled data, that arrive over time in any of these CGZSL settings. Our comprehensive experiments and analysis on five benchmark datasets and comparison with baselines show that our approach consistently outperforms existing methods, especially on the more practical Online setting. 

Requirements: \
Python: 3.8.8 \
PyTorch: 1.4.0 \
sklearn: 0.24.1 \
scipy: 1.6.2 \
numpy: 1.20.1 \
CUDA Version: 11.2 

Usage: \
python main.py

Please download the preprocessed ZSL datasets from publicly available repository.
(http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip)
Store the dataset in a seperate folder named "data"

Use the appropriate dataloader based on the setting (Static,Dynamic and Online).





