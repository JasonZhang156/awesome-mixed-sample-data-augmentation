# awesome-mixed-sample-data-augmentation

This repo is a collection of awesome things about mixed sample data augmentation, including papers, code, etc.

_ _ _

## Basic Method
We introduce a basic usage of mixed sample data augmentation, which was first proposed in **mixup: Beyond Empirical Risk Minimization [[ICLR2018]](https://arxiv.org/abs/1710.09412) [[code]](https://github.com/facebookresearch/mixup-cifar10)**.

**Formulation**

In mixup, the virtual training feature-target samples are produced as,

```
x˜ = λxi + (1 − λ)xj

y˜ = λyi + (1 − λ)yj
```

where (xi, yi) and (xj, yj) are two feature-target samples drawn at random from the training data, λ∈[0, 1]. The mixup hyper-parameter α controls the strength of interpolation between feature-target pairs and λ∼Beta(α, α).

**Training Pipeline**

The simple and basic training pipeline is shown as the following figure,

![](/image/mixup_pipeline.png)

**code**
The few lines of code necessary to implement mixup training in PyTorch

```
for (x1, y1), (x2, y2) in zip(loader1, loader2): 
  lam = numpy.random.beta(alpha, alpha) 
  x = Variable(lam * x1 + (1. - lam) * x2) 
  y = Variable(lam * y1 + (1. - lam) * y2) 
  optimizer.zero_grad() 
  loss(net(x), y).backward()
  optimizer.step()
```

_ _ _

## Application
### Classification Tasks (Image, Text, Audio, ...)
- Understanding and Enhancing Mixed Sample Data Augmentation [[Arxiv2020]](https://arxiv.org/abs/2002.12047) [[code]](https://github.com/ecs-vlc/FMix)
- Attentive CutMix: An Enhanced Data Augmentation Approach for Deep Learning Based Image Classification [[ICASSP2020]]
- Cutmix: Regularization strategy to train strong classifiers with localizable features [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf) [[code]](https://github.com/clovaai/CutMix-PyTorch)
- Improved Mixed-Example Data Augmentation [[WACV2019]](https://arxiv.org/abs/1805.11272) [[code]](https://github.com/ceciliaresearch/MixedExample)
- Manifold Mixup: Better Representations by Interpolating Hidden States [[ICML2019]](https://arxiv.org/abs/1806.05236) [[code]](https://github.com/vikasverma1077/manifold_mixup)
- Data augmentation using random image cropping and patching for deep CNNs [[TCSVT2019]](https://arxiv.org/abs/1811.09030) [[code]](https://github.com/jackryo/ricap)
- MixUp as Locally Linear Out-Of-Manifold Regularization [[AAAI2019]](https://www.aaai.org/ojs/index.php/AAAI/article/download/4256/4134)
- On Adversarial Mixup Resynthesis [[NeurIPS2019]](http://papers.nips.cc/paper/8686-on-adversarial-mixup-resynthesis.pdf) [[code]](https://github.com/christopher-beckham/amr)
- On mixup training: Improved calibration and predictive uncertainty for deep neural networks [[NeurIPS2019]](http://papers.nips.cc/paper/9540-on-mixup-training-improved-calibration-and-predictive-uncertainty-for-deep-neural-networks.pdf)
- mixup: Beyond Empirical Risk Minimization [[ICLR2018]](https://arxiv.org/abs/1710.09412) [[code]](https://github.com/facebookresearch/mixup-cifar10)
- Learning from between-class examples for deep sound recognition [[ICLR2018]](https://arxiv.org/abs/1711.10282) [[code]](https://github.com/mil-tokyo/bc_learning_sound/)
- Between-class Learning for Image Classification [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tokozume_Between-Class_Learning_for_CVPR_2018_paper.pdf) [[code]](https://github.com/mil-tokyo/bc_learning_image/)
- Data Augmentation by Pairing Samples for Images Classification [[Arxiv2018]](https://arxiv.org/abs/1801.02929)
- Mixup Learning Strategies for Text-independent Speaker Verification [[Interspeech2019]](https://pdfs.semanticscholar.org/0bc3/f8c6bc1f3568aac96d3ad0632ebe41134611.pdf)
- Deep Convolutional Neural Network with Mixup for Environmental Sound Classification [[PRCV2018]](https://arxiv.org/abs/1808.08405)
- An investigation of mixup training strategies for acoustic models in ASR [[Interspeech2018]](https://www.researchgate.net/profile/Ivan_Medennikov/publication/327389098_An_Investigation_of_Mixup_Training_Strategies_for_Acoustic_Models_in_ASR/links/5bc86248a6fdcc03c78f5a44/An-Investigation-of-Mixup-Training-Strategies-for-Acoustic-Models-in-ASR.pdf) [[code]](https://github.com/speechpro/mixup)
- Understanding Mixup Training Methods [[IEEE ACCESS 2018]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8478159)

### Semi-Supervised Learning
- Interpolation Consistency Training for Semi-Supervised Learning [[IJCAI2019]](https://arxiv.org/abs/1903.03825) [[code]](https://github.com/vikasverma1077/ICT)
- Unifying semi-supervised and robust learning by mixup [[ICLR2019 Workshop]](https://openreview.net/pdf?id=r1gp1jRN_4)
- Mixmatch: A holistic approach to semi-supervised learning [[NeurIPS2019]](https://papers.nips.cc/paper/8749-mixmatch-a-holistic-approach-to-semi-supervised-learning.pdf) [[code]](https://github.com/google-research/mixmatch)
- Semi-Supervised and Task-Driven Data Augmentation [[IPMI2019]](https://arxiv.org/abs/1902.05396)

### Object Localization
- Cutmix: Regularization strategy to train strong classifiers with localizable features [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf) [[code]](https://github.com/clovaai/CutMix-PyTorch)

### Natural Language Processing
- On mixup training: Improved calibration and predictive uncertainty for deep neural networks [[NeurIPS2019]](http://papers.nips.cc/paper/9540-on-mixup-training-improved-calibration-and-predictive-uncertainty-for-deep-neural-networks.pdf)

### Image Segmentation
- Improving Data Augmentation for Medical Image Segmentation [[MIDL2018]](https://openreview.net/pdf?id=rkBBChjiG)

### Generative Adversarial Networks
- Mixed batches and symmetric discriminators for GAN training [[ICML2018]](https://arxiv.org/abs/1806.07185)
- mixup: Beyond Empirical Risk Minimization [[ICLR2018]](https://arxiv.org/abs/1710.09412) [[code]](https://github.com/facebookresearch/mixup-cifar10)

### Domain Adaptation
- Improve Unsupervised Domain Adaptation with Mixup Training [[Arxiv2020]](https://arxiv.org/abs/2001.00677)

### Few-shot Learning
- Charting the Right Manifold: Manifold Mixup for Few-shot Learning [[WACV2020]](http://openaccess.thecvf.com/content_WACV_2020/papers/Mangla_Charting_the_Right_Manifold_Manifold_Mixup_for_Few-shot_Learning_WACV_2020_paper.pdf) [[code]](https://github.com/nupurkmr9/S2M2_fewshot)
