# All_projects_resume

### Neural Machine Translation [Github](https://github.com/RishikeshDhayarkar/cs224n/tree/master/a5) | Python, Pytorch, Sklearn, Numpy, Nltk
* Developed an NMT for Spanish to English translation.  This Seq2Seq network with Multiplicative Attentionincludes a character-based convolutional encoder with Highway layers and a character-based LSTM decoder.
* The character level decoder gets triggered when the word level decoder fails to translate
* Sub-word modelling helps in producing rare words, out of vocabulary target words, and also handles transliteration

### Object detection | Python, Pytorch, Matplotlib, Numpy 
* Implemented a single stage detector, YOLO v2 from scratch.Trained and tested on PASCAL VOC 2007 dataset |
[Github](https://github.com/RishikeshDhayarkar/UMich-Computer-Vision/tree/master/yolo)
* This model was then enhanced to create a region proposal network with an ROI align function. Classification and
regression heads were attached on top of this to form a Two stage detector (FasterRCNN) | [Github](https://github.com/RishikeshDhayarkar/UMich-Computer-Vision/tree/master/fasterRCNN)

### Image captioning | [Github](https://github.com/RishikeshDhayarkar/cs231n/blob/master/assignments/assignment3/LSTM_Captioning.ipynb)| [Github](https://github.com/RishikeshDhayarkar/UMich-Computer-Vision/tree/master/image_cap) | Python, Pytorch, Nltk, PIL, Matplotlib, Scipy
* LSTM based image captioning system with Scaled dot-product attention was implemented from scratch.
Captioning was performed on COCO dataset. Image features were extracted using MobileNet v2


### Style Transfer and GANs | Python, Pytorch, PIL, Matplotlib, Scipy
* Implemented style transfer and feature inversion from scratch.This model incorporates total variation regularization
in addition to the content and style losses | [Github](https://github.com/RishikeshDhayarkar/cs231n/blob/master/assignments/assignment3/StyleTransfer-PyTorch.ipynb)
* From scratch implementation of vanilla GAN, LS-GAN and, DC-GAN on the MNIST dataset to analyse the
variation in the generated images across the all three algorithms | [Github](https://github.com/RishikeshDhayarkar/cs231n/blob/master/assignments/assignment3/Generative_Adversarial_Networks_PyTorch.ipynb)
