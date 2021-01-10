# List of all projects

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

### Data Analysis | [Github](https://github.com/RishikeshDhayarkar/Duke_data_science) | R, RStudio
* Performed rigorous exploratory data analysis on the Behavioral Risk Factor Surveillance System dataset
* Conducted a statistical inference study on the General Social Survey dataset

### NYC taxi Dataset Analysis | python, pySpark, Hadoop
* Extensive exploration of NYC taxi dataset using Hadoop streaming. Includes fares, trips, and licences information
for all taxi trips for the years 2012 and 2013.(dataset size - 3 GB) | [Github](https://github.com/RishikeshDhayarkar/CS-GY-6513-Big-Data/tree/main/assignment_1)
* Thorough usage of Spark and SparkSQL to explore, analyse, and identify issues in NYC taxi dataset.(dataset size -
4.3 GB) | [Github](https://github.com/RishikeshDhayarkar/CS-GY-6513-Big-Data/tree/main/assignment_2)

### Question Answering on SQuAD 2.0 | [Github](https://github.com/RishikeshDhayarkar/CS-GY-9233-Deep-Learning) | pytorch, Nltk, Huggingface Transformers
* Utilized various tools and techniques such as Pre-trained Contextual Embeddings(PCE), Data Augmentation, and
merging PCE models with classical QA models such as BiDAF to solve the problem of question answering.
* Data Augmentation included synonym replacement and generating new sentences to augment the context
paragraphs in a coherent fashion. GLOVE embeddings in the BiDAF model were replaced by BERT embeddings.
Ensembling multiple models such as RoBERTa, BERT, ALBERT, and BiDAF models, we were able to achive a top
EM/F1 of 81.13/84.06.

### Adversarial Attacks on Spam Filters | [Github](https://github.com/RishikeshDhayarkar/ECE-GY-9163-ML-for-Cyber-Security/blob/main/A1/rbd291_ML_sec_A1.ipynb) | python, sklearn
* Performed feature selection using the information gain (IG) metric. Using these features Bernoulli Naive
Bayes(NB), Multinomial NB classifiers with binary and tern frequency features were implemented from scratch.
* Formalized the adversarial problem as a game between a cost- sensitive classifier and a cost-sensitive adversary.
Built a classifier that is optimal given the adversary’s attack strategy. The adversary immune classifier had a spam
recall of 94.28% and a spam precision of 78.75%

### Adversarial Attacks on Deep Neural Networks | python, tensorflow, keras
* Designed a backdoor detector for malicious neural nets trained on the YouTube Face dataset. Implemented a
modified version of this paper. Includes intentionally perturbing the incoming input, by superimposing various
image patterns and observe the randomness of the predicted classes for perturbed inputs from a deployed model. |
[Github](https://github.com/RishikeshDhayarkar/ML-security)
* Exhaustively investigated adversarial retraining as a defense against adversarial perturbations. Untargetted and
targetted attacks based on Fast Gradient Sign Methods(FSGM) were implemented. Experimentation was done to
analyse if adversarial retraining was a valid defense against these attacks. | [Github](https://github.com/RishikeshDhayarkar/ECE-GY-9163-ML-for-Cyber-Security/blob/main/A2/rbd291_ML_sec_A2.ipynb)

### Recommender System - !!!!! | [Github]() | python, pyspark
* Implemented a recommender system to predict the ratings/stars for the given user ids and business ids on the Yelp
reviews dataset. Collaborative Filtering algorithm was used and an RMS error of 1.115 was obtained using this
implementation

### LSH for Approximate Nearest Neighbours-!!!!! | [Github]() | python, pyspark
* Constructed an algorithm based on LSH to find similar businesses according to the customer ratings. This was
done on the Yelp reviews dataset and a similarity metric of Jaccard similarity was used.
* Performed a comparison of Linear Search and Approximate Nearest Neighbour search using LSH on a dataset of
images. Images of size 20x20 and a distance metric of L1 were used. LSH based approach had a speed-up of 4.21
over the linear search method.

### Generating Frequent itemsets-!!!!! | [Github]() | python, pyspark 
* Implemented an algorithm to find frequent businesses reviewed/visited by customers and frequent customers for all
businesses. SON algorithm on top of PCY was used on the Yelp reviews dataset to achieve this goal

### Sentiment Analysis using Contextual Embeddings-!!!!!|[Github]()|python, pytorch
* Built an algorithm to classify the customer reviews for apps on Google Playstore.  Review features were extractedusing BERT embeddings.  Reviews were classified as positive, negative, or neutral with an accuracy of 87.64%

### Neural Dependency Parser | [Github](https://github.com/RishikeshDhayarkar/cs224n/tree/master/a3/student) |Python, Pytorch, Numpy, Nltk
* This greedy model employed Arc-standard system of transitions to predict one transition (shift, left-arc, right-arc)
at a time based on the state of buffer, stack and the dependency arc set. Unlabelled Attachment score (UAS)-88.59

### Swype - A dating app | [Github]() | Java, Android studio, Firebase Database
* Uses the swiping mechanism for like/dislike. Candidates can be filtered based on distance, age, gender, etc.
Location and distance features handled using Google Maps API. Includes private chat functionality

### Prattle - A social media app | [Github](https://github.com/RishikeshDhayarkar/Swype)(https://github.com/RishikeshDhayarkar/Prattle) | Java, Android studio, Firebase
* Developed a social media application that facilitates image sharing experience. User data managed via Firebase
database. Includes features such as Post, Like, Comment, Connect/Follow, inbuilt private chat system

























































