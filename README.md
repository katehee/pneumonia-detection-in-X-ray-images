# X-ray_pneumonia_classification


#literature review papers (references)

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7554804/
- two pre-trained models, ResNet152V2 and MobileNetV2, a Convolutional Neural Network (CNN), and a Long Short-Term Memory (LSTM).



The main objective of this project is to detect pneumonia using chest X-ray images along with deep convolutional neural network. Razaak et al. discussed the challenges of AI-based methodology with regard to medical imaging [2]. Several biomedical image detection techniques have been proposed in the literature to help in diagnosing numerous diseases such as breast and renal cancers [3] [4]. 

For lung diseases in particular, some of proposed techniques used machine learning algorithms and others used deep learning methods for feature extraction and classification. The parameters of these methods are optimized to achieve high accuracy in disease classification. In Antin et al. study, the authors used logistic regression as their supervised machine learning method and DenseNet as their deep learning method to diagnose Pneumonia [1]. From a relatively low scale of The Area Under the Curve (AUC) of logistic regression model, the study conclude that logistic regression does not adequately capture the complexity of X-ray images. Whereas densely connected network having 121 layers (DenseNet) achieves a better result. Deep learning is a better advancement over machine learning as it can easily operate on images and extract the features responsible to classify the disease.  

There are also several recent works have proved the benefit of data augmentation in improving CNN performance for various deep learning applications [9].  Data Augmentation is a simple method to increase the dataset size by adding more invariant samples and thus reduce overfitting [10] [11]. In Monshi et al. study, they demonstrated that the optimization of data augmentation and CNN hyperparameters is an effective tool to extract features from chest X-rays [12]. The CNN architectures used in this study were VGG-19 and ResNet50 and they proposed that such optimization increased the accuracy of both models. In addition, Nishio et al. also proposed a computer-aided diagnosis system for detection of COVID-19 pneumonia [13]. Their proposed model utilized VGG16 and studied the effect of conventional and mix-up method of chest X-ray image augmentation. Their results proved that the combinational conventional and mix-up augmentation methods were more effective than single type or no augmentation method. The conventional data augmentation method included rotation (+/- 15), x-axis and y-axis shift (+/- 15%), horizontal flipping, scaling, and shear transformation. The mix-up augmentation was set to 0.1. 

In Hashmi et al.’s work [6], pre-trained ResNet50 multilayer architecture was used to identify pneumonia on chest X-ray images. The dataset used in this study is identical to the dataset I used for this project. As the dataset contain insufficient number of X-ray images, data augmentation techniques were deployed to increase the size of the training dataset. The authors further scale up ResNet50 model by compound scaling. Training images were resized to 224*224 and images of healthy chest X-rays were augmented twice with augmentation settings (crop and pad = 0.25, Horizontal shift =0.15, Rotation = 35, Vertical shift = 0.2). The most optimum results were obtained with the learning rate of 0.001 and Stochastic gradient descent optimizer. The proposed compound scaled ResNet50 attained a test accuracy of 98.14%, an AUC score of 99.71 and an F1 score of 98.3 on the test data from the Guangzhou Women and Children’s Medical Center pneumonia dataset.

Rajaraman et al. addressed the challenges of interpreting CNN model behavior that could adversely affect the clinical decision [7]. CNNs are often described as black boxes as their performance do not give adequate explanations and insight on the form of function.  The authors present a visualization strategy for localizing the region of interest in the Pediatric chest X-rays and customized VGG16 model and attain 96.2% and 93.6% of accuracy values in detecting bacterial and viral pneumonia classification respectively. They embedded a CAM-compatible architecture to the customized VGG16 model to visualize the model predictions. 

The authors of [5] presented several transfer learning methods as feature extractors to classify normal and pneumonia-infected chest X-rays. Training a deep Convolutional neural network models from scratch requires a lot of inputs and efforts because there are millions of trainable parameters. Instead, there are baseline models such as ResNet, Xception, or DenseNet available for researchers to reuse their pre-trained weights using transfer learning technique. In this study, Lujan-Garcia et al. use several convolution Neural Network pretrained weights on ImageNet as an initialization for the proposed model to automatically classify healthy and pneumonia chest X-ray images of people. Among the evaluated models, Xception outperformed VGG-16, ResNet and Inception-V3. This network contains 36 convolutional layers including a global average pooling, dropout, layers and adam optimizer is used. The transfer learning technique support numerous studies to resolve a small insufficient dataset problem and help in achieving a good generalization of models. It is reported that the outperformed Xception model achieved a precision value of 0.843, a recall value of 0.992, a F-1 score of 0.912 and an AUC value of 0.962 for the ROC curve. The dataset used in this study also had imbalance issue where pneumonia class had almost three times more samples than normal class samples. To solve the imbalance, the authors chose to use Random Under sampling. The Random Under sampling is a non-heuristic method to help to combat the imbalance problem [8]. The method eliminates a portion of samples from a majority class to balance out the training data.   


[1] . Antin B., Kravitz J., Martayan E. Detecting Pneumonia in Chest X-rays with Supervised Learning. Semanticscholar Org.; Allen Institute for Artificial intelligence, Seattle, WA, USA: 2017. [Google Scholar]

[2] Razzak MI, Naz S, Zaib A. Deep learning for medical image processing: overview, challenges and the future. 2017.

[3] A. Cruz-Roa, H. Gilmore, A. Basavanhally, M. Feldman, S. Ganesan, N.N.C. Shih, J. Tomaszewski, F.A. González, A. Madabhushi, Accurate and reproducible invasive breast cancer detection in whole-slide images: a deep learning approach for quantifying tumor extent, Sci. Rep. (2017) 7, doi:10.1038/srep46450. 
[4] Xi IL, Zhao Y, Wang R, Chang M, Purkayastha S, Chang K, et al. Deep Learning to Distinguish Benign from Malignant Renal Lesions Based on Routine MR Imaging. 
[5] Luján-García J.E., Yáñez-Márquez C., Villuendas-Rey Y., Camacho-Nieto O. A Transfer Learning Method for Pneumonia Classification and Visualization. Appl. Sci. 2020;10:2908. doi: 10.3390/app10082908.

[6] Pneumonia detection in chest X-ray images using compound scaled deep learning model https://www.tandfonline.com/doi/full/10.1080/00051144.2021.1973297

[7] Rajaraman S, Candemir S, Kim I, et al. Visualization and interpretation of convolutional neural network predictions in detecting pneumonia in pediatric chest radiographs. Appl Sci. 2018;8(10):1715.

[8] Batista, G.E.A.P.A.; Prati, R.C.; Monard, M.C. A study of the behavior of several methods for balancing machine learning training data.

[9] Calderon-Ramirez S. 2020. Correcting Data Imbalance for Semi-supervised Covid-19 Detection Using X-Ray Chest Images. 
[10] Shorten C., Khoshgoftaar T.M. A survey on image data augmentation for deep learning. 
[11] Taylor L., Nitschke G. 2017. Improving Deep Learning Using Generic Data Augmentation.

[12] Monshi MMA, Poon J, Chung V, Monshi FM (2021) Covidxraynet: Optimizing data augmentation and cnn hyperparameters for improved covid-19 detection from cxr. Computers in Biology and Medicine 133:104375

[13] Nishio, M., Noguchi, S., Matsuo, H. et al. Automatic classification between COVID-19 pneumonia, non-COVID-19 pneumonia, and the healthy on chest X-ray image: combination of data augmentation methods. Sci Rep 10, 17532 (2020). https://doi.org/10.1038/s41598-020-74539-2


