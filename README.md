## X-ray_pneumonia_classification

## 1. Objective 
- Detect pneumonia from chest X-ray images using deep Convolutional Neural Network.  
        
## 2. Project Code  
- <a href="https://github.com/katehee/X-ray_pneumonia_classification/blob/master/pneumonia.ipynb" target="_blank">code</a>

## 3. Data 
5932 chest X-ray images of one to five years old pediatric patients from Guangzhou Women and Children’s Medical center.

## 4. Method 

 (a). Image data Preprocessing
 
	- Image resize and normalization 
  
	- Augmentation (rotation, zoom, shift) and oversampling used to resolve class imbalance issue  
	
	- Three different training set
      1) data with augmentation
      2) data with augmentation and oversampling
      3) data without augmentation
![Screen Shot 2022-07-19 at 5 03 57 AM](https://user-images.githubusercontent.com/89289320/179712058-16ff99ad-8b2a-4405-ba1e-9c08ba7daf10.png)

- A pneumonia image in the first position (far left) is augmented to five different transformed images, shown in the second to sixth position.

 (b). Modeling: CNN
 
	- customized CNN model (5 convolutional layers, batch normalization, max-pooling, dropout) 
    
	- ReLU activation function, Adam optimizer, Epoch size = 15, Batch size = 32 
  
	- Performance measures: loss, accuracy, precision (specificity), recall (sensitivity) and f1-score 
  
	
![CNN architecture](https://user-images.githubusercontent.com/89289320/179708419-8f5ff198-8a4c-447c-ae5a-9bc4a2fdb282.png)
  
  
 (c). Modeling: Pre-trained models
 
	- ResNet50, DenseNet121, VGG16, Xception, InceptionV3: pretrained weights on ImageNet
  
	- All layers are trained (model.trainable = True), Epoch size = 15, Batch size = 32   
  
	- ReduceLROnPlateau used for training optimization  
  
	- Performance measures: loss, accuracy, precision (specificity), recall (sensitivity) and f1-score
  
	

   (d). Interpretation of visualization of intermediate layers in neural network 
 
	- Our understanding of how these models work, especially at the intermediate layers remains unclear
  
	- Feature maps are examined to observe which areas of images are activated by convolutional layers  
  
	
   (e). Conclusion 
 
	- With Xception architecture, able to detect pneumonia with 95% in accuracy
  
	- Feature maps are examined to observe which areas of images are activated by convolutional layers  
  
## 5. techniques/Tools: 
- CNN, pre-trained models for image classification (ResNet, DenseNet, Xception, Inception, VGG16) 
- image augmentation, model optimization, visualization of feature maps 
- confusion matrix, classification report 

<img width="479" alt="Screen Shot 2022-08-28 at 9 48 32 PM" src="https://user-images.githubusercontent.com/89289320/187107049-520cdef4-b336-4623-9289-8bcbbd8db432.png">

