# Literature Review of Face recognition projects

Below part is based on the [Trained Model Eng](https://github.com/betars/Face-Resources#trained-model) and [Trained Model CN](https://zhuanlan.zhihu.com/p/35339201). Give a more details to summarize the accuracy and key technologies in each work. Some discussion webpages included which most likely are Chinese. 

## Trained Model

1. [facenet](https://github.com/davidsandberg/facenet). Google's FaceNet deep neural network.
	* 99.63% @ 2015 by Google. 
	* CNN: Tripletloss via Tensorflow + Python
	* mtcnn to detect and align the face

2. [openface](https://github.com/cmusatyalab/openface). Face recognition with Google's FaceNet deep neural network using Torch.
	* 93% @ 2015 by CMU. 
	* CNN: Use FaceNet Architecture (Tripletloss via Torch)
	* dlib to detech and align the face
		* if it changes to mtcnn, the accruacy [drops badly](https://zhuanlan.zhihu.com/p/43804018).

3. [SeetaFace Engine](https://github.com/seetaface/SeetaFaceEngine). SeetaFace Engine is an open source C++ face recognition engine, which can run on CPU with no third-party dependence. 
	* @2017 by 计算所 CAS
	* https://zhuanlan.zhihu.com/p/22451474
	* http://www.seetatech.com

4. [Caffe-face](https://github.com/ydwen/caffe-face) - Caffe Face is developed for face recognition using deep neural networks. 
	* 99.28%@2016, 港中文+深圳先进院
	* Softmax Loss + Center Loss: 
		* Softmax Loss: 深度特征类内离散度较大
		* Center Loss: 深度特征和类别中心将趋近于 0.  是使得任意红色特征点之间的距离小于红色特征点与蓝色特征点之间的距离，以确保最好的不同特征类别的划分.
	* https://blog.csdn.net/qq_14845119/article/details/53308996
	* https://blog.csdn.net/zziahgf/article/details/78548663

5. [Norm-Face](https://github.com/happynear/NormFace) - Norm Face, finetuned from  [center-face](https://github.com/ydwen/caffe-face) and [Light-CNN](https://github.com/AlfredXiangWu/face_verification_experiment)
	* 98.95% @ 2017 电子科技大学(成都) + JHU(Maryland)
	* two strategies for training using normalized features. 
		* The first is a modification of softmax loss, which optimizes cosine similarity instead of inner-product. 
		* The second is a reformulation of metric learning by introducing an agent vector for each class

6. [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/): VGG-Face CNN descriptor. Impressed embedding loss.
	* 97% @ 2015 by University of Oxford 
	* CNN
  
7. [VGG-Face2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
	* 98% @ 2018 
	* CNN: SeNet & ResNet-50

## [Compare](https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/) the dlib and OpenCV lib
* General Case
  * In most applications, we won’t know the size of the face in the image before-hand. Thus, it is better to use OpenCV – DNN method as it is pretty fast and very accurate, even for small sized faces. It also detects faces at various angles. We recommend to use OpenCV-DNN in most
* For medium to large image sizes
  * Dlib HoG is the fastest method on CPU. But it does not detect small sized faces ( < 70x70 ). So, if you know that your application will not be dealing with very small sized faces ( for example a selfie app ), then HoG based Face detector is a better option. Also, If you can use a GPU, then MMOD face detector is the best option as it is very fast on GPU and also provides detection at various angles.

## Related opensource projects:

* dlib https://github.com/Malkhan52/face_recognition
* mtcnn face detection which based on FaceNet project from https://github.com/chenlinzhong/face-login
  * 人脸检测方法有许多，比如opencv自带的人脸Haar特征分类器和dlib人脸检测方法等。 对于opencv的人脸检测方法，有点是简单，快速；存在的问题是人脸检测效果不好。正面/垂直/光线较好的人脸，该方法可以检测出来，而侧面/歪斜/光线不好的人脸，无法检测。因此，该方法不适合现场应用。对于dlib人脸检测方法 ，效果好于opencv的方法，但是检测力度也难以达到现场应用标准。 本文中，我们采用了基于深度学习方法的mtcnn人脸检测系统（mtcnn：Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks）。
  * mtcnn人脸检测方法对自然环境中光线，角度和人脸表情变化更具有**鲁棒性，人脸检测效果更好；同时，内存消耗不大，可以实现实时人脸检测**。本文中采用mtcnn是基于python和tensorflow的实现（代码来自于davidsandberg，caffe实现代码参见：kpzhang93）
  **Why my testing codes cost about 1.5 second per picture?** 
