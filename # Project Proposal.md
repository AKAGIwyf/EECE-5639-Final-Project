# Project Proposal
#### 002839146 Yifan Wang                   

#### 002842604 Dawei Wang

#### 002839248 Jiatong Wu                    

#### 002240062 Tianqi Jiao

#### 

### Introduction

Image colorization is an essential image processing and computer vision branch to colorize images and videos. Recently, deep learning techniques progressed notably for image colorization. Given a gray photograph as input, hallucinating its color version is an ill-posed problem. First, gray images have much less information than colorful images, where each pixel value in the image is divided into three primary color components of R, G, and B, and each primary color component directly determines the intensity of its base color. However, A gray digital image is an image with only one sampled color per pixel. How to color the images and let them pass the color turing test is a big problem to solve. At the same time, the early AI was not perfect for the recognition of the shape of the picture, the adjacent grayscale of the edge of object’s contour could not accurately distinguish the recognition, and the color was more inclined to the low saturation color that was not vivid. What’s more, the training data done with color image decolorization will also have some impact, and the color information of the color photo to remove the RGB channel is not the real black and white camera original film, and there will be a big difference in the performance of grayscale.

### Related Works
- User-guided Image Colorization based on Palette: [Palette-based Photo Recoloring (princeton.edu)](https://gfx.cs.princeton.edu/pubs/Chang_2015_PPR/chang2015-palette_small.pdf)
	- Designing a system with GUI that allows users to indicate the color they want and automatically colorizes the image.
- Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization: [Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification (tsukuba.ac.jp)](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/data/colorization_sig2016.pdf)
	- Colorizing black-and-white image based on improved CNN.
- Generative Adversarial Networks (GAN): [1406.2661.pdf (arxiv.org)](https://arxiv.org/pdf/1406.2661.pdf)

### Dataset
As image data required to train our own model, we will download datasets from the Internet. 

- The link of the first one is: "https://www.kaggle.com/datasets/shravankumar9892/image-colorization/data". 
- It consists of around 25k images.

 We are looking for other datasets, and a total of 100k images are expected.

### Success Criteria
The objective of this work is to train a model based on Generative Adversarial Networks (GAN), and with the color palette provided, it is able to recolor the target images. The following picture is the result expected: 

<img src=".\result_image_sample.png" style="zoom:50%;" />



