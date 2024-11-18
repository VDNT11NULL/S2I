

1
# S2I
Dataset Link: https://drive.google.com/drive/folders/1-1KDpR3jGcd_21KgtmKCnDXiuc_W8NgP?usp=sharing
Comparative Analysis of Pix2Pix and CycleGAN: https://www.researchgate.net/publication/357301765_Comparison_and_Analysis_of_Image-to-Image_Generative_Adversarial_Networks_A_Survey
Bahadanau attentions: https://arxiv.org/abs/1409.0473
# UwU_Net_Generator
In the UwU_Net_Generator first a block is defined. When we are doing downsampling we are using Convolutional 2d layers and when we
are using doing upsampling then we are using 2D convolutional transpose layer. Each block uses leaky relu as an activation function 
and the architecture is made up of using these layers in a sequential manner
Then we define the Inner UNet where we define each block of the Inner layer(both up and down and the bottle neck)
Inside the forward function of the InnerNet class we define the connections
In the Outer UNet we define the blocks of the outer layer in a similar manner as that of InnerNet and in the forward function we also define 
the connections that should exist between the blocks of the outer net and the inner net.
# UwU_Net_Discriminator
In the UwU_Net_Discriminator we rely on the architecture of Pix2Pix discriminator. In the module we define the CNN Block where
each block is a convolutional 2d layer with Leaky ReLu as an activation function and in the Discriminator class we arrange the
blocks in a sequence. 
# Utils
In the Utils.py we define various functions 
1) save_some_example: For saving the output image in a folder named example
2) save_checkpoint: To set a check point such that after a certain no of epocks the weights get stored
3) load_checkpoint: So that the checkpoint is loaded and training can begin from the previously learnt weights at the checkpoint
# Config
In the Config.py file the hyperparameters are defined which have been derived experimentally
# HyperParameterTuning.py
In this HyperParameterTuning.py we use Optuna to do hyperparameter tuning. First we define a PairedImageDataset
that basically maps a sketch and a image. Then we define the objective where the hyperparameters are selected and then we do the
splitting into the validation size which is 0.2 times of the total training data set and then we do the Hyper Parmaeter Tuning
# Regularization.py
This is a regularized version of HyperParameterTuning.py where in the error function we add the gradient penalty term
and do the Hyperparameter Tuning
# Sample Datset
In the sample dataset directory we have 25 sketches and 25 coloured images that are extracted from the dataset of images
# Output
In the Output directory there are two folders Pix2Pix and UwU where the generated images on the validation dataset 
and the generated image for Pix2Pix and UwU architecture are stored
