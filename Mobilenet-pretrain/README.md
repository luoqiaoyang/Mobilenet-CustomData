## Mobilenet pretrain

This is a simple tutorials to **load a pretrained model** with imagenet dataset. 

There are two main ways to load the pretrained mobilenet model.

###  Load model parameters

**Method 1**

Transform the model to nn.DataParallel, then load model parameters.

**Method 2**

Remove the prefix in parameters, then load model parameters.

### Load entire model

As [official guide](http://pytorch.org/docs/0.3.1/notes/serialization.html#recommend-saving-models) told,  the serialized data is bound to the specific classes and the exact directory structure used, so it can break in various ways when used in other projects, or after some serious refactors. And the entire model may be very large. So try to save model with parameters only and load it.

### Save model

[Mobilenet-Sample](https://github.com/luoqiaoyang/Mobilenet-CustomData/tree/master/Mobilenet-Sample) introduces how to construct a mobilenet model and save it. 