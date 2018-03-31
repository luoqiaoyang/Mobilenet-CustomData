## Mobilenet with custom data

[**Project Home Page**](https://github.com/luoqiaoyang/Mobilenet-CustomData)

The project task is to classify the fruits in supermarkets based on deep CNN model. But the machines in supermarkets only have CPU. So my team select mobilenet as the basic NN model and train it to classify different types of fruits.

This project uses pytorch to implement Mobilenet with Jupyter Notebook. The main dataset consists of imagenet data and custom data. 

Hope this could help you to enjoy deep learning.

### Version 1

In version 1, we conduct two experiments on fruits classification.

First, we use watermelon and kiwi pictures from imagenet to fune-tune our trained mobilenet model. Then we validate it by our own collect data. 

The own collect data size:

watermelon: 18 , kiwi: 16

And we can achieve 100% validation accuracy.

Then, we add three other fruits and repeat the experiment.

**Training Dataset**

We select Tiny Imagenet to train the mobilenet model.

Tiny Imagenet: http://cs231n.stanford.edu/tiny-imagenet-200.zip

**Fine-tune Dataset**

Data download from Imagenet, and we filter a part of them to fit the specific classification task:

apple: 570, avocado: 307, banana: 701, kiwi: 128, watermelon: 778

**Validation Dataset**

Collected from supermarket, used phone type: Huawei P10, iPhone 7 plus, iPhone X, data size: 102 pictures, including

apple: 21, avocado: 20, banana: 27, kiwi: 16, watermelon: 18

**Result**

5 Classes, 30 epochs

![cc_30](https://github.com/luoqiaoyang/Mobilenet-CustomData/raw/master/acc_30e.png)

###Version 2 

We expand val dataset size into 283 pictures, including

Apple: 44, Avocado: 58, Banana: 57, Kiwi: 61, Watermelon: 63

And we use different learning rate on different layers for the last trained model.

### Reference

1. A good pytorch Mobilenet implementation:

   https://github.com/marvis/pytorch-mobilenet

2. There are lots of good pytorch tutorials in github, here recommend a Chinese version for new beginners:

   https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch

3. A jupyter notebook sample for mobilenet:

   https://github.com/leonardoaraujosantos/MobileNet