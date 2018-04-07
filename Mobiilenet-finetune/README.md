## Mobilenet Fine tune

This is a example to finetune mobilenet model with specific imagenet dataset. The validation dataset is custom dataset. 

I've pre-processed the validation with three different ways. In this example, the validation dataset with center crop has highest accuracy. The way is:

```
valid_tf = tfs.Compose([
    tfs.CenterCrop(224),
    tfs.ToTensor(),
    tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

