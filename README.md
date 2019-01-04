# gluon-east
reimplement EAST based gluoncv, which extends SegBaseModel

### Install dependencies

```shell
pip install -r requirement.txt
```

### pretrainmodels
you can set param `backbone='resnet50'`to get other backbone. Here are backbones in gluoncv:

```python
if backbone == 'resnet50':
    pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=True, **kwargs)
elif backbone == 'resnet101':
    pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=True, **kwargs)
elif backbone == 'resnet152':
    pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=True, **kwargs)
```
### Train

```

python scripts/train_east.py data_dir ckpt_path
```

### Test

```

python scripts/test_east.py

```

### TODO
- train and evaluation
- Bugs exists on Regression branch
