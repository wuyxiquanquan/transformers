# transformers
Data Augmentation for Classification, Point Localization,  Segmentation and so on by numpy and opencv.


**在有bbox的数据中，经过翻转，平移后的数据会出现丢失，所以bbox可能会超出图片的大小**
解决办法: clip bbox在图片的大小范围内。这样会导致有些框的width或height=0，注意调函数的时候去除掉这些框
