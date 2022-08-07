# CV_2022_cohort1
The working repo for the 2022 advance AI program - CV

- 0_image_format
处理raw data格式，全部转换成JPG格式，处理后的图片保存在data/data_jpg

- 1_augmentation
把JPG图片 augment

- 2.1_YOLO
提取Miranda标的 且保存在data/data_yolo_test2的图，放入yolo做training，训练weights
yolo threshold = 0.6, 低于threshold的box会被忽略。之后用best weights去predict所有data/data_jpg图片

- 2.2_classifier_tensorflow
用data/data_jpg里的 70%做training data，20%做val，10%留为test。
用pre-trained Xception epoch=20训练。根据最后的performance图，选择
epoch=12时的weights作为final model

- 2.3_classifier_pytorch
- 3_error_analysis
	用epoch=12在2.2_classifier_tensorflow训练出来的CNN classifier，去一张张classify test data(10%)
