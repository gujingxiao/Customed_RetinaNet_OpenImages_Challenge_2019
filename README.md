# Customed_RetinaNet_OpenImages_Challenge_2019
Customed Modified Keras RetinaNet for OpenImages Challenge 2019

# 系统需求
python 3.6

tensorflow-gpu  1.12.0

keras           2.2.4

1T固态硬盘（机械硬盘巨慢...）

本工程自带keras-retinanet，所以千万不要再安装了

# 成绩记录

|Models|Loss Function|Base LR|Batch Size|LR_Decay|MAP|
|:---|:---|:---|:---|:---|:---|
|Resnet50|Focal Loss + NMS|0.0001|4|0.85|0.41625|

# 数据存放格式
1. 假设数据存放根目录为 OpenimageV5，进入到labelProcess中，将label-levels.py中ROOT_PATH改为OpenimageV5的路径

2. 本工程不使用官方提供的validation验证集，而是将train随机分出验证集，因为有参赛选手说train拥有更好的连续性和分布

3. 在OpenimageV5目录下建立文件夹 label， output， train， test

4. 将所有关于标签的文件都存放在label文件夹下
   -- label -- challenge-2019-classes-description-500.csv
            -- challenge-2019-label500-hierarchy.json
            -- challenge-2019-train-detection-bbox.csv
            -- challenge-2019-validation-detection-bbox.csv

5. test存放所有测试集图片，train存放所有训练集图片，如果目前是分文件夹存放，可以使用代码里labelProcess/resize_move_images.py来进行搬运，速度很快

6. 所有数据准备好后，开始处理数据

# 数据处理
1. 首先可以运行 labelProcess/analyse_class_distribution.py， 这个脚本帮助你理解四个level是如何进行划分的，并且查看四个level的数据分布

2. 掌握了分布后，要分别将四个level的annotations抽取出来，下面运行 labelProcess/create_files_for_training_by_levels.py，将会在OpenimageV5中指定文件夹中生成每个level的描述文件和训练annotations文件

3. 生成了每个level的信息后，运行 labelProcess/gen_exists_annotation.py， 这个脚本将会根据train中存放的图片，将所有存在图片的annotation提取出来，并随机按比例生成val-exists-annotations-bbox-level-X.csv和train-exists-annotations-bbox-level-X.csv。训练和验证时使用的就是最后生成的这两个csv文件，四个level的全部生成完毕后，与标签相关的工作就做完了（验证集比例可以自行在脚本中修改）

4. 然后运行 labelProcess/find_image_parameters.py， 统计并保存训练集、测试集的图像信息，这个步骤大概需要15-20mins

5. 建议下载好所有数据后进行数据处理，避免反复操作

# 训练过程
1. 训练脚本是 retinanet_training_oid_2019/train_oid_2019_resnet50.py，打开后里面有配置params的地方。目前resnet50已经基本上训练okay，还需要训练的是resnet101和resnet152

2. 训练时建议先直接使用OID 2018的预训练模型，四个level都可以直接使用，retinanet_resnet101_converted.h5和retinanet_resnet152_converted.h5，将模型路径填写在--weights里面; --label-level就是当前想要训练的level，可修改1,2,3,4,建议按顺序训练; --backbone这里修改为resnet101或resnet152； 图像尺寸这里建议不要修改，方便后面进行ensemble； --freeze--backbone这里一定先选择True，这样训练很快，上分也比较容易，等成绩稳定后，我们再放开这里

3. GPU的使用可以自行修改，需要注意的是，如果使用单GPU，保存的模型为snapshots，预测和验证前要先使用convert_retinanet_model.py来转换，否则会出现问题; 如果使用多GPU，保存模型就是weights，可以直接验证和预测

# 验证过程
1. 验证脚本是 retinanet_training_oid_2019/eval_oid_2019_resnet50.py，配置params的地方基本与训练相同

2. 目前代码中有些小问题，验证时只能使用单GPU，并且batch-size只能是1,否则验证结果可能会出错

# 预测过程
未完待续
