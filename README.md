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
|Resnet50|Focal Loss + NMS|0.0001|4|0.85|0.44599|

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

4. 最开始训练时，会根据annotation信息生成一个json，这时会报一些warning，不用担心，等待处理完毕就好，大概需要3-5分钟; 后面再训练，就会直接加载生成好的json，不用再处理了

# 验证过程
1. 验证脚本是 retinanet_training_oid_2019/eval_oid_2019_resnet50.py，配置params的地方基本与训练相同

2. 目前代码中有些小问题，验证时只能使用单GPU，并且batch-size只能是1,否则验证结果可能会出错

# 预测过程
1. 整个预测过程共分为4个步骤：(1）分别生成每个level对应的预测文件 （2)每个level使用预测文件生成csv结果 (3）根据生成结果扩展到全label (4） 将所有结果进行合并

2. 首先使用retinanet_inference_submission/retinanet_inference_make_submission.py，修改label_level然后修改model path至与label相匹配的模型，修改inference_predict为True，然后即可执行;这里会将test中99999图片都跑一遍，如果你的卡很多，可以修改gpu_use到不同的卡，多次运行该脚本，可以达到并行运算的效果，预计4块卡同时跑，一个小时可以预测完毕。

3. 预测完毕后，将inference_predict改为False，这时iou_thr为0。55不要动，可以修改skip_box_condifence为0。05 0。1 0。15等不同的值，数值越小得分越高，但是生成出来的预测csv也会非常巨大。建议测评时使用0。1。修改好后，运行此脚本等待生成完毕。每个level都需要生成一次。

4。 生成预测csv完毕后，进入到create_higher_level_predctions.py中，修改level和相应路径，运行等待生成扩展csv，每个level都需要生成一次。

5。 扩展csv都生成完毕后，进入到concat_all_levels_submission.py中，修改不同文件路径，然后运行，等待用于提交的csv生成。生成完毕后，提交文件非常大，建议先压缩成zip格式，再进行上传，上传需要翻墙。
