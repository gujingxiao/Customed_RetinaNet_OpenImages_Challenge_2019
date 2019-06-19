# Customed_RetinaNet_OpenImages_Challenge_2019
Customed Modified Keras RetinaNet for OpenImages Challenge 2019

# Requirement
python 3.6
tensorflow-gpu  1.12.0
keras           2.2.4
本工程自带keras-retinanet，所以千万不要再安装了

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

3. 生成了每个level的信息后，运行 labelProcess/gen_exists_annotation.py， 这个脚本将会根据train中存放的图片，将所有存在图片的annotation提取出来，并随机按比例生成val-exists-annotations-bbox-level-X.csv和train-exists-annotations-bbox-level-X.csv。训练和验证时使用的就是最后生成的这两个csv文件，四个level的全部生成完毕后，与标签相关的工作就做完了

3. 然后运行find_image_parameters.py，生成训练集、验证集和测试集的信息
4. 所有文件生成好后，可以配置参数开始训练
