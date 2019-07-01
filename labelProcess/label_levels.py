# coding: utf-8
__modified_author__ = 'Jingxiao Gu : https://www.kaggle.com/gujingxiao0726'

import numpy as np
import gzip
import pickle
import cv2
import datetime
from collections import Counter
import operator
from PIL import Image
import json
import pyvips

ROOT_PATH = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/OpenimageV5/'
INPUT_PATH = ROOT_PATH + 'label/'
OUTPUT_PATH = ROOT_PATH + 'output/'
MODELS_PATH = ROOT_PATH + 'models/'
SUBM_PATH = ROOT_PATH + 'subm/'

# # Level 1 Top 0-80 Number:1418594 - 12135
# LEVEL_1_LABELS = ['Man', 'Tree', 'Human face', 'Person', 'Woman', 'Footwear', 'Window', 'Flower', 'Wheel', 'Car', 'Human hair',
#                   'Human arm', 'Human head', 'Girl', 'Building', 'House', 'Chair', 'Tire', 'Suit', 'Boy', 'Table', 'Skyscraper',
#                   'Land vehicle', 'Boat', 'Jeans', 'Human eye', 'Human hand', 'Human leg', 'Toy', 'Tower', 'Human nose',
#                   'Bicycle wheel', 'Glasses', 'Dress', 'Vehicle', 'Bird', 'Street light', 'Human mouth', 'Palm tree', 'Book',
#                   'Tableware', 'Drink', 'Bottle', 'Bicycle', 'Furniture', 'Sculpture', 'Flag', 'Dog', 'Dessert', 'Microphone',
#                   'Fruit', 'Jacket', 'Guitar', 'Drum', 'Sunglasses', 'Poster', 'Fish', 'Shelf', 'Houseplant', 'Flowerpot',
#                   'Airplane', 'Sports uniform', 'Door', 'Vegetable', 'Human ear', 'Animal', 'Shorts', 'Musical instrument',
#                   'Helmet', 'Bicycle helmet', 'Duck', 'Wine', 'Cat', 'Balloon', 'Motorcycle', 'Horse', 'Hat', 'Train',
#                   'Wine glass', 'Truck']
#
# # Level 2 Top 80-200 Number:12053 - 2388
# LEVEL_2_LABELS = ['Rose', 'Picture frame', 'Bus', 'Football helmet', 'Desk', 'Cattle', 'Bee', 'Tie', 'Butterfly', 'Swimwear',
#                   'Billboard', 'Goggles', 'Beer', 'Laptop', 'Cabinetry', 'Marine invertebrates', 'Insect', 'Trousers', 'Goose',
#                   'Strawberry', 'Vehicle registration plate', 'Van', 'Shirt', 'Traffic light', 'Bench', 'Umbrella', 'Sun hat',
#                   'Paddle', 'Tent', 'Ball', 'Sunflower', 'Coat', 'Lavender', 'Doll', 'Camera', 'Mobile phone', 'Tomato',
#                   'Office supplies', 'Orange', 'Pumpkin', 'Traffic sign', 'Computer monitor', 'Stairs', 'Candle', 'Cake',
#                   'Roller skates', 'Lantern', 'Plate', 'Box', 'Coffee cup', 'Coffee table', 'Bookcase', 'Watercraft', 'Football',
#                   'Office building', 'Maple', 'Curtain', 'Kitchen appliance', 'Muffin', 'Canoe', 'Computer keyboard', 'Swan',
#                   'Bowl', 'Mushroom', 'Cocktail', 'Drawer', 'Castle', 'Couch', 'Christmas tree', 'Taxi', 'Penguin', 'Cookie',
#                   'Apple', 'Swimming pool', 'Deer', 'Porch', 'Bread', 'Television', 'Fountain', 'Lifejacket', 'Lamp', 'Fedora',
#                   'Bed', 'Beetle', 'Pillow', 'Ski', 'Carnivore', 'Platter', 'Sheep', 'Elephant', 'Human beard', 'Boot',
#                   'High heels', 'Countertop', 'Salad', 'Cowboy hat', 'Seafood', 'Chicken', 'Coin', 'Monkey', 'Helicopter',
#                   'Tin can', 'Weapon', 'Sandal', 'Juice', 'Ice cream', 'Violin', 'Saucer', 'Grape', 'Cart', 'Bronze sculpture',
#                   'Necklace', 'Parachute', 'Skull', 'Surfboard', 'Rifle', 'Baseball glove', 'Handbag', 'Vase', 'Parrot']
#
# # Level 3 Top 200-400 Number:2384 - 481
# LEVEL_3_LABELS = ['Coffee', 'Scarf', 'Mug', 'Candy', 'Lily', 'Human foot', 'Luggage and bags', 'Goldfish',
#                   'Kitchen & dining room table', 'Lizard', 'French fries', 'Sushi', 'Barrel', 'Home appliance', 'Harbor seal',
#                   'Goat', 'Jellyfish', 'Spider', 'Pizza', 'Cello', 'Tortoise', 'Squirrel', 'Watch', 'Aircraft', 'studio couch',
#                   'Gondola', 'Egg', 'Moths and butterflies', 'Shrimp', 'Sea lion', 'Convenience store', 'Light bulb', 'Skateboard',
#                   'Waste container', 'Musical keyboard', 'Kite', 'Lemon', 'Marine mammal', 'Bull', 'Dinosaur', 'Falcon', 'Tank',
#                   'Spoon', 'Pen', 'Eagle', 'Tap', 'Brassiere', 'Fork', 'Owl', 'Lion', 'Sparrow', 'Sink', 'Rabbit', 'Pig', 'Banana',
#                   'Frog', 'Teddy bear', 'Mirror', 'Antelope', 'Invertebrate', 'Trumpet', 'Dolphin', 'Chest of drawers',
#                   'Lighthouse', 'Sofa bed', 'Dragonfly', 'Hamburger', 'Wheelchair', 'Carrot', 'Tripod', 'Earrings', 'Giraffe',
#                   'Sock', 'Snake', 'Piano', 'Cupboard', 'Tea', 'Camel', 'Shellfish', 'Grapefruit', 'Tiger', 'Skirt', 'Headphones',
#                   'Stool', 'Horn', 'Baseball bat', 'Clock', 'Backpack', 'Saxophone', 'Glove', 'Cucumber', 'Sandwich', 'Sea turtle',
#                   'Broccoli', 'Nightstand', 'Zebra', 'Mule', 'Toilet', 'Zucchini', 'Cannon', 'Crocodile', 'Wall clock', 'Bust',
#                   'Crab', 'Oyster', 'Whale', 'Whiteboard', 'Ladder', 'Plastic bag', 'Tennis racket', 'Barge', 'Tablet computer',
#                   'Tart', 'Accordion', 'Miniskirt', 'Trombone', 'Snowboard', 'Snail', 'Doughnut', 'Ant', 'Pear', 'Rocket',
#                   'Billiard table', 'Caterpillar', 'Coconut', 'Mouse', 'Knife', 'Table tennis racket', 'Watermelon', 'Alpaca',
#                   'Leopard', 'Bell pepper', 'Kangaroo', 'Pancake', 'Snowman', 'Pasta', 'Peach', 'Otter', 'Door handle', 'Willow',
#                   'Ladybug', 'Turkey', 'Computer Mouse', 'Wok', 'Handgun', 'Rhinoceros', 'Cheetah', 'Dice', 'Fireplace', 'Waffle',
#                   'Radish', 'Crown', 'Mechanical fan', 'Taco', 'Pomegranate', 'Polar bear', 'Volleyball', 'Pineapple', 'Kettle',
#                   'Washing machine', 'Bat', 'Sombrero', 'Brown bear', 'Bagel', 'Ostrich', 'Starfish', 'Oven', 'Teapot', 'Loveseat',
#                   'Suitcase', 'Shark', 'Chopsticks', 'Swim cap', 'Missile', 'Potato', 'Lobster', 'Golf cart', 'Bow and arrow',
#                   'Refrigerator', 'Jug', 'Jaguar', 'Shotgun', 'Reptile', 'Window blind', 'Raven', 'Sword', 'Segway', 'Fox',
#                   'Hamster', 'Bathtub', 'Jet ski', 'Gas stove', 'Scoreboard', 'Woodpecker', 'Beehive', 'Tennis ball', 'Nail',
#                   'Microwave oven', 'Hot dog', 'Plumbing fixture']
#
# # Level 4 Top 400-500 Number:478 - 14
# LEVEL_4_LABELS = ['Ceiling fan', 'Infant bed', 'Seat belt', 'Sewing machine', 'Ambulance', 'Croissant', 'Bidet', 'Cabbage',
#                   'Golf ball', 'Corded phone', 'Fire hydrant', 'Mango', 'Bear', 'Picnic basket', 'Belt', 'Dumbbell', 'Tiara',
#                   'Personal care', 'Scissors', 'Organ', 'Stop sign', 'Canary', 'Asparagus', 'Honeycomb', 'Raccoon', 'Toilet paper',
#                   'Frying pan', 'Artichoke', 'Squash', 'Filing cabinet', 'Dagger', 'Limousine', 'Snowmobile', 'Popcorn', 'Flute',
#                   'Bathroom cabinet', 'Kitchen knife', 'Pitcher', 'Towel', 'Stationary bicycle', 'Cake stand', 'Punching bag',
#                   'Coffeemaker', 'Common fig', 'Seahorse', 'Wood-burning stove', 'Snowplow', 'Pretzel', 'Rugby ball', 'Drinking straw',
#                   'Power plugs and sockets', 'Racket', 'Centipede', 'Telephone', 'Submarine sandwich', 'Dog bed', 'Printer', 'Burrito',
#                   'Blue jay', 'Adhesive tape', 'Ruler', 'Treadmill', 'Lynx', 'Shower', 'Blender', 'Harp', 'Porcupine', 'Guacamole',
#                   'Mixer', 'Cutting board', 'Harpsichord', 'Paper towel', 'Turtle', 'Wrench', 'Digital clock', 'Training bench',
#                   'Food processor', 'Salt and pepper shakers', 'Envelope', 'Stretcher', 'Alarm clock', 'Beaker', 'Oboe', 'Briefcase',
#                   'Crutch', 'Tick', 'Cricket ball', 'Slow cooker', 'Binoculars', 'Serving tray', 'Light switch', 'Flashlight',
#                   'Screwdriver', 'Ring binder', 'Measuring cup', 'Toaster', 'Spatula', 'Winter melon', 'Torch', 'Pressure cooker']


# Level 1 Top 0-80 Number:1418594 - 12135
LEVEL_1_LABELS = ['Man', 'Tree', 'Human face', 'Person', 'Woman', 'Footwear', 'Window', 'Flower', 'Wheel', 'Car',
                   'Girl', 'Building', 'House', 'Chair', 'Tire', 'Suit', 'Boy', 'Table', 'Skyscraper',
                  'Boat', 'Jeans', 'Tower', 'Bicycle wheel', 'Glasses', 'Dress', 'Bird', 'Palm tree',
                  'Bottle', 'Bicycle', 'Sculpture', 'Flag', 'Dog', 'Dessert', 'Microphone']

# Level 2 Top 80-200 Number:12053 - 2388
LEVEL_2_LABELS = ['Jacket', 'Guitar', 'Drum', 'Sunglasses', 'Poster', 'Fish', 'Houseplant', 'Flowerpot',
                  'Airplane', 'Sports uniform', 'Door', 'Shorts',
                  'Helmet', 'Bicycle helmet', 'Duck', 'Wine', 'Cat', 'Motorcycle', 'Horse', 'Train', 'Wine glass',
                  'Truck', 'Rose', 'Picture frame', 'Bus', 'Football helmet', 'Desk', 'Cattle', 'Bee', 'Tie', 'Butterfly',
                  'Swimwear', 'Billboard', 'Goggles', 'Beer', 'Laptop', 'Cabinetry', 'Marine invertebrates', 'Insect',
                  'Goose', 'Strawberry', 'Vehicle registration plate', 'Van', 'Shirt', 'Traffic light', 'Bench', 'Umbrella',
                  'Sun hat', 'Paddle', 'Tent', 'Ball', 'Sunflower', 'Coat', 'Lavender', 'Doll', 'Camera', 'Mobile phone', 'Tomato',
                  'Orange', 'Pumpkin', 'Traffic sign', 'Computer monitor', 'Stairs', 'Candle', 'Cake',
                  'Roller skates', 'Lantern', 'Plate', 'Box', 'Coffee cup', 'Coffee table', 'Bookcase', 'Football',
                  'Office building', 'Maple', 'Curtain', 'Muffin', 'Canoe', 'Computer keyboard', 'Swan',
                  'Bowl', 'Mushroom', 'Cocktail', 'Castle', 'Couch', 'Christmas tree', 'Taxi', 'Penguin', 'Cookie',
                  'Apple', 'Swimming pool', 'Deer', 'Bread', 'Television', 'Fountain', 'Lifejacket', 'Lamp', 'Fedora',
                  'Bed', 'Beetle', 'Pillow', 'Ski', 'Carnivore', 'Platter', 'Sheep', 'Elephant', 'Human beard', 'Boot',
                  'High heels', 'Countertop', 'Salad', 'Cowboy hat', 'Seafood', 'Chicken', 'Coin', 'Monkey', 'Helicopter',
                  'Tin can', 'Sandal', 'Juice', 'Ice cream', 'Violin', 'Saucer', 'Grape', 'Cart', 'Bronze sculpture',
                  'Necklace', 'Parachute', 'Skull', 'Rifle', 'Baseball glove', 'Handbag', 'Vase', 'Parrot']

# Level 3 Top 200-350 Number:2384 - 710
LEVEL_3_LABELS = ['Coffee', 'Scarf', 'Mug', 'Candy', 'Lily', 'Human foot', 'Luggage and bags', 'Goldfish', 'Kitchen & dining room table',
                  'Lizard', 'French fries', 'Sushi', 'Barrel', 'Home appliance', 'Harbor seal', 'Goat', 'Jellyfish', 'Spider',
                  'Pizza', 'Cello', 'Tortoise', 'Squirrel', 'Watch', 'Aircraft', 'studio couch', 'Gondola', 'Egg', 'Moths and butterflies',
                  'Shrimp', 'Sea lion', 'Convenience store', 'Light bulb', 'Skateboard', 'Waste container', 'Musical keyboard', 'Kite',
                  'Lemon', 'Marine mammal', 'Bull', 'Dinosaur', 'Falcon', 'Tank', 'Spoon', 'Pen', 'Eagle', 'Tap', 'Brassiere', 'Fork',
                  'Owl', 'Lion', 'Sparrow', 'Sink', 'Rabbit', 'Pig', 'Banana', 'Frog', 'Teddy bear', 'Mirror', 'Invertebrate', 'Antelope',
                  'Trumpet', 'Dolphin', 'Chest of drawers', 'Lighthouse', 'Sofa bed', 'Dragonfly', 'Hamburger', 'Wheelchair', 'Carrot',
                  'Tripod', 'Earrings', 'Giraffe', 'Sock', 'Snake', 'Piano', 'Cupboard', 'Tea', 'Camel', 'Shellfish', 'Grapefruit', 'Tiger',
                  'Skirt', 'Headphones', 'Stool', 'Horn', 'Baseball bat', 'Clock', 'Backpack', 'Saxophone', 'Glove', 'Cucumber', 'Sandwich',
                  'Sea turtle', 'Broccoli', 'Nightstand', 'Zebra', 'Mule', 'Toilet', 'Zucchini', 'Cannon', 'Crocodile', 'Wall clock',
                  'Bust', 'Crab', 'Oyster', 'Whale', 'Whiteboard', 'Ladder', 'Plastic bag', 'Tennis racket', 'Barge', 'Tablet computer',
                  'Tart', 'Accordion', 'Miniskirt', 'Trombone', 'Snowboard', 'Snail', 'Doughnut', 'Ant', 'Pear', 'Rocket', 'Billiard table',
                  'Caterpillar', 'Coconut', 'Mouse', 'Knife', 'Table tennis racket', 'Watermelon', 'Alpaca', 'Leopard', 'Bell pepper',
                  'Kangaroo', 'Pancake', 'Snowman', 'Pasta', 'Peach', 'Otter', 'Door handle', 'Willow', 'Turkey', 'Ladybug', 'Computer Mouse',
                  'Wok', 'Handgun', 'Rhinoceros', 'Cheetah', 'Dice', 'Fireplace', 'Waffle']

# Level 4 Top 350-500 Number:688 - 14
LEVEL_4_LABELS = ['Radish', 'Crown', 'Mechanical fan', 'Pomegranate', 'Taco', 'Polar bear', 'Volleyball', 'Pineapple', 'Kettle', 'Washing machine',
                  'Bat', 'Sombrero', 'Brown bear', 'Ostrich', 'Bagel', 'Starfish', 'Oven', 'Teapot', 'Loveseat', 'Suitcase', 'Shark', 'Chopsticks',
                  'Swim cap', 'Missile', 'Potato', 'Lobster', 'Golf cart', 'Bow and arrow', 'Refrigerator', 'Jug', 'Jaguar', 'Shotgun', 'Reptile',
                  'Window blind', 'Raven', 'Sword', 'Segway', 'Fox', 'Hamster', 'Bathtub', 'Jet ski', 'Gas stove', 'Scoreboard', 'Woodpecker',
                  'Beehive', 'Tennis ball', 'Nail', 'Microwave oven', 'Hot dog', 'Plumbing fixture', 'Ceiling fan', 'Infant bed', 'Seat belt',
                  'Sewing machine', 'Croissant', 'Ambulance', 'Bidet', 'Cabbage', 'Golf ball', 'Corded phone', 'Fire hydrant', 'Mango', 'Bear',
                  'Picnic basket', 'Belt', 'Dumbbell', 'Tiara', 'Personal care', 'Scissors', 'Organ', 'Stop sign', 'Canary', 'Asparagus',
                  'Honeycomb', 'Raccoon', 'Toilet paper', 'Frying pan', 'Artichoke', 'Squash', 'Filing cabinet', 'Dagger', 'Snowmobile',
                  'Limousine', 'Flute', 'Popcorn', 'Bathroom cabinet', 'Kitchen knife', 'Pitcher', 'Stationary bicycle', 'Towel', 'Cake stand',
                  'Punching bag', 'Coffeemaker', 'Common fig', 'Seahorse', 'Snowplow', 'Wood-burning stove', 'Rugby ball', 'Pretzel',
                  'Drinking straw', 'Power plugs and sockets', 'Racket', 'Centipede', 'Telephone', 'Submarine sandwich', 'Dog bed', 'Printer',
                  'Burrito', 'Blue jay', 'Adhesive tape', 'Ruler', 'Treadmill', 'Lynx', 'Shower', 'Blender', 'Harp', 'Porcupine', 'Guacamole',
                  'Mixer', 'Cutting board', 'Harpsichord', 'Paper towel', 'Turtle', 'Wrench', 'Digital clock', 'Training bench', 'Food processor',
                  'Salt and pepper shakers', 'Envelope', 'Stretcher', 'Alarm clock', 'Beaker', 'Oboe', 'Briefcase', 'Crutch', 'Tick',
                  'Cricket ball', 'Slow cooker', 'Binoculars', 'Serving tray', 'Light switch', 'Flashlight', 'Screwdriver', 'Ring binder',
                  'Measuring cup', 'Toaster', 'Spatula', 'Winter melon', 'Torch', 'Pressure cooker']

# Level Hard Very Hard Level! Eval Map less than 0.1
LEVEL_5_LABELS = ['Human hair', 'Human arm', 'Human head', 'Human eye', 'Human hand', 'Human leg', 'Human nose',
                     'Land vehicle', 'Vehicle', 'Street light', 'Human mouth', 'Tableware', 'Furniture', 'Shelf', 'Vegetable',
                     'Human ear', 'Musical instrument', 'Toy', 'Book', 'Drink', 'Fruit', 'Animal', 'Balloon', 'Hat',
                     'Office supplies', 'Kitchen appliance', 'Porch', 'Weapon', 'Trousers', 'Watercraft', 'Drawer', 'Surfboard']

def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3))


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'))


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def get_date_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


def sort_dict_by_values(a, reverse=True):
    sorted_x = sorted(a.items(), key=operator.itemgetter(1), reverse=reverse)
    return sorted_x


def value_counts_for_list(lst):
    a = dict(Counter(lst))
    a = sort_dict_by_values(a, True)
    return a


def read_single_image(path):
    use_pyvips = False
    try:
        if not use_pyvips:
            img = np.array(Image.open(path))
        else:
            # Much faster in case you have pyvips installed (uncomment import pyvips in top of file)
            img = pyvips.Image.new_from_file(path, access='sequential')
            img = np.ndarray(buffer=img.write_to_memory(),
                         dtype=np.uint8,
                         shape=[img.height, img.width, img.bands])
    except:
        try:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        except:
            print('Fail')
            return None

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 2:
        img = img[:, :, :1]

    if img.shape[2] == 1:
        img = np.concatenate((img, img, img), axis=2)

    if img.shape[2] > 3:
        img = img[:, :, :3]

    return img


def get_description_for_labels():
    out = open(INPUT_PATH + 'challenge-2019-classes-description-500.csv')
    lines = out.readlines()
    ret_1, ret_2 = dict(), dict()
    for l in lines:
        arr = l.strip().split(',')
        ret_1[arr[0]] = arr[1]
        ret_2[arr[1]] = arr[0]
    return ret_1, ret_2


def read_image_bgr_fast(path):
    img2 = read_single_image(path)
    # cv2.imshow('img2', img2)
    # cv2.waitKey(0)
    img2 = img2[:, :, ::-1]
    return img2


def get_subcategories(sub_cat, upper_cat, level, l, d1, sub):
    ret = []
    sub_cat[upper_cat] = ([], [])
    for j, k in enumerate(l[sub]):
        nm = d1[k['LabelName']]
        sub_cat[upper_cat][1].append(nm)
        if nm in sub_cat:
            continue
        ret.append(nm)
        if 'Subcategory' in k:
            get_subcategories(sub_cat, nm, level + 1, l, d1, 'Subcategory')
        else:
            sub_cat[nm] = ([upper_cat], [])
    return ret


def get_hierarchy_structures():
    sub_cat = dict()
    part_cat = dict()
    d1, d2 = get_description_for_labels()
    arr = json.load(open(INPUT_PATH + 'challenge-2019-label500-hierarchy.json', 'r'))
    lst = dict(arr.items())['Subcategory']
    for i, l in enumerate(lst):
        nm = d1[l['LabelName']]
        if 'Subcategory' in l:
            get_subcategories(sub_cat, nm, 1, l, d1, 'Subcategory')
        else:
            if nm in sub_cat:
                print('Strange!')
                exit()
            sub_cat[nm] = [], []
    return sub_cat


def set_parents(parents, name_list, l, d1):
    for j, k in enumerate(l['Subcategory']):
        nm = d1[k['LabelName']]
        parents[nm] += name_list
        if 'Subcategory' in k:
            set_parents(parents, name_list + [nm], k, d1)


def get_parents_labels():
    d1, d2 = get_description_for_labels()
    parents = dict()
    for r in d2.keys():
        parents[r] = []

    arr = json.load(open(INPUT_PATH + 'challenge-2019-label500-hierarchy.json', 'r'))
    lst = dict(arr.items())['Subcategory']
    for i, l in enumerate(lst):
        nm = d1[l['LabelName']]
        if 'Subcategory' in l:
            set_parents(parents, [nm], l, d1)
    # print(parents)
    for p in parents:
        parents[p] = list(set(parents[p]))
    return parents


def get_description_for_labels_500():
    out = open(INPUT_PATH + 'challenge-2019-classes-description-500.csv')
    lines = out.readlines()
    ret_1, ret_2 = dict(), dict()
    for l in lines:
        arr = l.strip().split(',')
        ret_1[arr[0]] = arr[1]
        ret_2[arr[1]] = arr[0]
    return ret_1, ret_2
