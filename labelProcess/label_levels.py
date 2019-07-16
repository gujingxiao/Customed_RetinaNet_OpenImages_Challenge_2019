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
# LEVEL_1_LABELS = ['Man', 'Tree', 'Human face', 'Person', 'Woman', 'Footwear', 'Window', 'Flower', 'Wheel', 'Car',
#                    'Girl', 'Building', 'House', 'Chair', 'Tire', 'Suit', 'Boy', 'Table', 'Skyscraper',
#                   'Boat', 'Jeans', 'Tower', 'Bicycle wheel', 'Glasses', 'Dress', 'Bird', 'Palm tree',
#                   'Bottle', 'Bicycle', 'Sculpture', 'Flag', 'Dog', 'Dessert', 'Microphone']
#
# # Level 2 Top 80-200 Number:12053 - 2388
# LEVEL_2_LABELS = ['Jacket', 'Guitar', 'Drum', 'Sunglasses', 'Poster', 'Fish', 'Houseplant', 'Flowerpot',
#                   'Airplane', 'Sports uniform', 'Door', 'Shorts',
#                   'Helmet', 'Bicycle helmet', 'Duck', 'Wine', 'Cat', 'Motorcycle', 'Horse', 'Train', 'Wine glass',
#                   'Truck', 'Rose', 'Picture frame', 'Bus', 'Football helmet', 'Desk', 'Cattle', 'Bee', 'Tie', 'Butterfly',
#                   'Swimwear', 'Billboard', 'Goggles', 'Beer', 'Laptop', 'Cabinetry', 'Marine invertebrates', 'Insect',
#                   'Goose', 'Strawberry', 'Vehicle registration plate', 'Van', 'Shirt', 'Traffic light', 'Bench', 'Umbrella',
#                   'Sun hat', 'Paddle', 'Tent', 'Ball', 'Sunflower', 'Coat', 'Lavender', 'Doll', 'Camera', 'Mobile phone', 'Tomato',
#                   'Orange', 'Pumpkin', 'Traffic sign', 'Computer monitor', 'Stairs', 'Candle', 'Cake',
#                   'Roller skates', 'Lantern', 'Plate', 'Box', 'Coffee cup', 'Coffee table', 'Bookcase', 'Football',
#                   'Office building', 'Maple', 'Curtain', 'Muffin', 'Canoe', 'Computer keyboard', 'Swan',
#                   'Bowl', 'Mushroom', 'Cocktail', 'Castle', 'Couch', 'Christmas tree', 'Taxi', 'Penguin', 'Cookie',
#                   'Apple', 'Swimming pool', 'Deer', 'Bread', 'Television', 'Fountain', 'Lifejacket', 'Lamp', 'Fedora',
#                   'Bed', 'Beetle', 'Pillow', 'Ski', 'Carnivore', 'Platter', 'Sheep', 'Elephant', 'Human beard', 'Boot',
#                   'High heels', 'Countertop', 'Salad', 'Cowboy hat', 'Seafood', 'Chicken', 'Coin', 'Monkey', 'Helicopter',
#                   'Tin can', 'Sandal', 'Juice', 'Ice cream', 'Violin', 'Saucer', 'Grape', 'Cart', 'Bronze sculpture',
#                   'Necklace', 'Parachute', 'Skull', 'Rifle', 'Baseball glove', 'Handbag', 'Vase', 'Parrot']
#
# # Level 3 Top 200-350 Number:2384 - 710
# LEVEL_3_LABELS = ['Coffee', 'Scarf', 'Mug', 'Candy', 'Lily', 'Human foot', 'Luggage and bags', 'Goldfish', 'Kitchen & dining room table',
#                   'Lizard', 'French fries', 'Sushi', 'Barrel', 'Home appliance', 'Harbor seal', 'Goat', 'Jellyfish', 'Spider',
#                   'Pizza', 'Cello', 'Tortoise', 'Squirrel', 'Watch', 'Aircraft', 'studio couch', 'Gondola', 'Egg', 'Moths and butterflies',
#                   'Shrimp', 'Sea lion', 'Convenience store', 'Light bulb', 'Skateboard', 'Waste container', 'Musical keyboard', 'Kite',
#                   'Lemon', 'Marine mammal', 'Bull', 'Dinosaur', 'Falcon', 'Tank', 'Spoon', 'Pen', 'Eagle', 'Tap', 'Brassiere', 'Fork',
#                   'Owl', 'Lion', 'Sparrow', 'Sink', 'Rabbit', 'Pig', 'Banana', 'Frog', 'Teddy bear', 'Mirror', 'Invertebrate', 'Antelope',
#                   'Trumpet', 'Dolphin', 'Chest of drawers', 'Lighthouse', 'Sofa bed', 'Dragonfly', 'Hamburger', 'Wheelchair', 'Carrot',
#                   'Tripod', 'Earrings', 'Giraffe', 'Sock', 'Snake', 'Piano', 'Cupboard', 'Tea', 'Camel', 'Shellfish', 'Grapefruit', 'Tiger',
#                   'Skirt', 'Headphones', 'Stool', 'Horn', 'Baseball bat', 'Clock', 'Backpack', 'Saxophone', 'Glove', 'Cucumber', 'Sandwich',
#                   'Sea turtle', 'Broccoli', 'Nightstand', 'Zebra', 'Mule', 'Toilet', 'Zucchini', 'Cannon', 'Crocodile', 'Wall clock',
#                   'Bust', 'Crab', 'Oyster', 'Whale', 'Whiteboard', 'Ladder', 'Plastic bag', 'Tennis racket', 'Barge', 'Tablet computer',
#                   'Tart', 'Accordion', 'Miniskirt', 'Trombone', 'Snowboard', 'Snail', 'Doughnut', 'Ant', 'Pear', 'Rocket', 'Billiard table',
#                   'Caterpillar', 'Coconut', 'Mouse', 'Knife', 'Table tennis racket', 'Watermelon', 'Alpaca', 'Leopard', 'Bell pepper',
#                   'Kangaroo', 'Pancake', 'Snowman', 'Pasta', 'Peach', 'Otter', 'Door handle', 'Willow', 'Turkey', 'Ladybug', 'Computer Mouse',
#                   'Wok', 'Handgun', 'Rhinoceros', 'Cheetah', 'Dice', 'Fireplace', 'Waffle']
#
# # Level 4 Top 350-500 Number:688 - 14
# LEVEL_4_LABELS = ['Radish', 'Crown', 'Mechanical fan', 'Pomegranate', 'Taco', 'Polar bear', 'Volleyball', 'Pineapple', 'Kettle', 'Washing machine',
#                   'Bat', 'Sombrero', 'Brown bear', 'Ostrich', 'Bagel', 'Starfish', 'Oven', 'Teapot', 'Loveseat', 'Suitcase', 'Shark', 'Chopsticks',
#                   'Swim cap', 'Missile', 'Potato', 'Lobster', 'Golf cart', 'Bow and arrow', 'Refrigerator', 'Jug', 'Jaguar', 'Shotgun', 'Reptile',
#                   'Window blind', 'Raven', 'Sword', 'Segway', 'Fox', 'Hamster', 'Bathtub', 'Jet ski', 'Gas stove', 'Scoreboard', 'Woodpecker',
#                   'Beehive', 'Tennis ball', 'Nail', 'Microwave oven', 'Hot dog', 'Plumbing fixture', 'Ceiling fan', 'Infant bed', 'Seat belt',
#                   'Sewing machine', 'Croissant', 'Ambulance', 'Bidet', 'Cabbage', 'Golf ball', 'Corded phone', 'Fire hydrant', 'Mango', 'Bear',
#                   'Picnic basket', 'Belt', 'Dumbbell', 'Tiara', 'Personal care', 'Scissors', 'Organ', 'Stop sign', 'Canary', 'Asparagus',
#                   'Honeycomb', 'Raccoon', 'Toilet paper', 'Frying pan', 'Artichoke', 'Squash', 'Filing cabinet', 'Dagger', 'Snowmobile',
#                   'Limousine', 'Flute', 'Popcorn', 'Bathroom cabinet', 'Kitchen knife', 'Pitcher', 'Stationary bicycle', 'Towel', 'Cake stand',
#                   'Punching bag', 'Coffeemaker', 'Common fig', 'Seahorse', 'Snowplow', 'Wood-burning stove', 'Rugby ball', 'Pretzel',
#                   'Drinking straw', 'Power plugs and sockets', 'Racket', 'Centipede', 'Telephone', 'Submarine sandwich', 'Dog bed', 'Printer',
#                   'Burrito', 'Blue jay', 'Adhesive tape', 'Ruler', 'Treadmill', 'Lynx', 'Shower', 'Blender', 'Harp', 'Porcupine', 'Guacamole',
#                   'Mixer', 'Cutting board', 'Harpsichord', 'Paper towel', 'Turtle', 'Wrench', 'Digital clock', 'Training bench', 'Food processor',
#                   'Salt and pepper shakers', 'Envelope', 'Stretcher', 'Alarm clock', 'Beaker', 'Oboe', 'Briefcase', 'Crutch', 'Tick',
#                   'Cricket ball', 'Slow cooker', 'Binoculars', 'Serving tray', 'Light switch', 'Flashlight', 'Screwdriver', 'Ring binder',
#                   'Measuring cup', 'Toaster', 'Spatula', 'Winter melon', 'Torch', 'Pressure cooker']
#
# # Level Hard Very Hard Level! Eval Map less than 0.1
# LEVEL_5_LABELS = ['Human hair', 'Human arm', 'Human head', 'Human eye', 'Human hand', 'Human leg', 'Human nose',
#                      'Land vehicle', 'Vehicle', 'Street light', 'Human mouth', 'Tableware', 'Furniture', 'Shelf', 'Vegetable',
#                      'Human ear', 'Musical instrument', 'Toy', 'Book', 'Drink', 'Fruit', 'Animal', 'Balloon', 'Hat',
#                      'Office supplies', 'Kitchen appliance', 'Porch', 'Weapon', 'Trousers', 'Watercraft', 'Drawer', 'Surfboard']

# Level 1 Top 0-38 Number:1418594 - 23566
LEVEL_1_LABELS = ['Man', 'Human face', 'Woman', 'Window', 'Wheel', 'Girl', 'House', 'Chair', 'Tire', 'Suit',
                  'Boy', 'Skyscraper', 'Jeans', 'Tower', 'Bicycle wheel', 'Glasses', 'Dress', 'Street light', 'Palm tree',
                  'Bottle', 'Bicycle', 'Flag', 'Dog', 'Microphone', 'Jacket', 'Guitar', 'Drum', 'Sunglasses', 'Poster']

# Level 2 Top 38-173 Number:22899 - 2088
LEVEL_2_LABELS = ['Shelf', 'Houseplant', 'Flowerpot', 'Airplane', 'Sports uniform', 'Door', 'Shorts',
                  'Bicycle helmet', 'Duck', 'Wine', 'Cat', 'Balloon', 'Motorcycle', 'Horse', 'Train', 'Wine glass',
                  'Truck', 'Rose', 'Picture frame', 'Bus', 'Football helmet', 'Desk', 'Cattle', 'Bee', 'Tie', 'Butterfly',
                  'Swimwear', 'Billboard', 'Goggles', 'Beer', 'Laptop', 'Cabinetry', 'Goose', 'Strawberry',
                  'Vehicle registration plate', 'Van', 'Shirt', 'Traffic light', 'Bench', 'Umbrella', 'Sun hat',
                  'Paddle', 'Tent', 'Sunflower', 'Coat', 'Lavender', 'Doll', 'Camera', 'Mobile phone', 'Tomato',
                  'Orange', 'Pumpkin', 'Computer monitor', 'Stairs', 'Candle', 'Cake', 'Roller skates', 'Lantern',
                  'Plate', 'Box', 'Coffee cup', 'Coffee table', 'Bookcase', 'Football', 'Office building', 'Maple',
                  'Curtain', 'Muffin', 'Canoe', 'Computer keyboard', 'Swan', 'Bowl', 'Mushroom', 'Cocktail', 'Drawer',
                  'Castle', 'Christmas tree', 'Taxi', 'Penguin', 'Cookie', 'Apple', 'Swimming pool', 'Deer', 'Porch',
                  'Bread', 'Television', 'Fountain', 'Lifejacket', 'Lamp', 'Fedora', 'Pillow', 'Ski', 'Platter', 'Sheep',
                  'Elephant', 'Human beard', 'Boot', 'High heels', 'Countertop', 'Salad', 'Cowboy hat', 'Chicken',
                  'Coin', 'Monkey', 'Helicopter', 'Tin can', 'Sandal', 'Juice', 'Ice cream', 'Violin', 'Saucer', 'Grape',
                  'Cart', 'Bronze sculpture', 'Necklace', 'Parachute', 'Skull', 'Surfboard', 'Rifle', 'Baseball glove',
                  'Handbag', 'Vase', 'Parrot', 'Coffee', 'Scarf', 'Mug', 'Candy', 'Lily', 'Goldfish',
                  'Kitchen & dining room table', 'Lizard', 'French fries', 'Sushi']

# Level 3 Top 173-308 Number:2086 - 661
LEVEL_3_LABELS = ['Barrel', 'Harbor seal', 'Goat', 'Jellyfish', 'Spider', 'Pizza', 'Cello', 'Tortoise', 'Squirrel',
                  'Watch', 'studio couch', 'Gondola', 'Egg', 'Shrimp', 'Sea lion', 'Convenience store', 'Light bulb',
                  'Skateboard', 'Waste container', 'Musical keyboard', 'Kite', 'Lemon', 'Bull', 'Dinosaur', 'Falcon',
                  'Tank', 'Spoon', 'Pen', 'Eagle', 'Tap', 'Brassiere', 'Fork', 'Owl', 'Lion', 'Sparrow', 'Sink', 'Rabbit',
                  'Pig', 'Banana', 'Frog', 'Teddy bear', 'Mirror', 'Antelope', 'Trumpet', 'Dolphin', 'Chest of drawers',
                  'Lighthouse', 'Sofa bed', 'Dragonfly', 'Hamburger', 'Wheelchair', 'Carrot', 'Tripod', 'Earrings',
                  'Giraffe', 'Sock', 'Snake', 'Piano', 'Cupboard', 'Tea', 'Camel', 'Grapefruit', 'Tiger', 'Headphones',
                  'Stool', 'Horn', 'Baseball bat', 'Backpack', 'Saxophone', 'Cucumber', 'Sea turtle', 'Broccoli',
                  'Nightstand', 'Zebra', 'Mule', 'Toilet', 'Zucchini', 'Cannon', 'Crocodile', 'Wall clock', 'Bust',
                  'Crab', 'Oyster', 'Whale', 'Whiteboard', 'Ladder', 'Plastic bag', 'Tennis racket', 'Barge',
                  'Tablet computer', 'Tart', 'Accordion', 'Miniskirt', 'Trombone', 'Snowboard', 'Snail', 'Doughnut',
                  'Ant', 'Pear', 'Rocket', 'Billiard table', 'Caterpillar', 'Coconut', 'Mouse', 'Knife',
                  'Table tennis racket', 'Watermelon', 'Alpaca', 'Leopard', 'Bell pepper', 'Kangaroo', 'Pancake',
                  'Snowman', 'Pasta', 'Peach', 'Otter', 'Door handle', 'Willow', 'Turkey', 'Ladybug', 'Computer Mouse',
                  'Wok', 'Handgun', 'Rhinoceros', 'Cheetah', 'Dice', 'Fireplace', 'Waffle', 'Radish', 'Crown',
                  'Mechanical fan', 'Taco', 'Pomegranate', 'Polar bear', 'Volleyball']

# Level 4 Top 308-443 Number:660 - 14
LEVEL_4_LABELS = ['Pineapple', 'Kettle', 'Washing machine', 'Bat', 'Sombrero', 'Brown bear', 'Bagel', 'Ostrich', 'Starfish',
                  'Oven', 'Teapot', 'Loveseat', 'Suitcase', 'Shark', 'Chopsticks', 'Swim cap', 'Missile', 'Potato',
                  'Lobster', 'Golf cart', 'Bow and arrow', 'Refrigerator', 'Jug', 'Jaguar', 'Shotgun', 'Window blind',
                  'Raven', 'Sword', 'Segway', 'Fox', 'Hamster', 'Bathtub', 'Jet ski', 'Gas stove', 'Scoreboard', 'Woodpecker',
                  'Beehive', 'Tennis ball', 'Nail', 'Microwave oven', 'Hot dog', 'Ceiling fan', 'Infant bed', 'Seat belt',
                  'Sewing machine', 'Croissant', 'Ambulance', 'Bidet', 'Cabbage', 'Golf ball', 'Corded phone', 'Fire hydrant',
                  'Mango', 'Picnic basket', 'Belt', 'Dumbbell', 'Tiara', 'Scissors', 'Organ', 'Stop sign', 'Asparagus',
                  'Canary', 'Honeycomb', 'Raccoon', 'Frying pan', 'Toilet paper', 'Artichoke', 'Filing cabinet', 'Dagger',
                  'Snowmobile', 'Limousine', 'Popcorn', 'Flute', 'Bathroom cabinet', 'Kitchen knife', 'Pitcher',
                  'Stationary bicycle', 'Towel', 'Cake stand', 'Punching bag', 'Coffeemaker', 'Common fig', 'Seahorse',
                  'Snowplow', 'Wood-burning stove', 'Rugby ball', 'Pretzel', 'Drinking straw', 'Power plugs and sockets',
                  'Centipede', 'Submarine sandwich', 'Dog bed', 'Printer', 'Burrito', 'Blue jay', 'Adhesive tape', 'Ruler',
                  'Treadmill', 'Lynx', 'Shower', 'Blender', 'Harp', 'Porcupine', 'Guacamole', 'Mixer', 'Cutting board',
                  'Harpsichord', 'Paper towel', 'Wrench', 'Digital clock', 'Training bench', 'Food processor',
                  'Salt and pepper shakers', 'Envelope', 'Stretcher', 'Alarm clock', 'Beaker', 'Oboe', 'Briefcase',
                  'Crutch', 'Tick', 'Cricket ball', 'Slow cooker', 'Binoculars', 'Serving tray', 'Light switch', 'Flashlight',
                  'Screwdriver', 'Ring binder', 'Measuring cup', 'Toaster', 'Spatula', 'Winter melon', 'Torch', 'Pressure cooker']

# Level Hard Very Hard Level! Eval Map less than 0.1
LEVEL_5_LABELS = ['Human hair', 'Human arm', 'Human head', 'Human eye', 'Human hand', 'Human leg', 'Human nose',
                  'Human mouth', 'Book', 'Human ear', 'Human foot']


# Level Parents 6: 1051344 - 18621
LEVEL_6_LABELS = ['Tree', 'Person', 'Footwear', 'Flower', 'Car', 'Building', 'Table', 'Land vehicle', 'Boat', 'Toy',
                  'Vehicle', 'Bird', 'Tableware', 'Drink', 'Furniture', 'Sculpture', 'Dessert', 'Fruit', 'Fish', 'Vegetable']


# Level Parents 7: 17442 - 205
LEVEL_7_LABELS = ['Animal', 'Musical instrument', 'Helmet', 'Hat', 'Marine invertebrates', 'Insect', 'Trousers', 'Ball',
                  'Office supplies', 'Traffic sign', 'Watercraft', 'Kitchen appliance', 'Couch', 'Bed', 'Beetle', 'Carnivore',
                  'Seafood', 'Weapon', 'Luggage and bags', 'Home appliance', 'Aircraft', 'Moths and butterflies', 'Marine mammal',
                  'Invertebrate', 'Shellfish', 'Skirt', 'Clock', 'Glove', 'Sandwich', 'Reptile', 'Plumbing fixture', 'Bear',
                  'Personal care', 'Squash', 'Racket', 'Telephone', 'Turtle']

# Children Level
LEVEL_CHILDREN = ['/m/03qrc', '/m/02wv84t', '/m/02jz0l', '/m/09ld4', '/m/01h44', '/m/0n28_', '/m/071qp', '/m/01xqw', '/m/01vbnl',
                  '/m/0by6g', '/m/03d443', '/m/03fwl', '/m/0162_1', '/m/0175cv', '/m/03fj2', '/m/01_5g', '/m/0h8n6ft', '/m/08pbxl',
                  '/m/04m9y', '/m/0dtln', '/m/031b6r', '/m/0388q', '/m/0_k2', '/m/099ssp', '/m/0h8my_4', '/m/03ldnb', '/m/01j51',
                  '/m/0642b4', '/m/0663v', '/m/054_l', '/m/0bh9flk', '/m/05kyg_', '/m/083wq', '/m/0_cp5', '/m/08hvt4', '/m/0mkg',
                  '/m/084rd', '/m/04yx4', '/m/09728', '/m/030610', '/m/02068x', '/m/05_5p_0', '/m/03m3pdh', '/m/0bjyj5', '/m/0cjs7',
                  '/m/0323sq', '/m/03tw93', '/m/0703r8', '/m/03g8mr', '/m/026qbn5', '/m/0pcr', '/m/0h23m', '/m/03bk1', '/m/01226z',
                  '/m/01dwwc', '/m/035r7c', '/m/09rvcxw', '/m/044r5d', '/m/04hgtk', '/m/0174k2', '/m/07crc', '/m/03m5k', '/m/025nd',
                  '/m/058qzx', '/m/04ctx', '/m/017ftj', '/m/039xj_', '/m/0ft9s', '/m/04kkgm', '/m/0f9_l', '/m/0hnnb', '/m/0d8zb',
                  '/m/0cydv', '/m/06pcq', '/m/0342h', '/m/07j87', '/m/0k65p', '/m/04zwwv', '/m/0c29q', '/m/01cmb2', '/m/02pdsw',
                  '/m/0fj52s', '/m/04p0qw', '/m/0k1tl', '/m/02pjr4', '/m/03k3r', '/m/047j0r', '/m/01j5ks', '/m/01xyhv', '/m/015h_t',
                  '/m/01f8m5', '/m/0cd4d', '/m/01pns0', '/m/0kpqd', '/m/04dr76w', '/m/05z55', '/m/06z37_', '/m/09tvcd', '/m/01599',
                  '/m/04tn4x', '/m/0449p', '/m/01rkbr', '/m/02d9qx', '/m/02s195', '/m/054fyh', '/m/01y9k5', '/m/01llwg', '/m/078jl',
                  '/m/07gql', '/m/0gm28', '/m/0bt_c3', '/m/01h8tj', '/m/01s55n', '/m/0dftk', '/m/02jfl0', '/m/061hd_', '/m/02f9f_',
                  '/m/03y6mg', '/m/0dv5r', '/m/0jy4k', '/m/09ddx', '/m/02wbtzl', '/m/02vqfm', '/m/07clx', '/m/034c16', '/m/0199g',
                  '/m/0h8ntjv', '/m/050gv4', '/m/03rszm', '/m/0cnyhnx', '/m/06k2mb', '/m/04h8sr', '/m/057cc', '/m/03jbxj', '/m/03dnzn',
                  '/m/0d20w4', '/m/0cyfs', '/m/07r04', '/m/0319l', '/m/0fly7', '/m/03v5tg', '/m/02tsc9', '/m/07c52', '/m/0dzf4',
                  '/m/09g1w', '/m/02p3w7d', '/m/0nl46', '/m/01x_v', '/m/01xq0k1', '/m/07fbm7', '/m/029bxz', '/m/052sf', '/m/01r546',
                  '/m/0kmg4', '/m/02x8cch', '/m/079cl', '/m/01nq26', '/m/015wgc', '/m/0djtd', '/m/01g3x7', '/m/06__v', '/m/09ct_',
                  '/m/024g6', '/m/02zn6n', '/m/0fz0h', '/m/01n4qj', '/m/06mf6', '/m/0fqfqc', '/m/02hj4', '/m/07030', '/m/0cdn1',
                  '/m/0pg52', '/m/02jnhm', '/m/021mn', '/m/0h8n6f9', '/m/04y4h8h', '/m/0mw_6', '/m/063rgb', '/m/0hkxq', '/m/0h8l4fh',
                  '/m/084zz', '/m/01j61q', '/m/032b3c', '/m/0h9mv', '/m/01gkx_', '/m/0cyf8', '/m/050k8', '/m/01b9xk', '/m/06j2d',
                  '/m/0ph39', '/m/02ctlc', '/m/03q5c7', '/m/05r5c', '/m/078n6m', '/m/076bq', '/m/09k_b', '/m/0b_rs', '/m/01tcjp',
                  '/m/01d40f', '/m/0h8n27j', '/m/057p5t', '/m/0dt3t', '/m/01bms0', '/m/01z1kdw', '/m/0k0pj', '/m/01xs3r', '/m/0jly1',
                  '/m/04m6gz', '/m/07dd4', '/m/011k07', '/m/03q5t', '/m/0167gd', '/m/05zsy', '/m/07kng9', '/m/0jyfg', '/m/013y1f',
                  '/m/01hrv5', '/m/07cmd', '/m/0cdl1', '/m/0176mf', '/m/02p5f1q', '/m/0dbvp', '/m/01dy8n', '/m/0ftb8', '/m/025dyy',
                  '/m/0584n8', '/m/0242l', '/m/02jvh9', '/m/03120', '/m/01xygc', '/m/0h8n5zk', '/m/01krhy', '/m/09kx5', '/m/03m3vtv',
                  '/m/0fx9l', '/m/020kz', '/m/07xyvk', '/m/0qmmr', '/m/01dwsz', '/m/01bqk0', '/m/01k6s3', '/m/04gth', '/m/014y4n',
                  '/m/029tx', '/m/01n5jq', '/m/0hg7b', '/m/016m2d', '/m/01jfsr', '/m/076lb9', '/m/0nybt', '/m/0fqt361', '/m/05gqfk',
                  '/m/04_sv', '/m/04g2r', '/m/0cmx8', '/m/054xkw', '/m/081qc', '/m/05vtc', '/m/027pcv', '/m/014j1m', '/m/0b3fp9',
                  '/m/0152hh', '/m/09f_2', '/m/047v4b', '/m/0fldg', '/m/03s_tn', '/m/033rq4', '/m/04yqq2', '/m/01x3jk', '/m/0l14j_',
                  '/m/01s105', '/m/0cffdh', '/m/09b5t', '/m/0120dh', '/m/0f6wt', '/m/02l8p9', '/m/0jwn_', '/m/07dm6', '/m/061_f',
                  '/m/07c6l', '/m/0420v5', '/m/0hqkz', '/m/0czz2', '/m/01lsmm', '/m/0fszt', '/m/0gxl3', '/m/0h8lkj8', '/m/02zvsm',
                  '/m/05n4y', '/m/06y5r', '/m/05441v', '/m/0dq75', '/m/06c54', '/m/0130jx', '/m/02g30s', '/m/0gjbg72', '/m/014sv8',
                  '/m/0fp6w', '/m/0h8mzrc', '/m/0cmf2', '/m/025rp__', '/m/0c568', '/m/0hdln', '/m/0306r', '/m/06_72j', '/m/06ncr',
                  '/m/02x984l', '/m/0gv1x', '/m/0h8mhzd', '/m/02_n6y', '/m/0ll1f78', '/m/0jqgx', '/m/0gj37', '/m/05r655', '/m/04169hn',
                  '/m/02pv19', '/m/01m4t', '/m/02d1br', '/m/0crjs', '/m/03jm5', '/m/0frqm', '/m/01yx86', '/m/019w40', '/m/0220r2',
                  '/m/05ctyq', '/m/026t6', '/m/0dzct', '/m/01kb5b', '/m/015x4r', '/m/05kms', '/m/07y_7', '/m/06_fw', '/m/0633h',
                  '/m/01b7fy', '/m/096mb', '/m/01gllr', '/m/0283dt1', '/m/02lbcq', '/m/01fh4r', '/m/0fm3zh', '/m/015x5n', '/m/09csl',
                  '/m/0dkzw', '/m/03__z0', '/m/02h19r', '/m/02rgn06', '/m/07v9_z', '/m/04ylt', '/m/0jg57', '/m/01940j', '/m/0bt9lr',
                  '/m/03nfch', '/m/01mzpv', '/m/0wdt60w', '/m/01jfm_', '/m/033cnk', '/m/01h3n', '/m/01m2v', '/m/0d4v4', '/m/04h7h',
                  '/m/05z6w', '/m/09qck', '/m/0fbw6', '/m/0cvnqh', '/m/0lt4_', '/m/080hkjn', '/m/02w3r3', '/m/0grw1', '/m/012w5l',
                  '/m/018p4k', '/m/03grzl', '/m/019h78', '/m/073bxn', '/m/01bl7v', '/m/03fp41', '/m/04v6l4', '/m/0gjkl', '/m/01fdzj',
                  '/m/0ccs93', '/m/03bbps', '/m/0llzx', '/m/021sj1', '/m/01dxs', '/m/02y6n', '/m/012n7d', '/m/03c7gz', '/m/020lf',
                  '/m/0cjq5', '/m/07qxg_', '/m/01b638', '/m/03p3bw', '/m/01lynh', '/m/01bjv', '/m/06nrc', '/m/01btn', '/m/0dbzx',
                  '/m/029b3', '/m/0cyhj_', '/m/0c06p', '/m/01gmv2', '/m/03bt1vf', '/m/06m11', '/m/01fb_0', '/m/02gzp', '/m/01c648',
                  '/m/0cxn2', '/m/040b_t', '/m/031n1', '/m/02522', '/m/0d5gx', '/m/02dgv', '/m/01j3zr', '/m/02fq_6', '/m/04rmv',
                  '/m/01f91_', '/m/01yrx', '/m/0cn6p', '/m/068zj', '/m/01lcw4', '/m/01knjb', '/m/09gtd', '/m/0h2r6', '/m/046dlr',
                  '/m/05bm6', '/m/03q69', '/m/043nyj', '/m/02cvgx', '/m/0gd36', '/m/04vv5k', '/m/07bgp', '/m/02zt3', '/m/03kt2w',
                  '/m/0898b', '/m/0dj6p', '/m/02z51p', '/m/01bfm9', '/m/07jdr', '/m/09d5_', '/m/04c0y', '/m/09kmb', '/m/015qff',
                  '/m/071p9', '/m/0bwd_0j']

# Parents Level
LEVEL_PARENTS = ['/m/07j7r', '/m/01g317', '/m/09j5n', '/m/0c9ph5', '/m/0k4j', '/m/0cgh4',
                   '/m/04bcr3', '/m/01prls', '/m/019jd', '/m/0138tl', '/m/07yv9',
                   '/m/015p6', '/m/04brg2', '/m/0271t', '/m/0c_jw', '/m/06msq', '/m/0270h',
                   '/m/02xwb', '/m/0ch_cf', '/m/0f4s2w', '/m/0jbk', '/m/04szw', '/m/0zvk5',
                   '/m/02dl1y', '/m/03hl4l9', '/m/03vt0', '/m/07mhn', '/m/018xm',
                   '/m/02rdsp', '/m/01mqdt', '/m/01rzcn', '/m/0h99cwc', '/m/02crq1',
                   '/m/03ssj5', '/m/020jm', '/m/01lrl', '/m/06nwz', '/m/083kb',
                   '/m/0hf58v5', '/m/019dx1', '/m/0k5j', '/m/0d_2m', '/m/0gd2v',
                   '/m/03xxp', '/m/0fbdv', '/m/02wv6h6', '/m/01x3z', '/m/0174n1',
                   '/m/0l515', '/m/06bt6', '/m/02pkr5', '/m/01dws', '/m/02w3_ws',
                   '/m/0dv77', '/m/0dv9c', '/m/07cx4', '/m/09dzg']

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
