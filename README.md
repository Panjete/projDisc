 ## Shape Annotations 
 
0. sleeve length: 0 sleeveless, 1 short-sleeve, 2 medium-sleeve, 3 long-sleeve, 4 not long-sleeve, 5 NA
1. lower clothing length: 0 three-point, 1 medium short, 2 three-quarter, 3 long, 4 NA
2. socks: 0 no, 1 socks, 2 leggings, 3 NA
3. hat: 0 no, 1 yes, 2 NA
4. glasses: 0 no, 1 eyeglasses, 2 sunglasses, 3 have a glasses in hand or clothes, 4 NA
5. neckwear: 0 no, 1 yes, 2 NA
6. wrist wearing: 0 no, 1 yes, 2 NA
7. ring: 0 no, 1 yes, 2 NA
8. waist accessories: 0 no, 1 belt, 2 have a clothing, 3 hidden, 4 NA
9. neckline: 0 V-shape, 1 square, 2 round, 3 standing, 4 lapel, 5 suspenders, 6 NA
10. outer clothing a cardigan?: 0 yes, 1 no, 2 NA
11. upper clothing covering navel: 0 no, 1 yes, 2 NA

Shape Format : <img_name> <shape_0> <shape_1> ... <shape_11>

## Fabric Annotations

0 denim, 1 cotton, 2 leather, 3 furry, 4 knitted, 5 chiffon, 6 other, 7 NA

Fabric format : <img_name> <upper_fabric> <lower_fabric> <outer_fabric>

## Color Annotations

0 floral, 1 graphic, 2 striped, 3 pure color, 4 lattice, 5 other, 6 color block, 7 NA

Color Annotations : <img_name> <upper_color> <lower_color> <outer_color>


Note: 'NA' means the relevant part is not visible.

TODO:

> Maybe, ask the user for help?
> Explore weighing strategies
> Explore distance functions
> Try adding color if possible
> Means and stds while learning images for classifier ? 
> Data set does not have colour description

- query_200.py , train_200.py , models_200, captions_200.json - for the Fashion200 Dataset
- training.py, query.py, classifier.py, visual.py, captions.json - For Inception, DeepFashion
- training_vgg.py, query_vgg.py, classifier_vgg.py, visual_vgg.py , captions.json- For VGG19, DeepFashion
- text_embedding.ipynb - for learning text embeddings
-