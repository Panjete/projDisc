

TODO:

> Maybe, ask the user for help?
> Explore weighing strategies
> Explore distance functions
> Try adding color if possible
> Means and stds while learning images for classifier ? 
> Data set does not have colour description


## File Structure

### Getting Started

1. Install `pytorch`, `torchvision`, `pickle`, `annoy` and `tkinter` libraries.
2. Ensure paths are correct , vis-a-vis the `model_path` and `pwd` variable in all the files of the relevant subdirectory.
3. When executing for the first time, the models will be saved in the `models/` directory. Later, they can be loaded from the corresponding `.pth` files.
4. Run `python user.py` for receiving a prompt to upload query image and text, and voila!


### Global files 

- `user.py` - Top Level User-Interface, prompts user for query image and text, and prresents retrieved results
- `metric.py` - for computing similarity index to analyse the quality of the retrieved results
- `text_embedding.ipynb` - contains functionality to generate the text embeddings from a text file

### `fashion200/` 

Contains files for the `Fashion 200k` dataset, where:- 

- `captions_200.json` - contains the per-image captions, generated from label files
- `train_200.py` - houses functionality to generate the learnt-database, compressing and storing visual and textual features
- `query_200.py` - houses functionality to read and query the database for an image-text pair
- `models_200.py` - implementation of the visual encoder used for this dataset
- `preprocess_words.py` - for stopword elimination and text -> vector conversion.

### `inception_df/` 

Contains files for the `DeepFashion` dataset, trained on the `inception_v3` model. Here, :- 

- `captions.json` - contains the per-image captions, generated from label files
- `train.py` - houses functionality to generate the learnt-database, compressing and storing visual and textual features
- `query.py` - houses functionality to read and query the database for an image-text pair
- `classifier.py` - implementation of the model for generating one-hot and (embeddings for )features inferred from the image.
- `visual.py` - implementation of the inception framework for extracting visual features of the image.
- `preprocess_words.py` - for stopword elimination and text -> vector conversion.

### `inception_df_color/` 

Contains files for the `DeepFashion` dataset, trained on the `inception_v3` model. The caption file used here has colors appended to it, this had to be done manually, since the dataset does not provision for colors. A separate multi-label classifier model for color extraction was trained for 11 colors, and the generated colors per image were appended to the caption file. The other files are similar to `inception_df/`. This lets the user also query for terms like `[blue]` or `[brown]` as well!

### `vgg_df/`

Contains files for the `DeepFashion` dataset, trained on the `VGG19` model. This was done to analyse perfomance difference of the retrieval system by contrasting with the `inception_v3` model, and to experiment with the vector-size of the visual features used in the dataset.

### Shape Annotations of the Deepfashion Framework
 
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

- Fabric and Color Annotations

Fabric : 0 denim, 1 cotton, 2 leather, 3 furry, 4 knitted, 5 chiffon, 6 other, 7 NA
Color : 0 floral, 1 graphic, 2 striped, 3 pure color, 4 lattice, 5 other, 6 color block, 7 NA

Fabric format : <img_name> <upper_fabric> <lower_fabric> <outer_fabric>
Color Annotations : <img_name> <upper_color> <lower_color> <outer_color>

Note: 'NA' means the relevant part is not visible.