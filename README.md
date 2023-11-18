# Multimodal Search with 

In the task of near similar image search, features from Deep Neural Network are often used to compare images and measure similarity. Our analysis explores the vast data to compute k-nearest neighbors using both image and text (both derived fron the image, and provided by the dataset/user). We reduce the problem of searching over multiple modes into a single space by drawing apt correlations between the modes. Algorithmic details can be found in [1] and `report.pdf.`

## Getting Started

1. Install `pytorch`, `torchvision`, `pickle`, `annoy` and `tkinter` libraries.
2. Ensure paths are correct , vis-a-vis the `model_path` and `pwd` variable in all the files of the relevant subdirectory.
3. When executing for the first time, the models will be saved in the `models/` directory. Later, they can be loaded from the corresponding `.pth` files.
4. Update the execution environment in `user.py` and run `python user.py` for receiving a prompt to upload query image and text, and voila!

## File Structure 

### Global files 

- `user.py` - Top Level User-Interface, prompts user for query image and text, and prresents retrieved results
- `metric.py` - for computing similarity index to analyse the quality of the retrieved results
- `text_embedding.ipynb` - contains functionality to generate the text embeddings from a text file


### `inception_df/` 

Contains files for the `DeepFashion` dataset, trained on the `inception_v3` model. `w = 2000` set as default. Here, :- 

- `captions.json` - contains the per-image captions, generated from label files
- `train.py` - houses functionality to generate the learnt-database, compressing and storing visual and textual features
- `query.py` - houses functionality to read and query the database for an image-text pair
- `classifier.py` - implementation of the model for generating one-hot and (embeddings for )features inferred from the image.
- `visual.py` - implementation of the inception framework for extracting visual features of the image.
- `preprocess_words.py` - for stopword elimination and text -> vector conversion.

### `inception_df_color/` 

Contains files for the `DeepFashion` dataset, trained on the `inception_v3` model. The caption file used here has colors appended to it, this had to be done manually, since the dataset does not provision for colors. A separate multi-label classifier model for color extraction was trained for 11 colors, and the generated colors per image were appended to the caption file. Files used for this have been bundled in the `color/` subdirectory. `w = 200` set by default. The other files are similar to `inception_df/`. This lets the user also query for terms like `[blue]` or `[brown]` as well! 

### `vgg_df/`

Contains files for the `DeepFashion` dataset, trained on the `VGG19` model. This was done to analyse perfomance difference of the retrieval system by contrasting with the `inception_v3` model, and to experiment with the vector-size of the visual features used in the dataset. `w = 500` set by default.

### `fashion200/` 

Contains files for the `Fashion 200k` dataset, (`w = 200` as default) where:- 

- `captions_200.json` - contains the per-image captions, generated from label files
- `train_200.py` - houses functionality to generate the learnt-database, compressing and storing visual and textual features
- `query_200.py` - houses functionality to read and query the database for an image-text pair
- `models_200.py` - implementation of the visual encoder used for this dataset
- `preprocess_words.py` - for stopword elimination and text -> vector conversion.

## References 

[1] Jonghwa Yim, Junghun James Kim, Daekyu Shin ,  [“One-Shot Item Search with Multimodal Data”](https://arxiv.org/abs/1811.10969) 

[2] [The DeepFashion - Multimodal Dataset](https://github.com/yumingj/DeepFashion-MultiModal) 

[3] [The Fashion200k Dataset](https://github.com/xthan/fashion-200k) 

[4] [Spotify's Annoy](https://github.com/spotify/annoy) - Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk

## Credits

A team project by [Atharv Dabli](https://github.com/atharvadabli) and [Gurarmaan S. Panjeta](https://github.com/Panjete) .

