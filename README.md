# Multimodal Movie Genre Classification

**Multi-Modal Data Analysis – Project 2.2**

## 1. Project Overview

This project focuses on predicting movie genres using both plot summaries and poster images. Since one movie can belong to more than one genre, I treat this as a multi-label classification problem with 23 possible genres.

The main goal is to compare how well text and image data perform separately and whether combining the two modalities can improve classification results.

I implemented and compared five approaches:

- Text-only model
- Image-only model
- Early Fusion
- Equal-Weight Late Fusion
- Tuned Late Fusion

I also examined class imbalance, training behaviour, per-genre performance and some limitations of the models.

## 2. Dataset

I used the [Simplified MM-IMDb dataset](https://www.kaggle.com/datasets/javierurea/simplified-mm-imdb) from Kaggle.

The dataset contains 25,959 movie records. Each record includes a movie ID, a plot summary, a poster image and a multi-label genre vector.

The dataset is divided into three sets:

| Dataset | Number of movies |
|---|---:|
| Training | 15581 |
| Validation | 2595 |
| Test | 7783 |

Before splitting the data, I checked for missing values, class imbalance and duplicated plot summaries and poster images. Records connected through exact duplicates were kept in the same split to reduce data leakage.

## 3. Methods

### Text Features

I used the pretrained DistilBERT model (`distilbert/distilbert-base-uncased`) to extract features from plot summaries.

The text was tokenised with a maximum length of 256 tokens. I used the first-token representation from DistilBERT as a 768-dimensional feature vector for each movie.

### Image Features

I used the pretrained CLIP ViT-B/32 model (`openai/clip-vit-base-patch32`) to extract poster features.

Poster images were converted into RGB images and processed using CLIP's image processor, including resizing, centre cropping and normalisation. Each poster was represented by a 512-dimensional feature vector.

### Classification Models

I used multilayer perceptron (MLP) classifiers for genre prediction. Because this is a multi-label classification task, the models were trained using binary cross-entropy with logits.

For Early Fusion, I combined the text and image features before classification.

For Late Fusion, I combined the predictions from the separately trained Text-only and Image-only models. I tested equal weighting and also tuned the fusion weights for individual genres using the validation set.

The pretrained DistilBERT and CLIP encoders were kept frozen during feature extraction.

## 4. Evaluation

I used Micro F1 and Macro F1 to compare the models. I also examined precision, recall and F1 scores for individual genres.

Training and validation losses were compared to identify possible overfitting. I also looked at prediction errors and differences in performance between common and less frequent genres.

## 5. Results

The final results on the test set are shown below.

| Model | Micro F1 | Macro F1 |
|---|---:|---:|
| Text-only | 0.6160 | 0.5078 |
| Image-only | 0.3830 | 0.1874 |
| Early Fusion | 0.6044 | 0.5183 |
| Equal-Weight Late Fusion | 0.5122 | 0.3105 |
| Tuned Late Fusion | 0.6186 | 0.5105 |

Tuned Late Fusion achieved the highest Micro F1 (0.6186), while Early Fusion obtained the highest Macro F1 (0.5183).

The Text-only model performed much better than the Image-only model, suggesting that plot summaries provide more useful information for this dataset. However, combining text and images improved the results for some individual genres, especially Animation.

Overall, multimodal fusion showed some benefits, but the improvements over Text-only were relatively small.

## 6. Repository Files

The main implementation is provided in `Project 2.2.ipynb`.

The notebook contains the data inspection, preprocessing, feature extraction, model training, evaluation, result visualisation and discussion.

The extracted features and trained models can be saved locally during execution.

## 7. How to Run the Project

### Step 1: Download the dataset

Download the Simplified MM-IMDb dataset from Kaggle:

https://www.kaggle.com/datasets/javierurea/simplified-mm-imdb

Extract the downloaded files. The notebook requires the movie records and poster image arrays.

### Step 2: Prepare the Python environment

Use a Python environment with Jupyter Notebook and the required libraries installed.

The main libraries used are:

- NumPy
- pandas
- Matplotlib
- Pillow
- PyTorch
- Hugging Face Transformers
- scikit-learn

### Step 3: Open the notebook

Download this repository and open `Project 2.2.ipynb` using Jupyter Notebook or JupyterLab.

### Step 4: Set up the dataset paths

Place the dataset files in the notebook's working directory, or update the paths in the data-loading cells to match the location of the downloaded files.

The notebook reads the movie information from `data.npy` and the poster images from `images.npz`.

### Step 5: Run the notebook

Run the notebook cells in order, starting with data loading and preprocessing.

The main stages are:

1. Inspect the dataset and check data quality.
2. Create the training, validation and test splits.
3. Preprocess the plot summaries and poster images.
4. Extract and save DistilBERT and CLIP features.
5. Train the Text-only and Image-only classifiers.
6. Train and evaluate the Early Fusion model.
7. Evaluate Equal-Weight and Tuned Late Fusion.
8. Compare the results and analyse prediction errors.

The first execution may take longer because the pretrained models need to be downloaded and features must be extracted.

The notebook uses a fixed random seed for data splitting and model training. However, some results may still vary slightly depending on the Python environment, library versions and hardware.

## 8. Limitations

One limitation of the dataset is that some genres have far fewer examples than others. This may affect classification performance, especially for less frequent genres.

Plot summaries longer than 256 tokens were truncated, so some information may have been removed. The CLIP image processor also crops poster images, which can remove visual details.

Another limitation is that the models were evaluated on one dataset. Their performance on other movie datasets has not been tested.

## 9. Conclusion

This project helped me understand how text and image features can be used separately and combined for a multi-label classification task.

The results show that multimodal fusion can be useful for predicting some movie genres, but using both modalities does not always produce better overall performance. The choice of fusion method depends on the evaluation metric and how much useful information each modality provides.

## 10. References

CLIP. (2021). Huggingface.Co. https://huggingface.co/docs/transformers/model_doc/clip

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. ArXiv. https://arxiv.org/abs/1810.04805

Openai/clip-vit-base-patch32 · hugging face. (2021). Huggingface.Co. https://huggingface.co/openai/clip-vit-base-patch32

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. PMLR. Proceedings.Mlr.Press. https://proceedings.mlr.press/v139/radford21a.html

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. arXiv.Org. https://arxiv.org/abs/1910.01108

Sanh, V., Debut,L., Chaumond,J., & Wolf, T. (2024). Distilbert/distilbert-base-uncased · hugging face. Huggingface.Co. https://huggingface.co/distilbert/distilbert-base-uncased

Ureña, J. (2024). Simplified MM-IMDb. Kaggledatasets. https://www.kaggle.com/datasets/javierurea/simplified-mm-imdb

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., … Rush, A. (2020). Transformers: State-of-the-art natural language processing. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 38‑45. https://doi.org/10.18653/v1/2020.emnlp-demos.6
