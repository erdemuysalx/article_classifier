# Challenge
Train a binary classifier on given 7494 academic article titles and/or abstracts to predict if the text belongs to `Material Science` or `Chemistry` category and evalute the best performing model's F-1, accurracy and AUC scores.

### Data
`wos2class.json` dataset is available in `data` folder. This dataset contains 7494 JSON objects with the following schema:
```
Title: String with maximum 279 characters.
Abstract: String with maximum 5289 characters.
Label: Categorical string. Either "Chemistry" or "Material Science".
```
