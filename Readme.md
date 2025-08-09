
# 🎥 Movie Recommendation System — Collaborative Filtering

## 📌 Overview

This repository implements a **movie recommendation system** using the **MovieLens dataset**.
It is presented via two interfaces:

- **Command Line Interface (CLI) script**: `movie.py`
- **Interactive Web App**: `app.py` (powered by Streamlit)

Both systems demonstrate popular collaborative filtering methods:

- **User-Based Collaborative Filtering**
- **Item-Based Collaborative Filtering**

Cosine similarity is used to measure closeness, and model accuracy is evaluated via **RMSE** and **MAE** metrics.

***

## 📊 Dataset — MovieLens

The dataset (`ratings.csv`) contains movie ratings on a **5-star scale** (0.5 to 5.0) by users.

**Structure**:

- `userId`: Unique ID for each user
- `movieId`: Unique ID for each movie
- `rating`: User's rating (float)
- `timestamp`: Time of rating (Unix format)

> Dataset source: *MovieLens (ml-latest-small)*
> Contains **100,836 ratings**, **9,742 movies**, **610 users**.

***

## ⚙️ Features

### 1. Data Preparation

- Load ratings into **Pandas DataFrame**
- Train–Test Split (**70–30%** default; adjustable in web app)
- Build **User-Item** and **Item-User** matrices
- Create **dummy matrices** to mask already-rated movies


### 2. User-Based Collaborative Filtering

- Compute user-to-user **cosine similarity**
- Predict rating using a weighted sum of similar users’ ratings
- Generate **Top-N recommendations** per user


### 3. Item-Based Collaborative Filtering

- Compute item-to-item **cosine similarity**
- Predict rating using weighted ratings on similar items
- Generate **Top-N recommendations** per user


### 4. Model Evaluation

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- Evaluation manually triggered in web app for current train/test split


### 5. **Streamlit Web Application Features**

- Interactive selection of user, recommender type, and number of recommendations
- Test/train split and random seed fully adjustable
- Inline display of top recommendations and predicted ratings
- Live computation and display of RMSE/MAE on test set
- Easy exploration of dataset samples

***

## 📦 Installation \& Requirements

### Dependencies

Both the CLI and web app require:

```
pandas
numpy
scikit-learn
matplotlib
streamlit   # for web application
```

Install all with:

```bash
pip install -r requirements.txt
```

or, for the app:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```


***

## ▶️ Usage

### Command Line Script (`movie.py`)

1. **Place** `ratings.csv` in the same folder.
2. **Run:**

```bash
python movie.py
```

3. **Outputs:**
    - Top 5 user-based and item-based recommendations for user `userId = 42`
    - RMSE and MAE results for both systems

### Web Application (`app.py` or `movie_recommender_streamlit.py`)

1. **Place** `ratings.csv` in the same folder.
2. **Install** requirements (`streamlit` etc).
3. **Run:**

```bash
streamlit run app.py
```

*(or use the actual filename, e.g. `streamlit run movie_recommender_streamlit.py`)*
4. **Features:**
    - Choose user and recommender system from sidebar
    - Adjust train/test split and random seed live
    - View top N recommendations for any user
    - Evaluate RMSE \& MAE with a button click
    - Explore sample ratings in-browser

***

## 📈 Example Output

#### CLI Script

```
Top 5 User-based recommendations for user 42:
movieId
1234    4.8
5678    4.6
...
Top 5 Item-based recommendations for user 42:
movieId
8910    4.7
1123    4.5
...
User-based CF Evaluation:
RMSE: 0.92
MAE: 0.75

Item-based CF Evaluation:
RMSE: 0.89
MAE: 0.73
```


#### Streamlit App (Sidebar \& Main Panel):

- Interactive selection for user, recommendation type, number of results
- Live display of recommendations table
- Button-triggered model evaluation (RMSE, MAE)
- Dataset samples for exploration

***
## 📚 Methodology

### User-Based Collaborative Filtering
![User CF](https://latex.codecogs.com/png.image?\dpi{150}\fn_phv\color{Black}\hat{R}_{u,i}=\frac{\sum_{v\in%20similar(u)}sim(u,v)\cdot%20R_{v,i}}{\sum_{v\in%20similar(u)}|sim(u,v)|})

### Item-Based Collaborative Filtering
![Item CF](https://latex.codecogs.com/png.image?\dpi{150}\fn_phv\color{Black}\hat{R}_{u,i}=\frac{\sum_{j\in%20similar(i)}sim(i,j)\cdot%20R_{u,j}}{\sum_{j\in%20similar(i)}|sim(i,j)|})

**Similarity: Cosine Similarity**  
![Cosine](https://latex.codecogs.com/png.image?\dpi{150}\fn_phv\color{Black}sim(A,B)=\frac{A\cdot%20B}{||A||\cdot||B||})

***

## 👤 Author \& Contact

**Shyam Sanjey S**
🔗 [LinkedIn](https://www.linkedin.com/in/shyamsanjey2004)
🔗 [GitHub](https://github.com/ShyamSanjeyS)
✉️ [shyamsanjey.s@gmail.com](mailto:shyamsanjey.s@gmail.com)

💡 *Suggestions, collaboration ideas, or feedback are always welcome!*

***

## 📝 Notes

- This is a simple collaborative filtering demo; for production, consider sparse matrix optimizations, regularization, cold-start handling, etc.
- The web application is ideal for exploring the recommender’s performance and how recommendations change interactively.

***
