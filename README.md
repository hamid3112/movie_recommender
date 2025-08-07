# 🎬 Movie Recommendation Chatbot with Streamlit

This project is a simple yet effective **Movie Recommendation System** built with `pandas`, `scikit-learn`, and `Streamlit`. It suggests 5 similar movies based on the genre, overview, and tagline of a user-input movie using **TF-IDF** and **cosine similarity**.

---

## 🚀 Features

- 📁 Uses real metadata from a movies dataset (`movies_metadata.csv`)
- 🧠 Extracts features from genres, plot summaries (overview), and taglines
- 🔍 Computes similarity between movies using TF-IDF vectorization
- 🤖 Provides a chatbot-like interface to enter a movie name and get recommendations
- 🛑 Includes basic error handling for missing files and memory issues

---

## 🧰 Requirements

Install the required Python packages:

```bash
pip install pandas scikit-learn streamlit
````

---

## 📁 Dataset

Make sure you have the file:

```
movies_metadata.csv
```

Place it in the following path (or change the `path` variable in the script):

```
C:\Users\SEYED\AI\project\movie\data_set\movies_metadata.csv
```

> You can download this file from [Kaggle - The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

---

## ▶️ How to Run

Launch the Streamlit app with the following command:

```bash
streamlit run your_script_name.py
```

Then open the web app in your browser and enter the title of a movie in English.

---

## 🧠 How It Works

1. **Data Cleaning**:

   * Keeps only essential columns: `title`, `genres`, `overview`, `tagline`
   * Fills missing values and standardizes text
   * Converts `genres` column from JSON to plain text

2. **Feature Engineering**:

   * Concatenates `genres`, `overview`, and `tagline` into a single text field
   * Uses **TF-IDF vectorization** to represent movies as vectors
   * Calculates **cosine similarity** matrix between all movies

3. **Recommendation**:

   * When the user inputs a movie title, the system:

     * Finds the index of the input movie
     * Retrieves top 5 most similar movies (excluding itself)
     * Displays them as recommendations

---

## 🧪 Sample

**Input:**

```
Toy Story
```

**Output:**

```
▪️ Toy Story 2  
▪️ A Bug's Life  
▪️ Monsters, Inc.  
▪️ Finding Nemo  
▪️ Shrek
```

---

## ❗ Error Handling

* 📂 If the dataset file is missing, the app will show a user-friendly error message.
* 🧠 If system memory is not sufficient, the app stops safely.

---

## 📌 Notes

* The model is limited to the top 8000 movies to avoid memory overload.
* Movie names must be entered in **English** (and match the dataset titles closely).
