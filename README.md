# Movie Recommendation System – Content-Based Filtering (TF-IDF + Cosine Similarity)

## Project Summary

In the fast-evolving **entertainment industry**, **personalized recommendations** have become essential for enhancing **user engagement** and ensuring **long-term retention**.
This project builds a **Content-Based Movie Recommendation System** using **TF-IDF vectorization** and **Cosine Similarity** to recommend movies similar to a user’s chosen title.

By analyzing **movie metadata** — including **cast**, **crew**, **genres**, **keywords**, and **overview** — the model identifies **semantic relationships** between movies and delivers **highly relevant recommendations**, similar to systems used by **Netflix** or **IMDb**.

## Dataset Overview

The project uses the **TMDB 5000 Movie Dataset**, which combines information from two major sources:

* **tmdb_5000_movies.csv** – contains movie-level details such as **genres**, **keywords**, and **overviews**
* **tmdb_5000_credits.csv** – provides detailed information about **cast** and **crew**

After merging and cleaning both datasets, each record represents a **unique movie**, described by its key attributes:

* **Movie Details**: Title, Overview, Keywords, Genres
* **People Information**: Top Cast Members, Director
* **Additional Features**: Movie ID, Crew roles

The merged dataset serves as the foundation for building a **content-based similarity model** that generates movie recommendations based on text analysis.

## Analysis Approach

The project follows a structured **data science workflow** designed for text-based similarity modeling:

* **Data Preprocessing**: Cleaned and merged the TMDB movies and credits datasets, removed **duplicates** and **missing values**, and selected key attributes such as **title**, **cast**, **crew**, **genres**, **keywords**, and **overview**.
* **Feature Engineering**: Extracted top 3 cast members and directors using safe parsing (`ast.literal_eval`), removed whitespace, and combined text data into a single **“tags”** column.
* **Text Processing**: Applied **tokenization**, **lemmatization**, and **stopword removal** using **NLTK**, ensuring consistent lowercase formatting and clean textual features.
* **Model Building**:

  * Converted processed text into numerical representations using **TF-IDF Vectorization**.
  * Calculated **Cosine Similarity** between all movie pairs to determine closeness based on metadata.
  * Implemented a `recommend()` function to fetch the **Top N most similar movies** for a given title.

## Model Performance

Although content-based models do not use traditional ML metrics (like accuracy or F1-score), qualitative evaluation demonstrates **highly relevant** and **contextually accurate** recommendations.

### Example Recommendations (Actual Output)

| Input Movie         | Top 5 Recommended Movies                                                                                              |
| ------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **The Dark Knight** | The Dark Knight Rises<br>Batman Returns<br>Batman Begins<br>Batman: The Dark Knight Returns, Part 2<br>Batman Forever |
| **Iron Man**        | Iron Man 2<br>Iron Man 3<br>Avengers: Age of Ultron<br>The Avengers<br>Captain America: Civil War                     |

**Insights from Performance**:

* The system accurately captures **contextual relationships** across movie universes (e.g., Marvel, DC).
* Recommendations are based on **cast**, **genre**, and **keywords**, leading to highly **relevant outputs**.
* Demonstrates the effectiveness of **TF-IDF + Cosine Similarity** in capturing **semantic similarity** among movie descriptions.

## Key Outcomes

* Built a **content-based recommendation system** capable of generating **personalized movie suggestions**.
* Processed and analyzed **5000+ movie entries** with detailed metadata.
* Achieved **high-quality and accurate recommendations** using pure **text-based features**.
* Demonstrated the practical use of **NLP and feature engineering** for similarity-driven recommendation tasks.

## Tools & Technologies

* **Python**: Core programming language for data analysis and modeling
* **Pandas, NumPy**: Data cleaning, preprocessing, and feature engineering
* **NLTK**: Natural language text preprocessing (tokenization, lemmatization, stopword removal)
* **Scikit-learn**: TF-IDF vectorization and Cosine Similarity computation

## Conclusion

This project demonstrates how **content-based filtering** can be used to create an **interpretable**, **scalable**, and **unsupervised** recommendation system.
By combining **movie metadata** with **NLP-based text analysis**, the system effectively identifies movies sharing **similar genres, casts, and plots**, delivering an engaging **personalized viewing experience**.

The approach can serve as a foundation for more advanced systems by integrating **collaborative filtering** or **user feedback data** in the future.
