# 🎬 Smart Movie Recommender

An intelligent movie recommendation system that suggests personalized movies using machine learning techniques such as popularity-based filtering, content-based filtering, collaborative filtering, and a hybrid approach.

---

## 🚀 Live Demo
https://smart-movie-recommender-vishal.streamlit.app/

---

## 📸 Project Preview

![App Screenshot](screenshot.png)

---

## 📌 Features

- 🔥 **Popularity-Based Filtering** – Recommends trending movies based on ratings and popularity  
- 🎯 **Content-Based Filtering** – Recommends movies based on similarity in genres  
- 🤝 **Collaborative Filtering** – Uses user ratings to find similar movies  
- 🚀 **Hybrid Recommendation System** – Combines multiple techniques for better accuracy  
- 🖼️ **Movie Posters Integration** – Fetches posters using TMDB API  
- 🎨 **Modern UI** – Netflix-style responsive interface using Streamlit  
- ⚡ **Fast Performance** – Optimized using caching  

---

## 🧠 How It Works

1. **Popularity-Based Filtering**

*Recommends movies based on overall ratings and number of user interactions  
*Helps suggest trending and widely liked movies  

2. **Data Preprocessing**

   * Cleaned and merged movie & rating datasets
   * Handled missing values

3. **Content-Based Filtering**

   * Used TF-IDF vectorization on genres
   * Applied cosine similarity to find similar movies

4. **Collaborative Filtering**

   * Built user-item matrix
   * Calculated similarity between movies using ratings

5. **Hybrid Model**

   * Combined:

     * Content similarity
     * Collaborative similarity
     * Popularity score

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Frontend/UI:** Streamlit
* **API:** TMDB API (for posters)
* **Version Control:** Git & GitHub

---

## 📂 Project Structure

```
movie-recommendation-system/
│
├── app.py
├── movies.csv
├── ratings.csv
├── screenshot.png
├── README.md
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/iamvishalyadav2005/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your TMDB API key

```python
API_KEY = "YOUR_API_KEY"
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## 📊 Dataset

* MovieLens Dataset
* Contains movie metadata and user ratings
* Used for building recommendation models

---

## 🎯 Future Improvements

* 🔐 User authentication system
* ❤️ Watchlist feature
* 🎥 Movie trailer integration
* 🌐 Deployment with public access
* 🧠 Personalized recommendations per user

---

## 🧠 Learning Outcomes

* Implemented real-world recommendation systems
* Understood ML concepts like TF-IDF and cosine similarity
* Built an interactive web app using Streamlit
* Integrated external APIs for enhanced UI

---

## 👨‍💻 Author

**Vishal Yadav**

* B.Tech CSE (AI & ML)
* Passionate about AI, ML, and Software Development

