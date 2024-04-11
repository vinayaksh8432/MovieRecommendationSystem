import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from difflib import get_close_matches
import random

width = 500
height = 200

width_2 = 500
height_2 = 300

def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

def recommend_movies():
    user_preferences = entry.get()
    if not user_preferences:
        return

    print("User Preferences: ", user_preferences)

    user_preferences_lower = user_preferences.lower()

    movies['title_lower'] = movies['title'].str.lower()

    matching_movies = movies[movies['title_lower'].str.contains(user_preferences_lower)]

    if matching_movies.empty:
        all_movie_titles = movies['title'].tolist()
        suggestions = get_close_matches(user_preferences, all_movie_titles, n=5, cutoff=0.6)
        not_found(suggestions)
        return

    matching_movies = matching_movies.drop('title_lower', axis=1)

    title_to_recommend = matching_movies.iloc[0]['title']

    idx = indices[title_to_recommend]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    recommended_movies_info = []

    for i, idx in enumerate(movie_indices):
        title = movies['title'].iloc[idx]
        score = sim_scores[i][1]
        recommended_movies_info.append((title, score * 100))

    recommendation_window = tk.Toplevel(root)
    recommendation_window.title(f"Movie : {title_to_recommend}")
    recommendation_window.geometry(f"{width_2}x{height_2}")
    center_window(recommendation_window, width_2, height_2)

    tree = ttk.Treeview(recommendation_window, columns=("Movie", "Score"), show="headings")
    tree.heading("Movie", text="Movies")
    tree.heading("Score", text="Scores")

    for movie, score in recommended_movies_info:
        tree.insert("", "end", values=(movie, f"{score:.2f}"))

    tree.pack(pady=20)

    back_button = ttk.Button(recommendation_window, text="Back", command=recommendation_window.destroy)
    back_button.pack()

def not_found(missing_movies):
    not_found_window = tk.Toplevel(root)
    not_found_label = ttk.Label(not_found_window, text=f"No matching movies found. \nSuggestions: {', '.join(missing_movies)}")
    not_found_window.geometry("400x120")
    center_window(not_found_window, 400, 120)
    not_found_label.pack(pady=20)

    back_button = ttk.Button(not_found_window, text="Back", command=not_found_window.destroy)
    back_button.pack()

def random():
    random_movie = movies.sample(n=1)
    
    random_movie_info = f"Title : {random_movie['title'].iloc[0]}\n\n" \
                        f"Overview :\n{random_movie['overview'].iloc[0]}\n\n" \
                        f"Scores : {random_movie['popularity'].iloc[0]:.2f}"

    random_movie_window = tk.Toplevel(root)
    random_movie_window.title(f"Random Movie : {random_movie['title'].iloc[0]}")
    random_movie_window.geometry(f"{width_2}x{height_2}")
    center_window(random_movie_window, width_2, height_2)
    
    text_widget = tk.Text(random_movie_window, wrap=tk.WORD)
    text_widget.insert(tk.END, random_movie_info)
    text_widget.pack(expand=True, fill=tk.BOTH)

credits = pd.read_csv('Datasets/tmdb_5000_credits.csv')
movies = pd.read_csv('Datasets/tmdb_5000_movies.csv')
credits.columns = ['id', 'title', 'cast', 'crew']
movies = movies.merge(credits, on='id')
movies = movies.drop('title_y', axis=1)
movies.rename(columns={'title_x': 'title'}, inplace=True)

C = movies['vote_average'].mean()
m = movies['vote_count'].quantile(0.9)
q_movies = movies.copy().loc[movies['vote_count'] >= m]

def wr(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (v + m) * C)

q_movies['score'] = q_movies.apply(wr, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)

movies.rename(columns={'title_x': 'title'}, inplace=True)

plot = TfidfVectorizer(stop_words='english')
movies['overview'] = movies['overview'].fillna('')
plot_matrix = plot.fit_transform(movies['overview'])

cosine_sim = linear_kernel(plot_matrix, plot_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

root = tk.Tk()
root.title("Recommend System")

label = ttk.Label(root, text="Enter a movie name:")
label.pack(pady=15)

entry = ttk.Entry(root, width=50)
entry.pack(pady=10)

recommend_button = ttk.Button(root, text="Get Recommendations", command=recommend_movies)
recommend_button.pack(pady=10)

exit_button = ttk.Button(root, text="Exit", command=root.destroy)
exit_button.pack(pady=10)

random_button = ttk.Button(root, text="Random", command=random)
random_button.pack(pady=10)

root.geometry(f"{width}x{height}")
center_window(root, width, height)

root.mainloop()