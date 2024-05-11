import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')
from scipy import spatial
import operator
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from customtkinter import *
import tkinter as tk
from tkinter import ttk

movies = pd.read_csv('Datasets/tmdb_5000_movies.csv')
credits = pd.read_csv('Datasets/tmdb_5000_credits.csv')

movies['genres'] = movies['genres'].apply(json.loads)
for index, i in zip(movies.index, movies['genres']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies.loc[index, 'genres']

movies['keywords'] = movies['keywords'].apply(json.loads)
for index, i in zip(movies.index, movies['keywords']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies.loc[index, 'keywords'] = str(list1)

movies['production_companies'] = movies['production_companies'].apply(json.loads)
for index, i in zip(movies.index, movies['production_companies']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies.loc[index, 'production_companies'] = str(list1)

credits['cast'] = credits['cast'].apply(json.loads)
for index, i in zip(credits.index, credits['cast']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    credits.loc[index, 'cast'] = str(list1)

credits['crew'] = credits['crew'].apply(json.loads)
def director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
credits['crew'] = credits['crew'].apply(director)
credits.rename(columns={'crew':'director'}, inplace=True)

movies = movies.merge(credits, left_on = 'id', right_on = 'movie_id', how = 'left')
movies = movies[['id', 'original_title', 'genres', 'cast', 'vote_average', 'director', 'keywords']]

movies['genres'] = movies['genres'].apply(lambda x: sorted(x, key=lambda genre: genre['name']))
movies['genres'] = movies['genres'].apply(lambda x: ','.join(genre['name'] for genre in x))
movies['genres'] = movies['genres'].str.split(',')

genreList = []
for index, row in movies.iterrows():
    genres = row["genres"]
    
    for genre in genres:
        if genre not in genreList:
            genreList.append(genre)

def binary(genre_list):
    binaryList = []
    
    for genre in genreList:
        if genre in genre_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList

movies['genres_bin'] = movies['genres'].apply(lambda x: binary(x))

movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
movies['cast'] = movies['cast'].str.split(',')

for i,j in zip(movies['cast'],movies.index):
    list2 = []
    list2 = i[:4]
    movies.loc[j,'cast'] = str(list2)
movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['cast'] = movies['cast'].str.split(',')
for i,j in zip(movies['cast'],movies.index):
    list2 = []
    list2 = i
    list2.sort()
    movies.loc[j,'cast'] = str(list2)
movies['cast']=movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')

castList = []
for index, row in movies.iterrows():
    cast = row["cast"]
    
    for i in cast:
        if i not in castList:
            castList.append(i)


def binary(cast_list):
    binaryList = []
    
    for genre in castList:
        if genre in cast_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList

movies['cast_bin'] = movies['cast'].apply(lambda x: binary(x))

def xstr(s):
    if s is None:
        return ''
    return str(s)
movies['director'] = movies['director'].apply(xstr)


directorList=[]
for i in movies['director']:
    if i not in directorList:
        directorList.append(i)

def binary(director_list):
    binaryList = []  
    for direct in directorList:
        if direct in director_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    return binaryList

movies['director_bin'] = movies['director'].apply(lambda x: binary(x))
movies.head()

movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
movies['keywords'] = movies['keywords'].str.split(',')
for i,j in zip(movies['keywords'],movies.index):
    list2 = []
    list2 = i
    movies.loc[j,'keywords'] = str(list2)
movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['keywords'] = movies['keywords'].str.split(',')
for i,j in zip(movies['keywords'],movies.index):
    list2 = []
    list2 = i
    list2.sort()
    movies.loc[j,'keywords'] = str(list2)
movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['keywords'] = movies['keywords'].str.split(',')

words_list = []
for index, row in movies.iterrows():
    genres = row["keywords"]

    for genre in genres:
        if genre not in words_list:
            words_list.append(genre)


def binary(words):
    binaryList = []
    for genre in words_list:
        if genre in words:
            binaryList.append(1)
        else:
            binaryList.append(0)
    return binaryList


movies['words_bin'] = movies['keywords'].apply(lambda x: binary(x))
movies = movies[(movies['vote_average']!=0)] 
movies = movies[movies['director']!='']


def Similarity(movieId1, movieId2):
    a = movies.iloc[movieId1]
    b = movies.iloc[movieId2]
    
    genresA = a['genres_bin']
    genresB = b['genres_bin']
    
    genreDistance = spatial.distance.cosine(genresA, genresB)
    
    scoreA = a['cast_bin']
    scoreB = b['cast_bin']
    scoreDistance = spatial.distance.cosine(scoreA, scoreB)
    
    directA = a['director_bin']
    directB = b['director_bin']
    directDistance = spatial.distance.cosine(directA, directB)
    
    wordsA = a['words_bin']
    wordsB = b['words_bin']
    wordsDistance = spatial.distance.cosine(directA, directB)
    return genreDistance + directDistance + scoreDistance + wordsDistance

new_id = list(range(0, movies.shape[0]))
movies['new_id'] = new_id
movies = movies[['original_title', 'genres', 'vote_average', 'genres_bin', 'cast_bin' , 'new_id' , 'director' , 'director_bin' , 'words_bin']]

def predict_score(name):
    loading_screen = show_loading_screen(app)
    root = CTk()
    root.geometry("800x600")

    frame = CTkFrame(master=root)
    frame.pack(side="top", expand=True, padx=(10, 50), pady=30)

    new_movie = movies[movies['original_title'].str.contains(name)].iloc[0].to_frame().T
    print('Selected Movie : ', new_movie.original_title.values[0])

    def getNeighbors(baseMovie, K):
        distances = []

        for index, movie in movies.iterrows():
            if movie['new_id'] != baseMovie['new_id'].values[0]:
                dist = Similarity(baseMovie['new_id'].values[0], movie['new_id'])
                distances.append((movie['new_id'], dist))

        distances.sort(key=operator.itemgetter(1))
        neighbors = []

        for x in range(K):
            neighbors.append(distances[x])
        return neighbors

    K = 10
    avgRating = 0
    neighbors = getNeighbors(new_movie, K)

    recommended_movies_label = CTkLabel(master=frame, font=("Arial", 18), text="Recommended Movies : ")
    recommended_movies_label.pack(anchor="w", pady=30, padx=50)

    tree = ttk.Treeview(frame, columns=("Genre", "Score"))

    tree.column("#0", minwidth=0, width=300, stretch=NO)
    tree.heading("#0", text="Name")
    tree.column("#1", stretch=NO, minwidth=0, width=300)
    tree.heading("#1", text="Genre")
    tree.column("#2", anchor="center", stretch=NO, minwidth=0, width=100)
    tree.heading("#2", text="Score")

    for neighbor in neighbors:
        movie_info = movies.iloc[neighbor[0]]
        name = movie_info['original_title']
        genre = ", ".join(movie_info['genres'])
        score = str(movie_info['vote_average'])

        tree.insert("", "end", text=name, values=(genre, score))

        avgRating += movie_info['vote_average']

    tree.pack(fill="both", expand=True, padx=50, pady=20)

    def plot_predictions():
        K = 10
        avgRating = 0
        neighbors = getNeighbors(new_movie, K)

        x = []
        y = []

        for i, neighbor in enumerate(neighbors, 1):
            movie_info = movies.iloc[neighbor[0]]
            x.append(i)
            y.append(movie_info['vote_average'])
            avgRating += movie_info['vote_average']

        avg_rating_btn.configure(text="Average Rating: {:.2f}".format(avgRating / K))

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker='o', linestyle='-')
        plt.title('Predicted Ratings for Nearest Neighbors')
        plt.xlabel('Neighbor')
        plt.ylabel('Rating')
        plt.grid(True)
        plt.show()

    avg_rating_btn = CTkButton(master=frame, text="Average Rating: {:.2f}".format(avgRating / K), command=plot_predictions)
    avg_rating_btn.pack(pady=30, padx=30)

    hide_loading_screen(loading_screen)

    root.mainloop()

def get_value():
    movie_name = entry.get()
    predict_score(movie_name)

app = CTk()
app.geometry("500x420")

tabview=CTkTabview(master=app,height=300,width=400)
tabview.pack(padx=20, pady=40)

tabview.add("Input")
tabview.add("Graph")

def mode():
    if appearnce.get() == "on":
        set_appearance_mode("light")
        set_default_color_theme("dark-blue")
    else:
        set_appearance_mode("system")

def get_graph():
    graph_window = CTk()
    graph_window.geometry("800x600")

    fig, ax = plt.subplots(figsize=(8, 6))

    list1 = []
    for i in movies['genres']:
        list1.extend(i)
    top_genres = pd.Series(list1).value_counts()[:10].sort_values(ascending=True)

    ax = top_genres.plot.barh(width=0.9, color=sns.color_palette('hls', 10))
    for i, v in enumerate(top_genres.values):
        ax.text(.8, i, v, fontsize=12, color='white', weight='bold')

    ax.set_title('Top Genres')

    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    def close_graph_window():
        plt.close(fig)
        canvas.get_tk_widget().destroy()
        graph_window.destroy()

    graph_window.protocol("WM_DELETE_WINDOW", close_graph_window)
    graph_window.mainloop()

class LoadingScreen(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Loading...")
        self.geometry("300x100")
        self.progress = ttk.Progressbar(self, orient="horizontal", length=200, mode="indeterminate")
        self.progress.pack(pady=20)
        self.center()

    def center(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.master.winfo_width() // 2) - (width // 2)
        y = (self.master.winfo_height() // 2) - (height // 2)
        self.geometry(f"+{x}+{y}")

def show_loading_screen(master):
    loading_screen = LoadingScreen(master)
    loading_screen.grab_set() 
    return loading_screen

def hide_loading_screen(loading_screen):
    loading_screen.grab_release()
    loading_screen.destroy()

label = CTkLabel(master=tabview.tab("Input"), text="Enter a movie Name", font=("Arial", 24))
entry = CTkEntry(master=tabview.tab("Input"), placeholder_text="Type name",height=35,width=250)
button = CTkButton(master=tabview.tab("Input"), text="Submit", command=get_value, corner_radius=8,width=100)

label.place(relx=0.22, rely=0.2)
entry.place(relx=0.18, rely=0.4)
button.place(relx=0.37, rely=0.6)

get_graph_label = CTkLabel(master=tabview.tab("Graph"), text="Get Genre Graph", font=("Arial", 24))
graph_btn = CTkButton(master=tabview.tab("Graph"), text="Get Graph", command=get_graph, corner_radius=8,width=100)

get_graph_label.place(relx=0.28, rely=0.2)
graph_btn.place(relx=0.37, rely=0.45)

appearnce = CTkSwitch(master=app, text="Light", command=mode, onvalue="on", offvalue="off")
appearnce.place(relx=0.85, rely=0.93, anchor="center")

app.mainloop()