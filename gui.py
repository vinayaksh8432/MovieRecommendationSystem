"""
Improved GUI for the Movie Recommendation System.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import pandas as pd
from typing import Optional, List
import threading


class LoadingDialog:
    """Simple loading dialog with progress indication."""
    
    def __init__(self, parent, title="Loading..."):
        self.parent = parent
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("300x120")
        self.dialog.resizable(False, False)
        
        # Center the dialog
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Create loading content
        label = ctk.CTkLabel(self.dialog, text="Processing your request...", font=("Arial", 14))
        label.pack(pady=20)
        
        self.progress = ctk.CTkProgressBar(self.dialog, width=250)
        self.progress.pack(pady=10)
        self.progress.set(0)
        
        # Start indeterminate progress
        self.start_progress()
    
    def start_progress(self):
        """Start the progress animation."""
        self.progress.start()
    
    def close(self):
        """Close the loading dialog."""
        try:
            self.progress.stop()
        except:
            pass
        try:
            self.dialog.grab_release()
        except:
            pass
        try:
            self.dialog.destroy()
        except:
            pass


class MovieRecommendationGUI:
    """Main GUI class for the Movie Recommendation System."""
    
    def __init__(self, recommendation_engine):
        """
        Initialize the GUI.
        
        Args:
            recommendation_engine: Instance of MovieRecommendationEngine
        """
        self.engine = recommendation_engine
        self.current_results = None
        self.is_closing = False
        
        # Set appearance
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Movie Recommendation System")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Set up proper window closure handling
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame, 
            text="🎬 Movie Recommendation System", 
            font=("Arial", 28, "bold")
        )
        title_label.pack(pady=(20, 30))
        
        # Create notebook for tabs
        self.notebook = ctk.CTkTabview(main_frame, height=500)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Add tabs
        self.notebook.add("Recommendations")
        self.notebook.add("Statistics")
        self.notebook.add("Top Movies")
        
        self.setup_recommendations_tab()
        self.setup_statistics_tab()
        self.setup_top_movies_tab()
        
    
    def setup_recommendations_tab(self):
        """Set up the recommendations tab."""
        tab = self.notebook.tab("Recommendations")
        
        # Search frame
        search_frame = ctk.CTkFrame(tab)
        search_frame.pack(fill="x", padx=20, pady=20)
        
        search_label = ctk.CTkLabel(
            search_frame, 
            text="Enter Movie Title:", 
            font=("Arial", 16, "bold")
        )
        search_label.pack(pady=(15, 5))
        
        # Search entry with autocomplete
        self.search_var = tk.StringVar()
        self.search_entry = ctk.CTkEntry(
            search_frame,
            textvariable=self.search_var,
            placeholder_text="Type movie name...",
            width=400,
            height=35,
            font=("Arial", 14)
        )
        self.search_entry.pack(pady=10)
        self.search_entry.bind('<KeyRelease>', self.on_search_change)
        
        # Suggestions listbox
        self.suggestions_frame = ctk.CTkFrame(search_frame)
        self.suggestions_frame.pack(pady=5)
        self.suggestions_frame.pack_forget()  # Initially hidden
        
        # Search button
        search_btn = ctk.CTkButton(
            search_frame,
            text="Get Recommendations",
            command=self.get_recommendations,
            width=200,
            height=40,
            font=("Arial", 14, "bold")
        )
        search_btn.pack(pady=15)
        
        # Results frame
        self.results_frame = ctk.CTkScrollableFrame(tab)
        self.results_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Initial message
        initial_msg = ctk.CTkLabel(
            self.results_frame,
            text="Enter a movie title above to get personalized recommendations!",
            font=("Arial", 16),
            text_color="gray"
        )
        initial_msg.pack(pady=50)
    
    def setup_statistics_tab(self):
        """Set up the statistics tab."""
        tab = self.notebook.tab("Statistics")
        
        # Controls frame
        controls_frame = ctk.CTkFrame(tab)
        controls_frame.pack(fill="x", padx=20, pady=20)
        
        stats_label = ctk.CTkLabel(
            controls_frame,
            text="Dataset Statistics",
            font=("Arial", 20, "bold")
        )
        stats_label.pack(pady=15)
        
        # Buttons for different charts
        btn_frame = ctk.CTkFrame(controls_frame)
        btn_frame.pack(pady=10)
        
        genre_btn = ctk.CTkButton(
            btn_frame,
            text="Genre Distribution",
            command=self.show_genre_chart,
            width=150
        )
        genre_btn.pack(side="left", padx=10)
        
        rating_btn = ctk.CTkButton(
            btn_frame,
            text="Rating Distribution",
            command=self.show_rating_chart,
            width=150
        )
        rating_btn.pack(side="left", padx=10)
        
        # Chart frame
        self.chart_frame = ctk.CTkFrame(tab)
        self.chart_frame.pack(fill="both", expand=True, padx=20, pady=10)
    
    def setup_top_movies_tab(self):
        """Set up the top movies tab."""
        tab = self.notebook.tab("Top Movies")
        
        # Controls
        controls_frame = ctk.CTkFrame(tab)
        controls_frame.pack(fill="x", padx=20, pady=20)
        
        title_label = ctk.CTkLabel(
            controls_frame,
            text="Top Rated Movies",
            font=("Arial", 20, "bold")
        )
        title_label.pack(pady=15)
        
        # Number of movies selector
        selector_frame = ctk.CTkFrame(controls_frame)
        selector_frame.pack(pady=10)
        
        ctk.CTkLabel(selector_frame, text="Number of movies:").pack(side="left", padx=10)
        self.top_n_var = tk.StringVar(value="20")
        top_n_selector = ctk.CTkComboBox(
            selector_frame,
            values=["10", "20", "50", "100"],
            variable=self.top_n_var,
            command=self.update_top_movies
        )
        top_n_selector.pack(side="left", padx=10)
        
        # Top movies frame
        self.top_movies_frame = ctk.CTkScrollableFrame(tab)
        self.top_movies_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Load initial top movies
        self.update_top_movies()
    
    def on_search_change(self, event):
        """Handle search entry changes for autocomplete."""
        query = self.search_var.get()
        if len(query) >= 2:
            suggestions = self.engine.get_movie_suggestions(query, max_suggestions=5)
            self.show_suggestions(suggestions)
        else:
            self.hide_suggestions()
    
    def show_suggestions(self, suggestions: List[str]):
        """Show autocomplete suggestions."""
        # Clear previous suggestions
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()
        
        if suggestions:
            self.suggestions_frame.pack(pady=5)
            for suggestion in suggestions:
                btn = ctk.CTkButton(
                    self.suggestions_frame,
                    text=suggestion,
                    command=lambda s=suggestion: self.select_suggestion(s),
                    width=380,
                    height=30,
                    fg_color="transparent",
                    hover_color=("gray80", "gray20")
                )
                btn.pack(pady=2)
        else:
            self.hide_suggestions()
    
    def hide_suggestions(self):
        """Hide autocomplete suggestions."""
        self.suggestions_frame.pack_forget()
    
    def select_suggestion(self, suggestion: str):
        """Handle suggestion selection."""
        self.search_var.set(suggestion)
        self.hide_suggestions()
        self.search_entry.focus()
    
    def get_recommendations(self):
        """Get movie recommendations in a separate thread."""
        movie_title = self.search_var.get().strip()
        if not movie_title:
            messagebox.showwarning("Warning", "Please enter a movie title!")
            return
        
        # Show loading dialog
        loading = LoadingDialog(self.root, "Getting Recommendations...")
        
        def fetch_recommendations():
            try:
                results = self.engine.predict_rating(movie_title, k=10)
                if not self.is_closing:
                    self.root.after(0, lambda: self.display_results(results, loading))
            except Exception as e:
                if not self.is_closing:
                    self.root.after(0, lambda: self.handle_error(str(e), loading))
        
        # Run in separate thread to prevent GUI freezing
        thread = threading.Thread(target=fetch_recommendations)
        thread.daemon = True
        thread.start()
    
    def display_results(self, results: dict, loading_dialog):
        """Display recommendation results."""
        loading_dialog.close()
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        if 'error' in results:
            error_label = ctk.CTkLabel(
                self.results_frame,
                text=f"❌ {results['error']}",
                font=("Arial", 16),
                text_color="red"
            )
            error_label.pack(pady=20)
            
            if 'suggestions' in results and results['suggestions']:
                suggestion_label = ctk.CTkLabel(
                    self.results_frame,
                    text="Did you mean one of these?",
                    font=("Arial", 14, "bold")
                )
                suggestion_label.pack(pady=(20, 10))
                
                for suggestion in results['suggestions'][:5]:
                    btn = ctk.CTkButton(
                        self.results_frame,
                        text=suggestion,
                        command=lambda s=suggestion: self.select_suggestion(s),
                        width=400,
                        fg_color="transparent",
                        border_width=1
                    )
                    btn.pack(pady=5)
            return
        
        self.current_results = results
        
        # Target movie info
        target_info = results['target_movie']
        target_frame = ctk.CTkFrame(self.results_frame)
        target_frame.pack(fill="x", pady=20)
        
        ctk.CTkLabel(
            target_frame,
            text=f"🎬 {target_info['title']}",
            font=("Arial", 20, "bold")
        ).pack(pady=10)
        
        info_text = f"Director: {target_info['director']} | Rating: {target_info['actual_rating']:.1f}/10\\nGenres: {target_info['genres']}"
        ctk.CTkLabel(target_frame, text=info_text, font=("Arial", 12)).pack(pady=5)
        
        # Prediction info
        pred_frame = ctk.CTkFrame(self.results_frame)
        pred_frame.pack(fill="x", pady=10)
        
        pred_text = f"Predicted Rating: {results['predicted_rating']:.1f}/10 | Accuracy: ±{results['prediction_accuracy']:.1f}"
        ctk.CTkLabel(pred_frame, text=pred_text, font=("Arial", 14, "bold")).pack(pady=10)
        
        # Recommendations
        rec_label = ctk.CTkLabel(
            self.results_frame,
            text="🌟 Recommended Movies:",
            font=("Arial", 18, "bold")
        )
        rec_label.pack(pady=(20, 10))
        
        # Create treeview for recommendations
        columns = ("Title", "Director", "Genres", "Rating", "Similarity")
        tree_frame = ctk.CTkFrame(self.results_frame)
        tree_frame.pack(fill="both", expand=True, pady=10)
        
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=10)
        
        # Configure columns
        tree.heading("Title", text="Title")
        tree.heading("Director", text="Director")
        tree.heading("Genres", text="Genres")
        tree.heading("Rating", text="Rating")
        tree.heading("Similarity", text="Similarity %")
        
        tree.column("Title", width=200)
        tree.column("Director", width=150)
        tree.column("Genres", width=200)
        tree.column("Rating", width=80, anchor="center")
        tree.column("Similarity", width=100, anchor="center")
        
        # Add recommendations
        for rec in results['recommendations']:
            tree.insert("", "end", values=(
                rec['title'],
                rec['director'],
                rec['genres'][:50] + "..." if len(rec['genres']) > 50 else rec['genres'],
                f"{rec['rating']:.1f}",
                f"{rec['similarity_score']*100:.1f}%"
            ))
        
        tree.pack(fill="both", expand=True, padx=20, pady=20)
    
    def handle_error(self, error_msg: str, loading_dialog):
        """Handle errors during recommendation fetching."""
        loading_dialog.close()
        messagebox.showerror("Error", f"An error occurred: {error_msg}")
    
    def show_genre_chart(self):
        """Show genre distribution chart."""
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        try:
            stats = self.engine.get_genre_statistics()
            if not stats or 'most_common_genres' not in stats:
                ctk.CTkLabel(
                    self.chart_frame,
                    text="No genre data available",
                    font=("Arial", 16)
                ).pack(pady=50)
                return
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            genre_data = stats['most_common_genres']
            genres = list(genre_data.keys())[:10]
            counts = list(genre_data.values())[:10]
            
            bars = ax.barh(genres, counts, color=sns.color_palette('viridis', len(genres)))
            ax.set_title('Top 10 Genres Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Number of Movies')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{int(width)}', ha='left', va='center')
            
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            ctk.CTkLabel(
                self.chart_frame,
                text=f"Error creating chart: {str(e)}",
                font=("Arial", 16),
                text_color="red"
            ).pack(pady=50)
    
    def show_rating_chart(self):
        """Show rating distribution chart."""
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        try:
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ratings = self.engine.movies_df['vote_average']
            ax.hist(ratings, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
            ax.set_title('Movie Rating Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Rating')
            ax.set_ylabel('Number of Movies')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_rating = ratings.mean()
            ax.axvline(mean_rating, color='red', linestyle='--', 
                      label=f'Mean: {mean_rating:.1f}')
            ax.legend()
            
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            ctk.CTkLabel(
                self.chart_frame,
                text=f"Error creating chart: {str(e)}",
                font=("Arial", 16),
                text_color="red"
            ).pack(pady=50)
    
    def update_top_movies(self, value=None):
        """Update the top movies display."""
        # Clear previous content
        for widget in self.top_movies_frame.winfo_children():
            widget.destroy()
        
        try:
            n = int(self.top_n_var.get())
            top_movies = self.engine.get_top_rated_movies(n)
            
            for i, (_, movie) in enumerate(top_movies.iterrows(), 1):
                movie_frame = ctk.CTkFrame(self.top_movies_frame)
                movie_frame.pack(fill="x", pady=5, padx=10)
                
                # Rank and title
                title_text = f"{i}. {movie['original_title']}"
                title_label = ctk.CTkLabel(
                    movie_frame,
                    text=title_text,
                    font=("Arial", 16, "bold")
                )
                title_label.pack(anchor="w", padx=15, pady=(10, 5))
                
                # Details
                details_text = f"Rating: {movie['vote_average']:.1f}/10 | Director: {movie['director']}"
                if isinstance(movie['genres'], list):
                    genres_text = ', '.join(movie['genres'])
                else:
                    genres_text = str(movie['genres'])
                details_text += f" | Genres: {genres_text}"
                
                details_label = ctk.CTkLabel(
                    movie_frame,
                    text=details_text,
                    font=("Arial", 12),
                    text_color="gray"
                )
                details_label.pack(anchor="w", padx=15, pady=(0, 10))
                
        except Exception as e:
            error_label = ctk.CTkLabel(
                self.top_movies_frame,
                text=f"Error loading top movies: {str(e)}",
                font=("Arial", 16),
                text_color="red"
            )
            error_label.pack(pady=50)
    
    def on_closing(self):
        """Handle window closing properly to prevent after() errors."""
        self.is_closing = True
        try:
            # Cancel any pending after() calls
            self.root.after_cancel("all")
        except:
            pass
        
        try:
            # Destroy the window
            self.root.quit()
            self.root.destroy()
        except:
            pass
    
    def toggle_theme(self):
        """Toggle between light and dark themes."""
        current_mode = ctk.get_appearance_mode()
        new_mode = "light" if current_mode == "dark" else "dark"
        ctk.set_appearance_mode(new_mode)
    
    def run(self):
        """Start the GUI application."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Application interrupted by user")
        except Exception as e:
            print(f"Application error: {e}")
            messagebox.showerror("Critical Error", f"Application error: {e}")
        finally:
            self.is_closing = True
            try:
                self.root.quit()
            except:
                pass