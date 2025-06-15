import tkinter as tk
from tkinter import ttk, messagebox
from recommender import MusicRecommender
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class MusicRecommenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Recommender System")
        self.root.geometry("800x600")
        
        # Initialize recommender
        self.recommender = MusicRecommender("Data/tcc_ceds_music.csv")
        self.recommender.load_and_prepare_data()
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create and configure style
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, relief="flat", background="#2196F3")
        self.style.configure("TLabel", padding=6, font=('Helvetica', 10))
        self.style.configure("TEntry", padding=6)
        
        # Title
        title_label = ttk.Label(
            self.main_frame,
            text="Music Recommender System",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Search frame
        search_frame = ttk.Frame(self.main_frame)
        search_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Song search
        ttk.Label(search_frame, text="Enter a song name:").grid(row=0, column=0, padx=5)
        self.song_entry = ttk.Entry(search_frame, width=40)
        self.song_entry.grid(row=0, column=1, padx=5)
        
        # Search button
        search_button = ttk.Button(
            search_frame,
            text="Get Recommendations",
            command=self.get_recommendations
        )
        search_button.grid(row=0, column=2, padx=5)
        
        # Results frame
        results_frame = ttk.Frame(self.main_frame)
        results_frame.grid(row=2, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        
        # Results listbox
        self.results_listbox = tk.Listbox(
            results_frame,
            width=70,
            height=15,
            font=('Helvetica', 10),
            selectmode=tk.SINGLE
        )
        self.results_listbox.grid(row=0, column=0, padx=5, pady=5)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Bind Enter key to search
        self.song_entry.bind('<Return>', lambda e: self.get_recommendations())
        
        # Center the window
        self.center_window()
        
    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def get_recommendations(self):
        """Get and display recommendations for the entered song"""
        song_name = self.song_entry.get().strip()
        if not song_name:
            messagebox.showwarning("Warning", "Please enter a song name")
            return
        
        try:
            # Clear previous results
            self.results_listbox.delete(0, tk.END)
            
            # Get recommendations
            self.status_var.set("Searching for recommendations...")
            self.root.update()
            
            # Find the song in the dataset
            if song_name not in self.recommender.df["track_name"].values:
                messagebox.showinfo("Not Found", f"'{song_name}' not found in the dataset")
                self.status_var.set("Ready")
                return
            
            # Get recommendations
            idx = self.recommender.df[self.recommender.df["track_name"] == song_name].index[0]
            song_vector = self.recommender.feature_matrix[idx].reshape(1, -1)
            
            # Compute similarities
            similarities = cosine_similarity(song_vector, self.recommender.feature_matrix)[0]
            
            # Get top 10 similar songs
            similar_indices = similarities.argsort()[::-1][1:11]
            recommendations = self.recommender.song_titles.iloc[similar_indices]
            
            # Display recommendations
            self.results_listbox.insert(tk.END, f"Recommendations for '{song_name}':")
            self.results_listbox.insert(tk.END, "")
            
            for i, title in enumerate(recommendations, 1):
                self.results_listbox.insert(tk.END, f"{i}. {title}")
            
            self.status_var.set("Ready")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred")

def main():
    root = tk.Tk()
    app = MusicRecommenderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 