import os
import pickle
from collections import defaultdict

# Update this path to your own dataset location
directory = r"C:\Users\hp\Downloads\Telegram Desktop\genres_original\genres_original"

# Create genre mappings
results = defaultdict(int)
i = 1
for folder in os.listdir(directory):
    results[i] = folder
    i += 1

# Save genre mappings in a pickle file
with open('genre_mappings.pickle', 'wb') as genre_file:
    pickle.dump(results, genre_file)

print("Genre mappings have been saved to genre_mappings.pickle.")
