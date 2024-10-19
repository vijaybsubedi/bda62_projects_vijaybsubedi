# Import necesary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
import requests

"""
STEP 1: WEBPAGE REQUEST
"""

# Fetching movies list from the IMDB website 
moviesurl = "https://www.imdb.com/search/title/?title_type=feature"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Request to fetch the data from the URL
pgrequest = requests.get(moviesurl, headers=headers) 
if pgrequest.status_code == 200:
    print(f"Webpage was successfully fetched. STATUS CODE: {pgrequest.status_code}")
else:
    print(f"Error in retrieving the webpage. STATUS CODE: {pgrequest.status_code}")


"""
STEP 2: PARSING THE HTML DATA CONTENT
"""
soup = BeautifulSoup(pgrequest.text, 'html.parser')
print(type(soup))

# scrapped movie names
movies = soup.find_all("li", class_="ipc-metadata-list-summary-item")
print(f"Total number of movies found: {len(movies)}")


"""
STEP 3: EXTRACT THE MOVIE DETAILS
"""
# Initialize the list
movies_data = []
# Iterating through each movie and getting relevant details
for movie in movies:
    title = movie.find('h3', class_="ipc-title__text").text.split('.')[1]   
    year = movie.find("div", class_="sc-732ea2d-5 kHnTQb dli-title-metadata").find_all('span')[0].text    
    duration = movie.find("div", class_="sc-732ea2d-5 kHnTQb dli-title-metadata").find_all('span')[1].text 
    film_rating = movie.find("div", class_="sc-732ea2d-5 kHnTQb dli-title-metadata").find_all('span')[2].text 
    star_rating = movie.find("span", class_="ipc-rating-star--rating").text if movie.find("span", class_="ipc-rating-star--rating") else np.nan
    voteCount = movie.find("span", class_="ipc-rating-star--voteCount").text if movie.find("span", class_="ipc-rating-star--voteCount") else np.nan
    metascore = movie.find("span", class_="sc-b0901df4-0 bXIOoL metacritic-score-box").text if movie.find("span", class_="sc-b0901df4-0 bXIOoL metacritic-score-box") else np.nan
    description= movie.find("div", class_="ipc-html-content-inner-div").get_text(strip=True) if movie.find("div", class_="ipc-html-content-inner-div") else np.nan
    
    movies_data.append({
        "Movie Title": title,
        "Release Year": year,
        "Movie Duration":duration,
        "MPA Rating":film_rating,
        "Audience Rating":star_rating,
        "Audience Votes":voteCount,
        "Metascore":metascore,
        "Movie Description":description
    })
#movies_data
# Convert the movie details data into pandas dataframe
df = pd.DataFrame(movies_data)

"""
STEP 4: DATA CLEANING
"""
# Count None values per row
# This should also give us missing values/ null values OR else could use df.isnull function
nan_counts = df.isna().sum(axis=1)
print(f"Total number of NaN values:\n {df.isna().sum()}") 
duplicates = df.duplicated(keep=False)
df.duplicated().sum()
print(f"There are {df.duplicated().sum()} duplicate entries.")

# Check for duplicates
duplicates_movie_title = df['Movie Title'].duplicated(keep=False)
duplicates_movie_description = df['Movie Description'].duplicated(keep=False)
print(f"There are {duplicates_movie_title.sum()} duplicates in Movie Title column, and {duplicates_movie_description.sum()} duplicates in Movie description column.")

# N/A or non-available metascores are assigned integer value 0
df['Metascore'] = df['Metascore'].map(lambda x: 0 if x is None else x)
# Metascore has 'Nonetype' attribute: Converting it into numeric integer value for future use
df['Metascore'] = pd.to_numeric(df['Metascore'], errors='coerce').fillna(0).astype(int)

# Removing the brackets from elements of Audience Votes column
df['Audience Votes'] = df['Audience Votes'].str.replace(r'[\[\]\(\)]', '', regex=True)

"""
STEP 5: DATA TRANSFORMATION
"""
# Converts xx'h'xx'm' time format to minutes only
def convert_duration_to_minutes(duration):
    if isinstance(duration, str):
        parts = duration.split()
        hours = int(parts[0][:-1])  # Remove 'h' and convert to int
        minutes = int(parts[1][:-1])  # Remove 'm' and convert to int
        return hours * 60 + minutes
    return None
df['Movie Duration'] = df['Movie Duration'].apply(convert_duration_to_minutes)
# Clean Audience Votes: Remove non-numeric characters (like 'K' for thousands) and convert to numeric
def convert_audience_votes(votes):
    if isinstance(votes, str):
        votes = votes.strip().upper()  # Remove spaces and standardize to uppercase
        if 'K' in votes:
            votes = votes.replace('K', '')
            return float(votes) * 1000  # Convert 'K' to thousands
        elif 'M' in votes:
            votes = votes.replace('M', '')
            return float(votes) * 1000000  # Convert 'M' to millions
        else:
            # Remove any other non-numeric characters and convert to numeric
            votes = votes.replace(',', '').replace(' ', '')  
    return pd.to_numeric(votes, errors='coerce')  # Handle other cases like plain numbers

# Apply this function to the 'Audience Votes' column
df['Audience Votes'] = df['Audience Votes'].apply(convert_audience_votes)

"""
STEP 6: DATA TOKENIZATION
"""
# Defining a custom tokenzier that removes duplicates 
def customtokenizer(text):
    # Lowercase the text
    text = text.lower()
    
    # Removing punctuation marks.
    punctuation = ['!', '.', ',', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}', "'", '"']
    # Loop through each punctuation character and replace it with an empty string
    for p in punctuation:
        text = text.replace(p, "")    
    tokens = text.split()

    # Remove duplicates while preserving order
    seenwords = set()
    tokens2 = [token for token in tokens if not (token in seenwords or seenwords.add(token))]

    return tokens2   
 
# Applying the custom tokenizer to the Movie Description column in df
df['Tokens'] = df['Movie Description'].apply(customtokenizer)

"""
STEP 7: DATA VISUALIZATION
"""
#Box plot
plt.figure(figsize = (15,17))

# Box plot for Movie Duration 
plt.subplot(3,2,1)
sns.boxplot(df['Movie Duration'], color = 'blue')
plt.title('Box Plot of Movie Duration')
plt.xlabel('Movie Duration')

# Box plot for Audience Rating
plt.subplot(3,2,2)
sns.boxplot(df['Audience Rating'], color = 'orange')
plt.title('Box Plot of Audience Rating')
plt.xlabel('Audience Rating')

# Scatter plot for Audience Rating and Movie Duration
plt.subplot(3, 2, 3)  
sns.scatterplot(x=df['Audience Rating'], y=df['Movie Duration'], color='purple')
plt.title('Movie Duration vs Audience Rating (Scatter Plot)')
plt.xlabel('Audience Rating')
plt.ylabel('Movie Duration')

# Scatter plot for Audience Rating and Audience Votes
plt.subplot(3, 2, 4)  
sns.scatterplot(x=df['Audience Rating'], y=df['Audience Votes'], color='blue')
plt.title('Audience Votes vs Audience Rating (Scatter Plot)')
plt.xlabel('Audience Rating')
plt.ylabel('Audience Votes')

# Scatter plot for Audience Rating and Metascore
plt.subplot(3, 2, 5)  
sns.scatterplot(x=df['Audience Rating'], y=df['Metascore'], color='red')
plt.title('Metascore vs Audience Rating (Scatter Plot)')
plt.xlabel('Audience Rating')
plt.ylabel('Metascore')

# Correlation Matrix Heatmap
df_reduced = df.filter(items=['Movie Duration', 'Audience Rating', 'Audience Votes', 'Metascore'])
correlation_matrix = df_reduced.corr()
plt.figure(figsize = (6,5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()

"""
STEP 8: SAVING FINAL DATA AS CSV FILE
"""

df.to_csv('IMDBmoviesfinal.csv', index=False) 
print("Final movies data saved in CSV format")