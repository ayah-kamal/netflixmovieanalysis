# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ![banner, Netflix Logo](https://torranceca.files.wordpress.com/2019/10/netflix.jpg)
# %% [markdown]
# # Netflix EDA and Visualization 
# ---
# ## Background
# > Netflix is a subscription-based streaming service that allows our members to watch TV shows and movies without commercials on an internet-connected device. [[Reference]](https://help.netflix.com/en/node/412#:~:text=Netflix%20is%20a%20subscription%2Dbased,on%20an%20internet%2Dconnected%20device.&text=If%20you're%20already%20a,visit%20Getting%20started%20with%20Netflix.)
# 
# In this analysis, the dataset used contains the information of all the movies and TV shows on Netflix. The dataset will be used to answer the research questions:
#  - What is Saudi Arabia's top genre?
#  - Which country produces the most content?
#  - What's the best month to release content?
# 
# ## About the Data
# The data represents the current catalog of Movies and TV shows on Netflix as of September 25, 2021. The dataset was sourced from Kaggle: [here](https://www.kaggle.com/shivamb/netflix-shows).
# The following is a brief description of each variable (there are a total of 12 columns):
# - `show_id`: Unique ID for every Movie / Tv Show
# - `type`: Identifier - A Movie or TV Show
# - `title`: Title of the Movie / Tv Show
# - `director`: Director of the Movie
# - `cast`: Actors involved in the movie / show
# - `country`: Country where the movie / show was produced
# - `date_added`: Date it was added on Netflix
# - `release_year`: Actual Release year of the move / show
# - `rating`: TV Rating of the movie / show
# - `duration`: Total Duration - in minutes or number of seasons
# - `listed_in`: Genre
# - `description`: The summary description
# 
#  
# %% [markdown]
# ## Importing required packages 

# %%
import pandas as pd
import numpy as np
import seaborn as sns
from plotnine import ggplot,geom_bar, aes
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import missingno as msno
from collections import Counter

# %% [markdown]
# ## Loading the dataset

# %%
netflix_df = pd.read_csv('netflix_titles.csv')
netflix_df.head()


# %%
netflix_df.shape

# %% [markdown]
# The dataset contains 12 columns that we can use for exploratory analysis. Just by observing the first 5 rows, we can see that there are NaN values in multiple columns. This takes us to the next step, cleaning the dataset. 
# %% [markdown]
# ## Cleaning the data

# %%
msno.matrix(netflix_df)
plt.show()

# %% [markdown]
# From the plot above, we see that the director column contains the most NaN values. With other columns such as cast and country showing null values. To get a better idea of just how much we will find the percentage of null values in each column and which columns contain null values.

# %%
print('\nColumns with missing value:') 
print(netflix_df.isnull().any())


# %%
(netflix_df.isnull().mean()*100).sort_values(ascending=False)[:6]

# %% [markdown]
# About 30% of the director column is null, followed by country and cast.
# Before we can begin any analysis of the data, we will have to first deal with these null values.
# 
# We cannot use imputation, which is a method for dealing with missing values by filling them either with their estimatied statistical "best guess" values (e.g.: mean, mode, median) or by using techniques such as KNN or tree-based. This is because it is better in the cases of director, country, and cast to have an unknown value than to have an incorrect value. So, instead, we will use use of the fillna function from Pandas to indicate that the information is missing. 

# %%
netflix_df.director.fillna("No Director", inplace=True)
netflix_df.cast.fillna("No Cast", inplace=True)
netflix_df.country.fillna("Country Unavailable", inplace=True)

# %% [markdown]
# Since the percentage of null values for date_added, rating, and duration are less than 1%, we will instead drop all the rows that contain NaN values for any of these columns. 

# %%
netflix_df.dropna(subset=["date_added", "rating", "duration"], inplace=True)


# %%
print('\nColumns with missing value:') 
print(netflix_df.isnull().any())


# %%
netflix_df.shape

# %% [markdown]
# Only 17 rows out of the original 8807 rows are dropped.
# %% [markdown]
# ---
# ## EDA and Visualization
# We will first perform some analysis that will help us better understand our dataset. 

# %%
plt.figure(figsize=(12,6))
plt.title("Percentage of Netflix Titles as either Movies or TV Shows")
plt.pie(netflix_df.type.value_counts(),explode=(0.01,0.01),labels=netflix_df.type.value_counts().index, colors=['#b1a7a6',"#a4161a"],autopct="%1.2f%%")
plt.show()

# %% [markdown]
# So, there are approximately 4000 movies and 2000 TV shows. 
# 
# We now want to plot which year the titles where added to Netflix, however since the date_added column is an object, we want to convert it into date-time format first. 

# %%
netflix_df.dtypes


# %%
netflix_df['date_added'] =  pd.to_datetime(netflix_df['date_added'])
netflix_df.head()


# %%
min(pd.DatetimeIndex(netflix_df['date_added']).year)


# %%
netflix_df.groupby([pd.DatetimeIndex(netflix_df['date_added']).year, 'type'])['type'].count().unstack(level=1).plot(kind='line', figsize=(15, 8), color =['#b1a7a6','#a4161a'], linewidth = 4)
plt.xlim([2008,2021])
plt.xticks(np.arange(2008, 2022, step=1))
plt.show()

# %% [markdown]
# From the plot we can see that most titles (both movies and TV show) are added in the year 2019. We can also note that movies have been added as early as 2008, while TV shows have only been added since 2013.
# %% [markdown]
# ### What is Saudi Arabia's top genre?
# 
# Notice that the genres are listed under the listed_in column, however, we want to extract each genre individually. So, we will first create a Panda Series that contains each individual genre under the listed_in colum and then observe the most popular genres.

# %%
genre = netflix_df['listed_in']
seperated_genre = ','.join(genre).replace(' ,',',').replace(', ',',').split(',')
genre_count = pd.Series(dict(Counter(seperated_genre))).sort_values(ascending=False)
genre_count


# %%
genre_top = genre_count[:20]
plt.figure(figsize=(20,12))
sns.barplot(genre_top, genre_top.index, palette="RdGy")
plt.show()

# %% [markdown]
# Now we will find the most popular genre produced by Saudi Arabia:

# %%
netflix_genre_country = pd.DataFrame([netflix_df['country'].apply(lambda x: x.split(',')[0]), netflix_df['listed_in']])
netflix_genre_country_t = netflix_genre_country.T
netflix_df_exploded = netflix_genre_country_t.set_index(['country']).apply(lambda x: x.str.split(',').explode()).reset_index()
country_count_df = netflix_df_exploded.value_counts().rename_axis().reset_index(name='counts')
country_count_df


# %%
sa_count = country_count_df.loc[country_count_df['country'] == 'Saudi Arabia'].reset_index(drop=True)
del sa_count['country']
sa_count


# %%
sa_count.plot.bar(x = 'listed_in',y = 'counts', color = '#a4161a')
plt.title("Most popular genre produced by Saudi Arabia on Netflix")
plt.xlabel("Genre")
plt.show()

# %% [markdown]
# From the plot, we can see that most content produced by Saudi Arabia is listed under *International Movies*. However, we can say that the most popular genre of movies or TV shows produced fall under the genre of *Comedy*
# %% [markdown]
# ### Which country produces the most content?
# 
# For our next question, we're going to look at which country produced the most content in general. 
# 

# %%
country_count=netflix_df['country'].value_counts().sort_values(ascending=False)
country_count=pd.DataFrame(country_count)
topcountries=country_count[0:14]
topcountries


# %%
topcountries.plot.bar(color = '#a4161a')
plt.title("Country with most content produced on Netflix")
plt.xlabel("Country")
plt.legend([]) 
plt.show()

# %% [markdown]
# So, we can see that the country with the most productions on Netflix is teh United States by quite a large gap with the second highest producer India. 
# %% [markdown]
# ### What's the best month to release content?
# 
# Besides seasonal movies and TV shows, there is usually a preference to release new content during months where there are not alot of other new releases. This helps decrease competition for the new release. 

# %%
netflix_date_df = pd.DataFrame()
netflix_date_df['content_added_month'] = netflix_df['date_added'].dt.month
netflix_date_df['type'] = netflix_df['type']
"""netflix_date_df['content_added_month'] = netflix_date_df['content_added_month'].map({
    1: 'January', 2: 'February', 3: 'March', 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"})"""
netflix_date_df


# %%
netflix_date_df.groupby(['content_added_month', 'type'])['type'].count().unstack(level=1).sort_values('content_added_month', ascending = True).plot(kind='bar', subplots=False, figsize=(15, 8), colormap="RdGy")
my_xticks = ["January", "February", "March", "April", "May", "June", "July", "August",  "September", "October", "November", "December"]
y_pos = np.arange(len(my_xticks))
plt.xticks(y_pos, my_xticks, rotation=45, horizontalalignment='right')
plt.show()

# %% [markdown]
# So from the plot above we can see that the best months to release a movie or TV show onto Netflix would be February and then May since these are the months that have the least amount of content released.
# %% [markdown]
# ## Conclusion
# We have been able to come to many informational inferences from our Netflix titles dataset, to summarize some of these inferences:
# 1. Most of the content on Netflix is Movies 
# 2. Movies have been on Netflix since 2008, but TV shows weren't added until 2013
# 3. Netflix saw much more content added after the years 2015 - 2016
# 4. Saudi Arabia produces mostly Comedies and their content falls mostly under the international movies category.
# 5. International movies is the most popular listing for movies, with the most popular genre on Netflix being *Dramas*
# 6. The United States produces the most content on Netflix followed by India and the United Kingdom.
# 7. The best month to release content is February. 
# %% [markdown]
# 

