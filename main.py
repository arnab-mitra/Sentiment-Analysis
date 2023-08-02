import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from google_play_scraper import reviews

nltk.download('vader_lexicon')

app_id = 'com.avira.passwordmanager'

# Scraping reviews
num_reviews = 100
reviews_list, _ = reviews(app_id, lang='en', count=num_reviews)

# Performing sentiment analysis using VADER
sid = SentimentIntensityAnalyzer()
sentiments = [sid.polarity_scores(review['content'])['compound'] for review in reviews_list]

# Categorizing sentiment
def categorize_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

sentiments_categorized = [categorize_sentiment(score) for score in sentiments]

# Creating a pandas DataFrame for storing the reviews and sentiment
data = {'reviews': [review['content'] for review in reviews_list], 'sentiment': sentiments_categorized}
df = pd.DataFrame(data)

csv_file = 'result.csv'
df.to_csv(csv_file, index=False)

print(f'Reviews and sentiment scores saved to {csv_file}')

# Pie chart
sentiment_counts = df['sentiment'].value_counts()

fig_pie = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, hole=0.5)])
colors = {'Positive': 'limegreen', 'Negative': 'tomato', 'Neutral': 'gold'}
fig_pie.update_traces(marker=dict(colors=[colors[sentiment] for sentiment in sentiment_counts.index]),
                       textinfo='label+percent',
                       textfont_size=14,
                       hoverinfo='percent+value')
fig_pie.update_layout(
    title_text='Sentiment Analysis of Avira Password Manager Reviews (Pie Chart)',
    showlegend=False,
    annotations=[dict(text='Sentiment', x=0.5, y=0.5, font_size=20, showarrow=False)]
)
fig_pie.show()

# Bar graph
plt.figure(figsize=(8, 6))
counts = df['sentiment'].value_counts()
sentiment_colors = {'Positive': 'limegreen', 'Negative': 'tomato', 'Neutral': 'gold'}
bar_colors = [sentiment_colors[sentiment] for sentiment in counts.index]
plt.bar(counts.index, counts.values, color=bar_colors, alpha=0.7, edgecolor='black')
plt.title('Sentiment Analysis of Avira Password Manager Reviews (Bar Graph)', fontsize=16)
plt.xlabel('Sentiment', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()
