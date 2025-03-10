import praw 
import datetime
import os 
import pandas as pd
from tqdm import tqdm

user_agent = "Scraper 1.0 by/u/Srishtiiaggarwall99"
reddit = praw.Reddit(
    client_id = "6Q_O3l4A4AOs_7E8tOuIiA",
    client_secret = "sDpxkEQRafcWTUBhjR_5u-NkPiFuug",
    user_agent = user_agent
)

os.makedirs("Data", exist_ok=True)

subreddits = ["investing", "wallstreetbets", "stocks", "StockMarket", "finance"]
queries = ["market crash", "investors opinion on market crash", "investors behaviour on market crash"]

articles = []
for sub in tqdm(subreddits):
    subreddit = reddit.subreddit(sub)
    for query in queries:
        posts = subreddit.search(query=query, limit=5000)
        for post in posts:
            articles.append([
                post.created_utc,
                post.title,
                post.selftext,
                post.num_comments,
                post.score,
            ])

df = pd.DataFrame(articles, columns=["Date Posted", "Title", "Text", "Comments", "Upvotes"])
df["Date Posted"] = pd.to_numeric(df["Date Posted"], errors="coerce")
df = df.dropna(subset=["Date Posted"])
df["Date Posted"] = pd.to_datetime(df["Date Posted"], unit="s")
df.drop_duplicates(inplace=True)
df.dropna(axis=0, how="any", inplace=True)
print(df.shape)
df.to_csv("Data/reddit_data.csv", index=False)