
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk_resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'vader_lexicon', 'averaged_perceptron_tagger']
for resource in nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ErrorHandlingAnalyzer:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.lemmatizer = WordNetLemmatizer()
        self.sia = SentimentIntensityAnalyzer()
        
        # Domain-specific stopwords to KEEP 
        self.keep_words = {'error', 'exception', 'throw', 'catch', 'try', 
                          'result', 'return', 'type', 'handle', 'handling'}
        
        # Create custom stopwords list
        self.stop_words = set(stopwords.words('english')) - self.keep_words
        
        # for classification
        self.exception_keywords = {
            'try', 'catch', 'throw', 'finally', 'except', 'raise',
            'exception', 'throwing', 'catching', 'trycatch'
        }
        
        self.value_keywords = {
            'result', 'either', 'option', 'maybe', 'monad', 'monadic',
            'railway', 'algebraic', 'sum type', 'tagged union', 'variant'
        }
        
    def preprocess_text(self, text: str) -> str:
        """
        Args:
            text: Raw HTML text from post
        """
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        
        # Extract and remove code blocks
        for code in soup.find_all(['code', 'pre']):
            code.decompose()
        
        text = soup.get_text()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text.lower()
    
    def tokenize_and_lemmatize(self, text: str) -> list:
        """
        Tokenize and lemmatize text.
        Args:
            text: Preprocessed text
        """
        tokens = word_tokenize(text)
        
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return lemmatized
    
    def classify_post(self, text: str, current_type: str) -> str:
        """
        Args:
            text: Preprocessed text
            current_type: Initial classification from query
        """
        text_lower = text.lower()
        
        exception_count = sum(1 for kw in self.exception_keywords if kw in text_lower)
        value_count = sum(1 for kw in self.value_keywords if kw in text_lower)
        
        # If both paradigms heavily discussed, mark as comparative
        if exception_count >= 3 and value_count >= 3:
            return 'comparative'
        
        # Otherwise use keyword dominance
        if value_count > exception_count * 1.5:
            return 'value_based'
        elif exception_count > value_count * 1.5:
            return 'exception_based'
        
        # Default
        return current_type
    
    def sentiment_analysis(self, text: str) -> dict:
        """
        Args:
            text: Text to analyze
        """
        scores = self.sia.polarity_scores(text)
        
        # Benefit-cost analysis
        benefit_words = ['elegant', 'clean', 'safe', 'maintainable', 'clear', 
                        'explicit', 'composable', 'type-safe', 'robust']
        cost_words = ['verbose', 'complex', 'boilerplate', 'steep', 'difficult',
                     'overhead', 'nested', 'unpredictable', 'hidden', 'fragile']
        
        text_lower = text.lower()
        benefit_count = sum(text_lower.count(word) for word in benefit_words)
        cost_count = sum(text_lower.count(word) for word in cost_words)
        
        benefit_cost_ratio = (benefit_count - cost_count) / max(benefit_count + cost_count, 1)
        
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'benefit_cost_ratio': benefit_cost_ratio,
            'benefit_mentions': benefit_count,
            'cost_mentions': cost_count
        }
    
    def extract_complexity_indicators(self, text: str) -> dict:
        """
        Args:
            text: Preprocessed text
        """
        text_lower = text.lower()
        
        indicators = {
            'nested': text_lower.count('nested') + text_lower.count('nesting'),
            'unpredictable': text_lower.count('unpredictable') + text_lower.count('unpredictability'),
            'hidden': text_lower.count('hidden') + text_lower.count('implicit'),
            'maintainability': text_lower.count('maintain') + text_lower.count('maintenance'),
            'type_safety': text_lower.count('type safe') + text_lower.count('type-safe'),
            'verbosity': text_lower.count('verbose') + text_lower.count('verbosity')
        }
        
        return indicators
    
    def process_dataset(self) -> pd.DataFrame:
        print("Processing dataset...")
        
        # Combine title and body
        self.df['full_text'] = self.df['title'] + ' ' + self.df['body']
        
        self.df['cleaned_text'] = self.df['full_text'].apply(self.preprocess_text)
        
        self.df['tokens'] = self.df['cleaned_text'].apply(self.tokenize_and_lemmatize)
        self.df['token_count'] = self.df['tokens'].apply(len)
        
        self.df['refined_type'] = self.df.apply(
            lambda row: self.classify_post(row['cleaned_text'], row['type']), 
            axis=1
        )
        
        # Sentiment analysis
        sentiment_results = self.df['cleaned_text'].apply(self.sentiment_analysis)
        sentiment_df = pd.DataFrame(list(sentiment_results))
        self.df = pd.concat([self.df, sentiment_df], axis=1)
        
        # Extract complexity indicators
        complexity_results = self.df['cleaned_text'].apply(self.extract_complexity_indicators)
        complexity_df = pd.DataFrame(list(complexity_results))
        self.df = pd.concat([self.df, complexity_df], axis=1)
        
        print("Processing complete!")
        return self.df
    
    def topic_modeling(self, post_type: str, num_topics: int = 8) -> tuple:
        """
        Args:
            post_type: 'exception_based' or 'value_based'
            num_topics: Number of topics to extract
        """
        print(f"\nPerforming topic modeling for {post_type} posts...")
        
        # Filter posts
        subset = self.df[self.df['refined_type'] == post_type]
        
        if len(subset) == 0:
            print(f"No posts found for type: {post_type}")
            return None, None, None, None
        
        documents = subset['tokens'].tolist()
        
        dictionary = corpora.Dictionary(documents)
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=50,
            alpha=0.01,
            per_word_topics=True
        )
        
        top_topics = []
        for idx in range(num_topics):
            topic_words = lda_model.show_topic(idx, topn=10)
            top_topics.append({
                'topic_id': idx,
                'words': [word for word, _ in topic_words],
                'weights': [weight for _, weight in topic_words]
            })
            print(f"  Topic {idx}: {', '.join([w for w, _ in topic_words[:5]])}")
        
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=documents,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        print(f"  Coherence Score: {coherence_score:.4f}")
        
        return lda_model, corpus, dictionary, top_topics
    
    def calculate_term_frequencies(self, post_type: str, top_n: int = 30) -> pd.DataFrame:
        """
        Args:
            post_type: Type of posts to analyze
            top_n: Number of top terms to return
        """
        subset = self.df[self.df['refined_type'] == post_type]
        
        # Flatten all tokens
        all_tokens = [token for tokens in subset['tokens'] for token in tokens]
        
        term_freq = Counter(all_tokens)
        
        # Calculate per-post frequency
        num_posts = len(subset)
        freq_df = pd.DataFrame([
            {'term': term, 'count': count, 'per_post': count / num_posts}
            for term, count in term_freq.most_common(top_n)
        ])
        
        return freq_df
    
    def temporal_analysis(self) -> pd.DataFrame:
        # Create quarter column
        self.df['quarter'] = pd.to_datetime(self.df['creation_date']).dt.to_period('Q')
        
        # Group by quarter and type
        temporal = self.df.groupby(['quarter', 'refined_type']).agg({
            'compound': 'mean',
            'benefit_cost_ratio': 'mean',
            'question_id': 'count'
        }).reset_index()
        
        temporal.columns = ['quarter', 'type', 'avg_sentiment', 'avg_benefit_cost', 'post_count']
        
        return temporal
    
    def generate_summary_statistics(self) -> dict:
        stats = {}
        
        stats['total_posts'] = len(self.df)
        stats['exception_based'] = len(self.df[self.df['refined_type'] == 'exception_based'])
        stats['value_based'] = len(self.df[self.df['refined_type'] == 'value_based'])
        stats['comparative'] = len(self.df[self.df['refined_type'] == 'comparative'])
        
        # Sentiment by type
        for ptype in ['exception_based', 'value_based']:
            subset = self.df[self.df['refined_type'] == ptype]
            stats[f'{ptype}_sentiment'] = subset['compound'].mean()
            stats[f'{ptype}_benefit_cost'] = subset['benefit_cost_ratio'].mean()
        
        # Complexity indicators
        for ptype in ['exception_based', 'value_based']:
            subset = self.df[self.df['refined_type'] == ptype]
            stats[f'{ptype}_nested_per_post'] = subset['nested'].sum() / len(subset)
            stats[f'{ptype}_unpredictable_per_post'] = subset['unpredictable'].sum() / len(subset)
            stats[f'{ptype}_maintainability_mentions_pct'] = (subset['maintainability'] > 0).sum() / len(subset) * 100
        
        return stats
    
    def save_results(self, output_prefix: str = "analysis_results"):
        """
        Args:
            output_prefix: Prefix for output files
        """
        # Save
        self.df.to_csv(f"{output_prefix}_processed.csv", index=False)
        print(f"Saved processed dataset to {output_prefix}_processed.csv")
        
        # Save summary 
        stats = self.generate_summary_statistics()
        with open(f"{output_prefix}_statistics.txt", 'w') as f:
            f.write("=== Summary Statistics ===\n\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        print(f"Saved statistics to {output_prefix}_statistics.txt")
        
        # Save term frequencies
        for ptype in ['exception_based', 'value_based']:
            freq_df = self.calculate_term_frequencies(ptype)
            freq_df.to_csv(f"{output_prefix}_{ptype}_terms.csv", index=False)
            print(f"Saved term frequencies to {output_prefix}_{ptype}_terms.csv")


if __name__ == "__main__":
    df = pd.read_csv("stackoverflow_error_handling.csv")
    
    analyzer = ErrorHandlingAnalyzer(df)
    
    processed_df = analyzer.process_dataset()
    
    # Generate summary statistics
    stats = analyzer.generate_summary_statistics()
    print("\n=== Summary Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Perform topic modeling
    exception_topics = analyzer.topic_modeling('exception_based', num_topics=8)
    value_topics = analyzer.topic_modeling('value_based', num_topics=8)
    
    temporal_df = analyzer.temporal_analysis()
    print("\n=== Temporal Trends ===")
    print(temporal_df.head(10))
    
    analyzer.save_results()
    
    print("\n=== Complete ===")
    print("All results saved to disk.")