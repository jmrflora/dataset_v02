import requests
import time
import json
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except ImportError:
    from requests.packages.urllib3.util.retry import Retry

class StackOverflowCollector:
    """
    Collects Stack Overflow posts about error handling paradigms.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.stackexchange.com/2.3"
        self.api_key = api_key
        self.rate_limit_remaining = 300
        
        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def _make_request(self, endpoint: str, params: dict, timeout: int = 30) -> Optional[dict]:
        """
        Args:
            endpoint: API endpoint
            params: Query parameters
            timeout: Request timeout in seconds
            
        """
        if self.api_key:
            params["key"] = self.api_key
            
        try:
            response = self.session.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                timeout=timeout
            )
            
            # rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f" Rate limited. Waiting {retry_after} seconds ")
                time.sleep(retry_after)
                return None 
            
            response.raise_for_status()
            data = response.json()
            
            # quota tracking
            self.rate_limit_remaining = data.get("quota_remaining", self.rate_limit_remaining)
            
            return data
            
        except requests.exceptions.Timeout:
            print(f" Request timeout after {timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            print(f" Request error: {e}")
            return None
    
    def search_questions(self, 
                    query: str, 
                    min_score: int = 2,    
                    max_pages: int = 15): 
        """
        Args:
            query: Search query string
            min_score: Minimum question score
            max_pages: Maximum pages to retrieve per query
        """
        
        all_questions = []
        page = 1
        
        # Use reasonable date range (2015-2024)
        from_timestamp = int(datetime(2019, 1, 1).timestamp())
        to_timestamp = int(datetime(2025, 8, 30).timestamp())
        
        print(f"\n  Searching: '{query}'")
        
        while page <= max_pages:
            print(f"    Page {page}/{max_pages}...", end=" ", flush=True)
            
            params = {
                "order": "desc",
                "sort": "votes",
                "q": query,
                "site": "stackoverflow",
                "filter": "!9_bDDxJY5",  # Includes body but not answers
                "fromdate": from_timestamp,
                "todate": to_timestamp,
                "min": min_score,
                "page": page,
                "pagesize": 50,
                "accepted": "True", 
                "answers": 1 
            }
            
            data = self._make_request("search/advanced", params)
            
            if data is None:
                print("Failed, retrying...")
                time.sleep(3)
                data = self._make_request("search/advanced", params)
                if data is None:
                    print("Failed again, skipping")
                    break
            
            items = data.get("items", [])
            all_questions.extend(items)
            
            print(f"Got {len(items)} questions (Total: {len(all_questions)}, Quota: {self.rate_limit_remaining})")
            
            # Check if there are more pages
            has_more = data.get("has_more", False)
            if not has_more or len(items) == 0:
                break
            
            # Check quota
            if self.rate_limit_remaining < 50:
                break
            
            page += 1
            time.sleep(2) 
        
        print(f"  ✓ Collected {len(all_questions)} questions from '{query}'")
        return all_questions
    
    def get_top_answer_score(self, question_id: int) -> int:
        """
        Args:
            question_id: Stack Overflow question ID
        """
        params = {
            "order": "desc",
            "sort": "votes",
            "site": "stackoverflow",
            "filter": "!9Z(-wwYGT",  # Minimal filter, just score
            "pagesize": 1 
        }
        
        data = self._make_request(f"questions/{question_id}/answers", params)
        
        if data and data.get("items"):
            return data["items"][0].get("score", 0)
        
        return 0
    
    def collect_error_handling_dataset(self, 
                                      queries_per_type: int = 2,
                                      target_posts: int = 150) -> pd.DataFrame:
        """
        Args:
            queries_per_type: Number of queries to run per paradigm
            target_posts: Target number of total posts to collect
        """
        
        exception_queries = [
    "exception handling",
    "try catch",
    "throw exception",
    "exception best practice",
    "when to throw exception",
    "exception vs error code",
    "nested try catch",
    "exception handling pattern",
    "error handling exception",
    "exception finally block"
    ]

        value_queries = [
    "Result type",
    "Either monad",
    "Option type",
    "Maybe monad",
    "railway oriented",
    "Rust Result",
    "Swift Result",
    "Haskell Either",
    "Scala Either",
    "F# Result",
    "functional error",
    "algebraic error",
    "sum type error",
    "tagged union error",
    "monadic error"
    ]
        
        all_posts = []
        
        print("=" * 60)
        print("COLLECTING EXCEPTION-BASED POSTS")
        print("=" * 60)
        
        for query in exception_queries:
            if len(all_posts) >= target_posts:
                print(f"\n Target of {target_posts} posts reached, stopping collection")
                break
                
            questions = self.search_questions(query, min_score=5, max_pages=3)
            
            print(f"  Processing {len(questions)} questions...")
            processed = 0
            
            for idx, q in enumerate(questions, 1):
                if len(all_posts) >= target_posts:
                    break
                
                # Progress indicator 
                if idx % 10 == 0:
                    print(f"    Progress: {idx}/{len(questions)} ({processed} added)")
                
                # Check if question has enough engagement
                if q.get("answer_count", 0) < 1 or len(q.get("body", "")) < 200:
                    continue
                
                # For first 50 questions, verify answer quality
                if len(all_posts) < 50:
                    top_answer_score = self.get_top_answer_score(q["question_id"])
                    if top_answer_score < 3:
                        continue
                    time.sleep(1.5) 
                else:
                    top_answer_score = -1  # Not checked
                
                post_data = {
                    "question_id": q["question_id"],
                    "type": "exception_based",
                    "title": q.get("title", ""),
                    "body": q.get("body", ""),
                    "score": q.get("score", 0),
                    "view_count": q.get("view_count", 0),
                    "answer_count": q.get("answer_count", 0),
                    "creation_date": datetime.fromtimestamp(q.get("creation_date", 0)),
                    "tags": ",".join(q.get("tags", [])),
                    "link": q.get("link", ""),
                    "best_answer_score": top_answer_score if top_answer_score >= 0 else q.get("score", 0),
                    "query_source": query
                }
                all_posts.append(post_data)
                processed += 1
            
            print(f" Added {processed} posts from this query\n")
            time.sleep(3) 
        
        print("=" * 60)
        print("COLLECTING VALUE-BASED POSTS")
        print("=" * 60)
        
        for query in value_queries:
            if len(all_posts) >= target_posts:
                print(f"\nTarget of {target_posts} posts reached, stopping collection")
                break
                
            questions = self.search_questions(query, min_score=5, max_pages=3)
            
            print(f"  Processing {len(questions)} questions...")
            processed = 0
            
            for idx, q in enumerate(questions, 1):
                if len(all_posts) >= target_posts:
                    break
                
                if idx % 10 == 0:
                    print(f"    Progress: {idx}/{len(questions)} ({processed} added)")
                
                if q.get("answer_count", 0) < 1 or len(q.get("body", "")) < 200:
                    continue
                
                if len(all_posts) < 100: 
                    top_answer_score = self.get_top_answer_score(q["question_id"])
                    if top_answer_score < 3:
                        continue
                    time.sleep(1.5)
                else:
                    top_answer_score = -1
                
                post_data = {
                    "question_id": q["question_id"],
                    "type": "value_based",
                    "title": q.get("title", ""),
                    "body": q.get("body", ""),
                    "score": q.get("score", 0),
                    "view_count": q.get("view_count", 0),
                    "answer_count": q.get("answer_count", 0),
                    "creation_date": datetime.fromtimestamp(q.get("creation_date", 0)),
                    "tags": ",".join(q.get("tags", [])),
                    "link": q.get("link", ""),
                    "best_answer_score": top_answer_score if top_answer_score >= 0 else q.get("score", 0),
                    "query_source": query
                }
                all_posts.append(post_data)
                processed += 1
            
            print(f" Added {processed} posts from this query\n")
            time.sleep(3)
        
        # Create DataFrame and remove duplicates
        df = pd.DataFrame(all_posts)
        
        if len(df) == 0:
            print("\n No posts collected!")
            return df
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=['question_id'])
        
        if initial_count > len(df):
            print(f"  Removed {initial_count - len(df)} duplicate posts")
        
        # Final filtering
        df['body_length'] = df['body'].str.len()
        df = df[df['body_length'] >= 200]
        
        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print("=" * 60)
        print(f"Total posts collected: {len(df)}")
        print(f"Exception-based: {len(df[df['type'] == 'exception_based'])}")
        print(f"Value-based: {len(df[df['type'] == 'value_based'])}")
        print(f"API quota remaining: {self.rate_limit_remaining}")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "stackoverflow_error_handling.csv"):
        if len(df) == 0:
            print("No data to save")
            return
            
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\n Dataset saved to {filename}")
        
        # save question IDs for reproducibility
        id_df = df[['question_id', 'type', 'creation_date', 'link', 'score']]
        id_filename = f"question_ids_{filename}"
        id_df.to_csv(id_filename, index=False)
        print(f" Question IDs saved to {id_filename}")
        
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"Date range: {df['creation_date'].min()} to {df['creation_date'].max()}")
        print(f"Average score: {df['score'].mean():.1f}")
        print(f"Median score: {df['score'].median():.0f}")
        print(f"Average views: {df['view_count'].mean():.0f}")
        print(f"Average answers: {df['answer_count'].mean():.1f}")
        
        print("\nTop 10 tags:")
        all_tags = ','.join(df['tags']).split(',')
        from collections import Counter
        tag_counts = Counter(all_tags)
        for tag, count in tag_counts.most_common(10):
            if tag:  # Skip empty tags
                print(f"  {tag}: {count}")


if __name__ == "__main__":
    
    # I paste the api key directly here
    api_key = None
    if api_key:
        print(" Using provided API key")
    else:
        print(" No API key provided. Limited to 300 requests/day.")
    
    collector = StackOverflowCollector(api_key)
    
    df = collector.collect_error_handling_dataset(
    queries_per_type=10, 
    target_posts=1000   
)
    
    if len(df) > 0:
        collector.save_dataset(df)
    else:
        print("\nCollection failed. Check your API key and internet connection.")
        sys.exit(1)