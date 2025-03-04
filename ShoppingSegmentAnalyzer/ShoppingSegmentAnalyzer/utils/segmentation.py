import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def perform_customer_segmentation(df):
    """
    Perform customer segmentation analysis using K-means clustering
    """
    # Aggregate data at user level
    user_features = df.groupby('user_id').agg({
        'event_type': 'count',
        'purchased': 'sum',
        'session_duration': 'mean',
        'total_sessions': 'first',
        'time_between_events': 'mean'
    }).reset_index()
    
    # Prepare features for clustering
    features = ['event_type', 'purchased', 'session_duration', 'total_sessions']
    X = user_features[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    user_features['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Create segment descriptions
    segment_stats = calculate_segment_stats(user_features)
    
    return create_segment_summary(segment_stats)

def calculate_segment_stats(user_features):
    """
    Calculate statistics for each segment
    """
    return user_features.groupby('cluster').agg({
        'event_type': ['mean', 'count'],
        'purchased': 'mean',
        'session_duration': 'mean',
        'total_sessions': 'mean'
    }).round(2)

def create_segment_summary(segment_stats):
    """
    Create a readable summary of segment characteristics
    """
    segment_descriptions = {
        'High-Value Customers': 'Frequent purchases, long sessions',
        'Regular Customers': 'Average purchase frequency',
        'Occasional Browsers': 'Low purchase rate, short sessions',
        'New/Inactive Users': 'Very few interactions'
    }
    
    summary = pd.DataFrame({
        'Segment': segment_descriptions.keys(),
        'Description': segment_descriptions.values(),
        'Size': segment_stats[('event_type', 'count')].values,
        'Avg_Purchases': segment_stats[('purchased', 'mean')].values,
        'Avg_Session_Duration': segment_stats[('session_duration', 'mean')].values
    })
    
    return summary
