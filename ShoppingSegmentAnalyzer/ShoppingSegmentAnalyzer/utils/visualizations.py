import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_shopping_patterns(df):
    """
    Create a line plot showing shopping patterns over time
    """
    daily_patterns = df.groupby(df['timestamp'].dt.date).agg({
        'event_type': 'count',
        'purchased': 'sum'
    }).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_patterns['timestamp'],
        y=daily_patterns['event_type'],
        name='Total Page Views',
        line=dict(color='#1f77b4')
    ))
    fig.add_trace(go.Scatter(
        x=daily_patterns['timestamp'],
        y=daily_patterns['purchased'],
        name='Orders',
        line=dict(color='#2ca02c')
    ))

    fig.update_layout(
        title='Daily Shopping Activity',
        xaxis_title='Date',
        yaxis_title='Count',
        hovermode='x unified'
    )
    return fig

def plot_user_clusters(df):
    """
    Create a scatter plot of user clusters
    """
    user_data = df.groupby('user_id').agg({
        'total_events': 'first',
        'total_purchases': 'first',
        'segment': 'first'
    }).reset_index()

    fig = px.scatter(
        user_data,
        x='total_events',
        y='total_purchases',
        color='segment',
        title='User Behavior Clusters',
        labels={
            'total_events': 'Page Views',
            'total_purchases': 'Orders',
            'segment': 'Customer Segment'
        }
    )
    return fig

def plot_session_duration(df):
    """
    Create a histogram of session durations
    """
    fig = px.histogram(
        df,
        x='session_duration',
        color='segment',
        title='Session Duration Distribution',
        labels={'session_duration': 'Session Duration (minutes)'},
        nbins=30
    )
    return fig

def plot_purchase_funnel(df):
    """
    Create a purchase funnel visualization based on page categories
    """
    main_categories = df['page 1 (main category)'].value_counts()
    product_views = df['page 2 (clothing model)'].notna().sum()
    orders = df['order'].notna().sum()

    funnel_data = [
        main_categories.sum(),  # Total page views
        product_views,          # Product detail views
        orders                  # Completed orders
    ]

    fig = go.Figure(go.Funnel(
        y=['Category Views', 'Product Views', 'Orders'],
        x=funnel_data,
        textinfo="value+percent initial"
    ))

    fig.update_layout(
        title='Shopping Funnel Analysis',
        showlegend=False
    )
    return fig