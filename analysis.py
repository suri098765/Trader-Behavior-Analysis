import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_and_clean_data(sentiment_path, trades_path):
    # Load datasets
    sentiment_df = pd.read_csv(sentiment_path)
    trades_df = pd.read_csv(trades_path)
    
    # Basic Exploration
    print(f"Sentiment Data: {sentiment_df.shape[0]} rows, {sentiment_df.shape[1]} cols")
    print(f"Trades Data: {trades_df.shape[0]} rows, {trades_df.shape[1]} cols")
    
    # Handle Missing Values
    sentiment_df = sentiment_df.dropna()
    trades_df = trades_df.dropna(subset=['closedPnL', 'leverage', 'account'])
    
    # Date Alignment
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    trades_df['time'] = pd.to_datetime(trades_df['time'])
    trades_df['Date'] = trades_df['time'].dt.normalize() # Align to daily
    
    return sentiment_df, trades_df

def engineer_metrics(trades_df):
    # Aggregate daily metrics per trader
    daily_trader_stats = trades_df.groupby(['Date', 'account']).agg({
        'closedPnL': 'sum',
        'size': 'sum',
        'leverage': 'mean',
        'symbol': 'count' # Number of trades
    }).rename(columns={'symbol': 'trade_count'})
    
    # Calculate Long/Short Ratio
    # side: 1 for long, 0 for short (assuming 'side' column exists)
    trades_df['is_long'] = trades_df['side'].apply(lambda x: 1 if str(x).lower() == 'long' else 0)
    long_ratio = trades_df.groupby(['Date', 'account'])['is_long'].mean().to_frame('long_ratio')
    
    return daily_trader_stats.join(long_ratio).reset_index()


def perform_analysis(df):
    # 1. Performance: Fear vs Greed
    perf_stats = df.groupby('Classification').agg({
        'closedPnL': 'mean',
        'trade_count': 'mean',
        'leverage': 'mean'
    })
    print("\n--- Summary Stats by Sentiment ---")
    print(perf_stats)
    
    # 2. Segmentation: High vs Low Leverage
    df['Leverage_Segment'] = np.where(df['leverage'] > 10, 'High Leverage', 'Low Leverage')
    segment_analysis = df.groupby('Leverage_Segment')['closedPnL'].mean()
    
    # 3. Visualization
    plt.figure(figsize=(12, 6))
    
    # Plot 1: PnL Distribution
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x='Classification', y='closedPnL', hue='Leverage_Segment')
    plt.title('Average PnL: Sentiment & Leverage')
    
    # Plot 2: Trade Frequency
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='Classification', y='trade_count')
    plt.title('Trade Frequency by Sentiment')
    
    plt.tight_layout()
    plt.savefig('insights_chart.png') # Save for GitHub
    plt.show()


# Note: Update these filenames to match the ones you upload to GitHub
SENTIMENT_FILE = 'bitcoin_sentiment.csv' 
TRADES_FILE = 'hyperliquid_trades.csv'

try:
    # 1. Load and Clean
    sentiment, trades = load_and_clean_data(SENTIMENT_FILE, TRADES_FILE)
    
    # 2. Process
    daily_metrics = engineer_metrics(trades)
    
    # 3. Merge Sentiment with Metrics
    final_df = pd.merge(daily_metrics, sentiment, on='Date', how='inner')
    
    # 4. Analyze
    perform_analysis(final_df)
    
    print("\nAnalysis Complete. Check 'insights_chart.png' for visualizations.")

except Exception as e:
    print(f"Error: {e}. Ensure CSV files are in the same folder as this script.")
