import pandas as pd
import ast
import numpy as np

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Trade_History'])  # Drop missing values
    df['Trade_History'] = df['Trade_History'].apply(ast.literal_eval)  # Convert to list of dicts
    return df

# Extract relevant trade details
def extract_trade_details(df):
    trade_data = []
    for _, row in df.iterrows():
        port_id = row['Port_IDs']
        for trade in row['Trade_History']:
            trade['Port_ID'] = port_id
            trade_data.append(trade)
    return pd.DataFrame(trade_data)

# Calculate financial metrics
def calculate_metrics(trade_df):
    grouped = trade_df.groupby('Port_ID')
    metrics = []
    
    for port_id, trades in grouped:
        initial_balance = trades.iloc[0]['price'] * trades.iloc[0]['qty'] if not trades.empty else 1
        total_profit = trades['realizedProfit'].sum()
        roi = (total_profit / initial_balance) * 100 if initial_balance else 0
        
        returns = trades['realizedProfit'] / initial_balance
        sharpe_ratio = (returns.mean() / returns.std()) if returns.std() != 0 else 0
        
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        win_positions = (trades['realizedProfit'] > 0).sum()
        total_positions = len(trades)
        win_rate = (win_positions / total_positions) * 100 if total_positions else 0
        
        metrics.append({
            'Port_ID': port_id,
            'ROI': roi,
            'PnL': total_profit,
            'Sharpe_Ratio': sharpe_ratio,
            'MDD': max_drawdown,
            'Win_Rate': win_rate,
            'Win_Positions': win_positions,
            'Total_Positions': total_positions
        })
    
    return pd.DataFrame(metrics)

# Rank accounts
def rank_accounts(metrics_df):
    metrics_df['Score'] = metrics_df['ROI'] + metrics_df['PnL'] + (metrics_df['Sharpe_Ratio'] * 10) - (metrics_df['MDD'] * 100)
    return metrics_df.sort_values(by='Score', ascending=False).head(20)

# Main function
def main(file_path):
    df = load_data(file_path)
    trade_df = extract_trade_details(df)
    metrics_df = calculate_metrics(trade_df)
    top_accounts = rank_accounts(metrics_df)
    
    # Save results
    metrics_df.to_csv('calculated_metrics.csv', index=False)
    top_accounts.to_csv('top_20_accounts.csv', index=False)
    
    print("Analysis complete. Files saved.")

# Run the script
if __name__ == "__main__":
    file_path = "TRADES_CopyTr_90D_ROI.csv"  # Update with actual file path
    main(file_path)