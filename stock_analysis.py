#imports
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#fetches S&P 500 tickers from wikipedia, if there's any errors,
#returns hard coded S&P tickers.
def get_sp500_tickers():
    try:
        print("Fetching S&P 500 tickers from Wikipedia...")
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = list(table[0]['Symbol'])
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        # Fallback to top companies if Wikipedia scraping fails
        print("Using fallback list of major S&P 500 companies")
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'BRK-B', 
                'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC', 
                'ABBV', 'PFE', 'AVGO', 'COST', 'DIS', 'KO', 'MRK', 'PEP', 'CSCO', 'TMO']

#creates dataframe to store all the stock data
#splits list of tickers into smaller chunks 
#creates smaller lists of tickers called sublists
def download_data_in_chunks(tickers, start_date, end_date, chunk_size=25):
    all_data = pd.DataFrame()
    
    ticker_chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    for i, chunk in enumerate(ticker_chunks):
        print(f"Downloading chunk {i+1}/{len(ticker_chunks)} ({len(chunk)} tickers)")
        try:
            chunk_data = yf.download(chunk, start=start_date, end=end_date)
            
            #goes through each group of tickers
            #prints messages indicating progress
            #fetches stock data for each chunk
            if 'Adj Close' in chunk_data.columns:
                close_data = chunk_data['Adj Close']
            else:
                close_data = chunk_data['Close']
            
            #checks if data has adj close, if not,
            #uses regular close price
            if all_data.empty:
                all_data = close_data
            else:
                all_data = pd.concat([all_data, close_data], axis=1)
                
            #adds new downloaded data to dataframe
            if i < len(ticker_chunks) - 1:
                time.sleep(1)
            
        except Exception as e:
            print(f"Error downloading chunk {i+1}: {e}")
    
    return all_data

#main part of the code
if __name__ == "__main__":
    #calls function to get S&P tickers
    #prints how many tickers were found
    all_tickers = get_sp500_tickers()
    print(f"Found {len(all_tickers)} S&P 500 tickers")
    
    #downloads stock data for selected timeframe
    start_date = '2024-01-01'  
    end_date = '2025-01-31'
    
    print(f"Downloading stock data from {start_date} to {end_date}")
    close_prices = download_data_in_chunks(all_tickers, start_date, end_date)
    
    if close_prices.empty:
        print("No stock data was downloaded. Please check your internet connection or tickers.")
        exit(1)
    
    #data cleaning, removing cols with missing vals
    missing_pct = close_prices.isnull().mean()
    valid_columns = missing_pct[missing_pct < 0.1].index.tolist()
    print(f"Keeping {len(valid_columns)} stocks with sufficient data")
    
    if len(valid_columns) == 0:
        print("No valid columns found after filtering. Using all columns and filling missing values.")
        valid_columns = close_prices.columns.tolist()
    
    close_prices = close_prices[valid_columns]
    
    #filling missing vals
    close_prices = close_prices.ffill().bfill()
    
    #choose target stock for predictions here
    target_stock = 'AAPL' if 'AAPL' in valid_columns else valid_columns[0]
    print(f"Using {target_stock} as the target stock for prediction")
    
    #shifted target price for prediction
    close_prices[f'{target_stock}_shifted'] = close_prices[target_stock].shift(-1)
    close_prices.dropna(inplace=True)
    
    #selecting most relevant features
    correlations = close_prices.corr()[f'{target_stock}_shifted'].abs().sort_values(ascending=False)
    top_features = correlations[1:21].index.tolist() 
    
    print("\nTop 5 most correlated stocks:")
    for i, ticker in enumerate(correlations[1:6].index.tolist()):
        corr_value = correlations[ticker]
        print(f"{i+1}. {ticker}: {corr_value:.4f} correlation")
    
    #x is features, y is target
    X = close_prices[top_features]
    y = close_prices[f'{target_stock}_shifted']
    
    #splits data to train ML model 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining model with {len(X_train)} samples, testing with {len(X_test)} samples")
    
    #training random forest algorithm
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    #makes predictions
    y_pred = model.predict(X_test)
    
    #calculating r squared and MAE
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nPrediction Results:")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"R-Squared (R²): {r2:.4f}")
    
    #feature importance analysis 
    feature_importance = pd.DataFrame({
        'Feature': top_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 5 most important features for prediction:")
    for i, (index, row) in enumerate(feature_importance.head(5).iterrows()):
        print(f"{i+1}. {row['Feature']}: {row['Importance']:.4f} importance")
    
    #converts predictions to series so we can plot it
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    
    #getting last 60 days for a zoomed in view
    days_to_show = min(60, len(y_test))
    zoom_start = y_test.index[-days_to_show]
    y_test_zoom = y_test.loc[zoom_start:]
    y_pred_zoom = y_pred_series.loc[zoom_start:]
    
    #plotting
    fig, axes = plt.subplots(3, 1, figsize=(9, 7))
    
    #plot - actual vs predicted - zoomed in
    axes[0].plot(y_test_zoom.index, y_test_zoom, color='skyblue', label=f'Actual {target_stock} Price', linewidth=2)
    axes[0].plot(y_pred_zoom.index, y_pred_zoom, color='midnightblue', label='Predicted Price', 
                linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0].set_title(f'Actual vs Predicted {target_stock} Stock Price (Last {days_to_show} Days)', fontsize=14)
    axes[0].set_xlabel('Date', fontsize=10)
    axes[0].set_ylabel('Price (USD)', fontsize=10)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    #plot 2 - all data 
    axes[1].plot(y_test.index, y_test, color='skyblue', label=f'Actual {target_stock} Price', linewidth=1.5)
    axes[1].plot(y_pred_series.index, y_pred_series, color='midnightblue', label='Predicted Price', 
                linestyle='--', linewidth=1, alpha=0.7)
    axes[1].set_title(f'Full Test Period: Actual vs Predicted {target_stock} Stock Price', fontsize=14)
    axes[1].set_xlabel('Date', fontsize=10)
    axes[1].set_ylabel('Price (USD)', fontsize=10)
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    #plot 3 - feature importance 
    top_n_features = 10
    importance_data = feature_importance.head(top_n_features)
    bars = axes[2].barh(importance_data['Feature'], importance_data['Importance'], color='lightseagreen')
    axes[2].set_title(f'Top {top_n_features} Most Important Features for Prediction', fontsize=14)
    axes[2].set_xlabel('Importance', fontsize=10)
    axes[2].invert_yaxis()  # Display the highest importance at the top
    
    #importance values added as text
    for bar in bars:
        width = bar.get_width()
        axes[2].text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.show()
    
    #display final summary of results 
    print(f"\nAnalysis Summary for {target_stock}:")
    print(f"- Period analyzed: {start_date} to {end_date}")
    print(f"- Model accuracy (R²): {r2:.4f}")
    print(f"- Average prediction error (MAE): ${mae:.2f}")
    print(f"- Top predictive stock: {feature_importance.iloc[0]['Feature']}")
    print(f"- Data points analyzed: {len(close_prices)}")
