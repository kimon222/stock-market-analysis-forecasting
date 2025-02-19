#imports
import sqlite3
import pandas as pd
import yfinance as yf
import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#fetching S&P 500 tickers from Wikipedia
def get_sp500_tickers():
    try:
        print("Fetching S&P 500 tickers from Wikipedia...")
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = list(table[0]['Symbol'])
        #cleans ticker data, replacing dots with hyphens 
        #for yfinance
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        #if wikipedia scraping fails, we have a plan B
        #of top company tickers 
        print("Using fallback list of major S&P 500 companies")
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'BRK-B', 
                'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC', 
                'ABBV', 'PFE', 'AVGO', 'COST', 'DIS', 'KO', 'MRK', 'PEP', 'CSCO', 'TMO']

#downloads data in chunks 
#to prevent issues like timeout
def download_data_in_chunks(tickers, start_date, end_date, chunk_size=25):
    all_data = pd.DataFrame()
    
    #splits tickers into chunks 
    ticker_chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    for i, chunk in enumerate(ticker_chunks):
        print(f"Downloading chunk {i+1}/{len(ticker_chunks)} ({len(chunk)} tickers)")
        try:
            chunk_data = yf.download(chunk, start=start_date, end=end_date)
            
            #extracting close prices
            if 'Adj Close' in chunk_data.columns:
                close_data = chunk_data['Adj Close']
            else:
                close_data = chunk_data['Close']
            
            #merging with existing data, if any 
            if all_data.empty:
                all_data = close_data
            else:
                all_data = pd.concat([all_data, close_data], axis=1)
                
            #pauses a bit between chunks to avoid
            #rate limiting
            if i < len(ticker_chunks) - 1:
                time.sleep(1)
            
        except Exception as e:
            print(f"Error downloading chunk {i+1}: {e}")
    
    return all_data

#getting S&P 500 tickers 
all_tickers = get_sp500_tickers()
print(f"Found {len(all_tickers)} S&P 500 tickers")

#downloading stock data for 
#a given timeframe
start_date = '2024-01-01'  
end_date = '2025-01-31'

print(f"Downloading stock data from {start_date} to {end_date}")
close_prices = download_data_in_chunks(all_tickers, start_date, end_date)

#check if we have any data
if close_prices.empty:
    print("No stock data was downloaded. Please check your internet connection or tickers.")
    exit(1)

#data cleaning - checking if we have missing values 
missing_pct = close_prices.isnull().mean()
valid_columns = missing_pct[missing_pct < 0.1].index.tolist()
print(f"Keeping {len(valid_columns)} stocks with sufficient data")

if len(valid_columns) == 0:
    print("No valid columns found after filtering. Using all columns and filling missing values.")
    valid_columns = close_prices.columns.tolist()

close_prices = close_prices[valid_columns]

#filling any missing vals
close_prices = close_prices.ffill().bfill() 

#creating a next day prices column for predictions 
print("Creating shifted prices...")
#dont shift too many stocks so that we avoid SQLite column limits
shift_limit = min(len(valid_columns), 20)  
for i, ticker in enumerate(valid_columns[:shift_limit]):
    if i % 5 == 0:
        print(f"Processing ticker {i+1}/{shift_limit}")
    close_prices[f'{ticker}_shifted'] = close_prices[ticker].shift(-1)

#dropping last row which is NaN bc of shifting
close_prices = close_prices.dropna()

#connecting to sql db
print("Connecting to database...")
conn = sqlite3.connect('sp500_stocks.db')
cursor = conn.cursor()

#creating the table 
def create_table_with_columns(cursor, columns):
    #formatting
    column_defs = ["Date TEXT PRIMARY KEY"]
    for col in columns:
        #replacing any special characters that could cause issues 
        safe_col = f'"{col.replace(".", "_")}"'
        column_defs.append(f"{safe_col} REAL")
    
    create_table_sql = f'''
    CREATE TABLE IF NOT EXISTS sp500_stock_prices (
        {", ".join(column_defs)}
    )
    '''
    
    try:
        cursor.execute(create_table_sql)
        return True
    except sqlite3.OperationalError as e:
        print(f"Error creating table: {e}")
        return False

#create table with columns
print("Creating database table...")
all_columns = close_prices.columns.tolist()
if not create_table_with_columns(cursor, all_columns):
    # try fewer columns if there's an error  
    print("Trying with fewer columns due to SQLite limitations...")
    key_stocks = valid_columns[:10]  #first 10 stocks
    shifted_stocks = [f"{s}_shifted" for s in key_stocks[:5]]  
    reduced_columns = key_stocks + shifted_stocks
    
    if not create_table_with_columns(cursor, reduced_columns):
        print("Could not create table even with reduced columns.")
        conn.close()
        exit(1)
    
    #updating dataframe 
    close_prices = close_prices[reduced_columns]

#inserting data into table
print("Inserting data into database...")
#creating column names for sql  
safe_columns = ['Date'] + [col.replace(".", "_") for col in close_prices.columns]
placeholders = ', '.join(['?'] * len(safe_columns))
insert_sql = f'''
INSERT OR REPLACE INTO sp500_stock_prices ({', '.join([f'"{c}"' for c in safe_columns])})
VALUES ({placeholders})
'''

#insert by batches
batch_size = 50
total_rows = len(close_prices)
batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division

for batch in range(batches):
    start_idx = batch * batch_size
    end_idx = min((batch + 1) * batch_size, total_rows)
    
    print(f"Inserting batch {batch+1}/{batches}")
    
    batch_data = []
    for idx in range(start_idx, end_idx):
        date_str = close_prices.index[idx].strftime('%Y-%m-%d')
        row_values = [date_str] + list(close_prices.iloc[idx])
        batch_data.append(row_values)
    
    try:
        cursor.executemany(insert_sql, batch_data)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error inserting data: {e}")
        print("Sample row that caused the issue:", batch_data[0][:5], "...")
        break

#verify data 
print("\nVerifying data insertion...")
try:
    cursor.execute('SELECT COUNT(*) FROM sp500_stock_prices')
    count = cursor.fetchone()[0]
    print(f"Total rows in database: {count}")
    
    cursor.execute('SELECT * FROM sp500_stock_prices LIMIT 2')
    sample_data = cursor.fetchall()
    print(f"\nSample data (first 2 rows, showing first few columns):")
    for row in sample_data:
        print(row[0], row[1:5], "...")
    
    cursor.execute("PRAGMA table_info(sp500_stock_prices)")
    columns_info = cursor.fetchall()
    print(f"\nTotal columns in database: {len(columns_info)}")
    print("First 5 column names:", [col[1] for col in columns_info[:5]])
    
except sqlite3.Error as e:
    print(f"Error querying database: {e}")

#closing db connection
print("Closing database connection...")
conn.close()
print("Done! S&P 500 stock data successfully saved to 'sp500_stocks.db'")