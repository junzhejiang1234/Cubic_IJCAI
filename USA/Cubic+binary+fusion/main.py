import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split,Subset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F  
import cupy as cp 

import torch
import torch.optim as optim
import math
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime, timedelta
from scipy.stats import pearsonr, spearmanr
import random
import traceback
import csv
import os
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(5)

class DataProcessor:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
    def download_stocks_data(self):
        dfs = []
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=self.start_date, end=self.end_date)
                if not df.empty:
                    df['Symbol'] = ticker
                    dfs.append(df)
                    print(f"Successfully downloaded data for {ticker}")
            except Exception as e:
                print(f"Error downloading {ticker}: {str(e)}")
        
        if not dfs:
            raise ValueError("No data was downloaded for any ticker")
        
        
        
        df_combined = pd.concat(dfs, axis=0)
        df_clean = df_combined.dropna()
        return df_clean
    
    def filter_stocks_with_max_data(self, stocks_data):
        symbol_counts = stocks_data.groupby('Symbol').size().sort_values(ascending=False)
        print("\nData points per symbol:")
        print(symbol_counts)
        
        stocks_with_max_rows = symbol_counts[symbol_counts == symbol_counts.max()].index.tolist()
        print(f"\nSelected stocks with {symbol_counts.max()} data points: {stocks_with_max_rows}")
        
        all_data = stocks_data[stocks_data['Symbol'].isin(stocks_with_max_rows)].copy()
        return all_data
    
    @staticmethod
    def calculate_returns(prices):
        returns = (prices.shift(-1) - prices) / prices
        return returns
 

    def create_time_features_array(self, df, window=5):
        
        symbols = df['Symbol'].unique()
        dates = df.index.unique().sort_values()
        n_symbols = len(symbols)
        n_features = 9
        n_dates = len(dates)
        
        features_array = cp.zeros((n_dates - window + 1, n_symbols, window, n_features))
        
        feature_columns = ['Close','MA20','RSI','MACD','Volatility','Momentum','Volume_Change','VWAP','Volume']
        df_features = df[feature_columns].values
        df_features_gpu = cp.array(df_features)
        
        symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}
        date_to_idx = {date: i for i, date in enumerate(dates)}
        def process_window(i, j, symbol, window_size):
            for k in range(window_size):
                mask = (df.index == dates[i+k]) & (df['Symbol'] == symbol)
                if mask.any():
                    features_array[i, j, k] = df_features_gpu[mask]
    
        for i in range(n_dates - window + 1):
            for j, symbol in enumerate(symbols):
                process_window(i, j, symbol, window)
        
        return cp.asnumpy(features_array), dates[window:-4]


def save_features_data(features_array, dates, file_path='d:/BaiduNetdiskDownload/mamba (2)/mamba/nine_indicator.npz'):
    dates_str = np.array([str(d) for d in dates])
    np.savez(file_path, features_array=features_array, dates=dates_str)

def load_features_data(file_path='d:/BaiduNetdiskDownload/mamba (2)/mamba/usa_16_indicators.npz'):
    loaded_data = np.load(file_path, allow_pickle=True)
    features_array = loaded_data['features_array']
    dates = pd.to_datetime(loaded_data['dates'])
    return features_array, dates

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)  
        self.y = torch.FloatTensor(y)  
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attn(x, x, x)
        x = residual + self.dropout(attn_output)
        
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        return x

class StockFeatureEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0.5):
        super(StockFeatureEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, x):
        batch_size, num_stocks, look_back, feature_dim = x.shape
        x = x.view(-1, feature_dim)
        encoded = self.mlp(x)
        encoded = encoded.view(batch_size, num_stocks, look_back, -1)
        return encoded

class StockSelectorWithMultiplePooling(nn.Module):
    def __init__(self, input_dim, latent_dim, pool_size=2, pool_stride=2):
        super(StockSelectorWithMultiplePooling, self).__init__()
        self.encoder = StockFeatureEncoder(input_dim, latent_dim)
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.latent_dim = latent_dim
        
        # 定义三种池化层
        self.max_pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.min_pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride) 
        self.mean_pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride)
        
        # 定义 MLP 来处理拼接后的池化特征
        self.mlp = nn.Sequential(
            nn.Linear(3 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        encoded = self.encoder(x)  
        
        encoded = encoded.permute(0, 1, 3, 2).contiguous().view(batch_size * encoded.size(1), self.latent_dim, -1)
        
  
        max_pooled = self.max_pool(encoded)
        
        min_pooled = self.max_pool(-encoded).neg()  
        mean_pooled = self.mean_pool(encoded)  
        
  
        combined_pooled = torch.cat([max_pooled, min_pooled, mean_pooled], dim=1)
        
    
        combined_pooled = combined_pooled.mean(dim=2)  
        
        final_features = self.mlp(combined_pooled) 
        
        final_features = final_features.view(batch_size, -1, self.latent_dim) 
        
        return final_features

class StockLSTM(nn.Module):
    def __init__(self, 
                 feature_dim=9,
                 latent_dim=64, 
                 hidden_dim=128, 
                 num_layers=2,   
                 dropout=0.1,
                 num_stocks=29,
                 look_back=5,
                 num_binary_positions=15,
                 pool_size=2,
                 pool_stride=2, 
                 device='cuda'):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_stocks = num_stocks
        self.look_back = look_back
        self.device = device
        self.num_binary_positions = num_binary_positions
        
        self.selector = StockSelectorWithMultiplePooling(input_dim=feature_dim, latent_dim=latent_dim, pool_size=pool_size, pool_stride=pool_stride)
        
        self.lstm = nn.LSTM(
            input_size=latent_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
   
        input_dim = hidden_dim * num_stocks  
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_binary_positions * 2)  
        )
        self._init_weights()
        self.to(device)
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.size(0)
        
        if x.size(1) == self.look_back:
            x = x.permute(0, 2, 1, 3)
        
        pooled_features = self.selector(x)  
        

        lstm_input = pooled_features 
        lstm_output, _ = self.lstm(lstm_input)  
        
        flattened = lstm_output.reshape(batch_size, -1) 
        logits = self.output_layer(flattened) 
        

        logits = logits.view(batch_size, self.num_binary_positions, 2) 
        
        return logits




import math

def binary_to_float(binary_list, min_val=-1, max_val=1, precision=0.0001):
    
    total_steps = int((max_val - min_val) / precision)  
    bit_length = math.ceil(math.log2(total_steps))

    if len(binary_list) != bit_length:
        raise ValueError(f"Expected binary_list of length {bit_length}, but got {len(binary_list)}.")


    binary_str = ''.join(str(b) for b in binary_list) 
    integer_val = int(binary_str, 2)
    integer_val = ((integer_val - 10000))*0.0001
    

    # 保留 4 位小数
    return round(integer_val, 4)





import torch
import torch.optim as optim

class ModelTrainer:
    def __init__(self, model, criterion=None, optimizer=None, scheduler=None, device='cuda'):
        self.model = model.to(device)
        self.device = device

        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=0.001)
        
        self.scheduler = scheduler or torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3)
        
        self.criterion =torch.nn.CrossEntropyLoss()
        
        # 调试输出
        print(f"Type of self.criterion: {type(self.criterion)}")

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_bit_losses = []  

        for data, target in train_loader:
            data = data.to(self.device)
            target = target.long().to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)  

            bit_losses = []  

            for bit_pos in range(len(output)):
                bit_predictions = output[bit_pos] 
                bit_targets = target[bit_pos]     

                bit_loss = self.criterion(bit_predictions, bit_targets)
                bit_losses.append(bit_loss) 


            all_bit_losses.append([loss.item() for loss in bit_losses])

            loss = torch.stack(bit_losses).mean()  

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)



    
    def validate(self, val_loader):
        self.model.eval() 
        total_loss = 0  
        all_bit_losses = [] 
        predictions = []  
        actuals = [] 

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)  # [batch_size, look_back, num_stocks, feature_dim]
                target = target.long().to(self.device)  # [batch_size, num_binary_positions]

                output = self.model(data)  # [batch_size, num_binary_positions, 2]

                bit_losses = []

                for bit_pos in range(len(output)):
                    bit_predictions = output[bit_pos]  # [num_binary_positions, 2]
                    bit_targets = target[bit_pos]      # [num_binary_positions]

                    bit_loss = self.criterion(bit_predictions, bit_targets)
                    bit_losses.append(bit_loss) 

                all_bit_losses.append([loss.item() for loss in bit_losses])

                loss = torch.stack(bit_losses).mean()  
                total_loss += loss

                preds = torch.argmax(output, dim=2)  
                predictions.append(preds.cpu())  
                actuals.append(target.cpu())  

        predictions = torch.cat(predictions, dim=0).numpy()
        actuals = torch.cat(actuals, dim=0).numpy()  
        decoded_predictions = [binary_to_float(row, min_val=-1, max_val=1, precision=0.0001) for row in predictions]
        decoded_actuals  = [binary_to_float(row, min_val=-1, max_val=1, precision=0.0001) for row in actuals]
        return total_loss / len(val_loader),predictions, actuals


    def train(self, train_loader, val_loader, num_epochs=100, patience=10):
       
        self.model.to(self.device)
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
     
            train_loss = self.train_epoch(train_loader)
            val_loss, predictions, actuals = self.validate(val_loader)
            
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                current_path= os.path.dirname(os.path.abspath(__file__))

                output_file = os.path.join(current_path, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, output_file)
                print(f"  Best model saved with Val Loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                print(f"  Patience Counter: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after epoch {epoch+1}")
                    break

        return train_losses, val_losses


def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()
def calculate_ic(actuals, predictions):
    ic_daily = [pearsonr(a.flatten(), p.flatten())[0] for a, p in zip(actuals, predictions)]
    return np.mean(ic_daily), np.std(ic_daily)

def calculate_rank_ic(actuals, predictions):
    rank_ic_daily = [spearmanr(a.flatten(), p.flatten())[0] for a, p in zip(actuals, predictions)]
    return np.mean(rank_ic_daily), np.std(rank_ic_daily)
def process_stock_data(stocks_data):
    
    def calculate_features(group):
      
        group['MA5'] = group['Close'].rolling(window=5).mean()
        group['MA20'] = group['Close'].rolling(window=20).mean()
        
 
        delta = group['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        group['RSI'] = 100 - (100 / (1 + rs))
        
   
        exp1 = group['Close'].ewm(span=12, adjust=False).mean()
        exp2 = group['Close'].ewm(span=26, adjust=False).mean()
        group['MACD'] = exp1 - exp2
        group['Signal_Line'] = group['MACD'].ewm(span=9, adjust=False).mean()
        
    
        group['Volatility'] = group['Close'].rolling(window=20).std()
        
    
        group['Momentum'] = group['Close'] - group['Close'].shift(10)
        
       
        group['Volume_Change'] = group['Volume'].pct_change()
   
        group['BB_Middle'] = group['Close'].rolling(window=20).mean()
        rolling_std = group['Close'].rolling(window=20).std()
        group['BB_Upper'] = group['BB_Middle'] + 2 * rolling_std
        group['BB_Lower'] = group['BB_Middle'] - 2 * rolling_std
        
     
        group['Price_Change'] = group['Close'].pct_change()
        
      
        group['HL_Position'] = (group['Close'] - group['Low']) / (group['High'] - group['Low'])
        
    
        group['VWAP'] = (group['Volume'] * (group['High'] + group['Low'] + group['Close']) / 3).cumsum() / group['Volume'].cumsum()
        
        return group


    processed_data = stocks_data.groupby('Symbol', group_keys=False).apply(calculate_features)
    

    processed_data = processed_data.fillna(0)
    
  
    features_to_standardize = ['Close','MA20','RSI','MACD','Volatility','Momentum','Volume_Change','VWAP','Volume']
    
  
    def normalize_group(group):
        for feature in features_to_standardize:
            if feature in group.columns:
               
                non_zero_values = group[feature][group[feature] != 0]
                min_val = non_zero_values.min() if not non_zero_values.empty else 0
                max_val = group[feature].max()
              
                if max_val != min_val:
                    group[feature] = (group[feature] - min_val) / (max_val - min_val)
                else:
                    group[feature] = 0.5  
        return group


    normalized_data = processed_data.groupby('Symbol', group_keys=False).apply(normalize_group)
    
    return normalized_data

def backtest_result(test_predictions,test_actuals):
    test_predictions = np.array(test_predictions)
    test_actuals = np.array(test_actuals)

    signals = np.where(test_predictions > 0, 1, 0)

 
    initial_capital = 1000000  # 100万

 
    position_returns = signals * test_actuals  

    strategy_cumulative_returns = np.cumprod(1 + position_returns) - 1
    portfolio_value = initial_capital * (1 + strategy_cumulative_returns)

    market_cumulative_returns = np.cumprod(1 + test_actuals) - 1
    market_value = initial_capital * (1 + market_cumulative_returns)

    excess_returns = position_returns - test_actuals

    total_return = (portfolio_value[-1] - initial_capital) / initial_capital
    market_total_return = (market_value[-1] - initial_capital) / initial_capital

    days_per_year = 252
    total_days = len(test_predictions)
    ar = (1 + total_return) ** (days_per_year/total_days) - 1
    market_ar = (1 + market_total_return) ** (days_per_year/total_days) - 1

    excess_return = ar - market_ar
    tracking_error = np.std(excess_returns) * np.sqrt(days_per_year)
    ir = excess_return / tracking_error if tracking_error != 0 else 0



    risk_free_rate = 0 
    excess_return_over_rf = ar - risk_free_rate
    volatility = np.std(position_returns) * np.sqrt(days_per_year)
    sharpe_ratio = excess_return_over_rf / volatility if volatility != 0 else 0

     
    def calculate_max_drawdown(portfolio_values):
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        max_drawdown = np.min(drawdown)
        return max_drawdown

    max_drawdown = calculate_max_drawdown(portfolio_value)

    calmar_ratio = ar / abs(max_drawdown) if max_drawdown != 0 else 0

    
    downside_returns = np.minimum(position_returns - risk_free_rate / days_per_year, 0)
    downside_deviation = np.std(downside_returns) * np.sqrt(days_per_year)
    sortino_ratio = excess_return_over_rf / downside_deviation if downside_deviation != 0 else 0

    profitable_trades = position_returns > 0
    win_rate = np.sum(profitable_trades) / len(position_returns)
    average_gain = np.mean(position_returns[profitable_trades]) if np.sum(profitable_trades) > 0 else 0
    average_loss = np.mean(position_returns[~profitable_trades]) if np.sum(~profitable_trades) > 0 else 0
    profit_loss_ratio = abs(average_gain / average_loss) if average_loss != 0 else float('inf')



    covariance = np.cov(position_returns, test_actuals)[0][1]
    variance = np.var(test_actuals)
    beta = covariance / variance if variance != 0 else 0

    alpha = ar - (risk_free_rate + beta * (market_ar - risk_free_rate))
    direction_accuracy = np.mean((test_predictions > 0) == (test_actuals > 0))
    return direction_accuracy, sharpe_ratio, ar,position_returns
def calculate_backtest_ic(test_predictions, test_actuals, window_size=40, plot=True):
 
    
    data = pd.DataFrame({
        'factor': test_predictions,
        'future_return': test_actuals
    })
    
 
    data['date'] = data.index
    

    ic_series = []
    dates = []
    
    for date, group in data.groupby('date'):
        factor_values = group['factor']
        future_returns = group['future_return']
        
      
        ic, _ = stats.spearmanr(factor_values, future_returns)
        ic_series.append(ic)
        dates.append(date)
    
    ic_series = np.array(ic_series)
    
 
    ic = np.mean(ic_series)
    
  
    ic_std = np.std(ic_series)
    icir = ic / ic_std if ic_std != 0 else 0
    

    
    print(f"IC: {ic:.6f}")
    print(f"ICIR: {icir:.6f}")
    
    return ic, icir, ic_series



def calculate_portfolio_metrics(actuals, predictions, top_n=30):
    daily_returns = []
    for day_actuals, day_predictions in zip(actuals, predictions):
        top_indices = np.argsort(day_predictions.flatten())[-top_n:]
        daily_return = np.mean(day_actuals.flatten()[top_indices])
        daily_returns.append(daily_return)
    
    excess_returns = np.array(daily_returns) - np.mean([np.mean(a) for a in actuals])
    ar = np.mean(excess_returns) * 252  
    ir = ar / (np.std(excess_returns) * np.sqrt(252))
    
    return ar, ir
def plot_predictions(predictions, actuals, dates=None):
    plt.figure(figsize=(12, 6))
    if dates is not None:
        plt.plot(dates, actuals, label='Actual Returns')
        plt.plot(dates, predictions, label='Predicted Returns')
        plt.gcf().autofmt_xdate() 
    else:
        plt.plot(actuals, label='Actual Returns')
        plt.plot(predictions, label='Predicted Returns')
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.title('Stock Returns: Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()
    


def float_to_binary(number, min_val=-1, max_val=1, precision=0.0001):
    
    integer_val = int(number * 10000 + 10000)

   
    binary = format(integer_val, '015b')
    return [int(b) for b in binary]
def main():
    # 配置参数
    START_DATE = '2012-11-08'
    END_DATE = '2024-11-08'
    WINDOW_SIZE = 5 
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 1000
    PATIENCE = 40
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('./USA.csv')

    DJIA_returns = df['DJIA100_returns'].tolist()
    X, dates = load_features_data()
    print("X shape:", X.shape)

    Y = np.array(DJIA_returns[WINDOW_SIZE:])
    Y[-1] = 0
    binary_data = [float_to_binary(num) for num in Y]
    binary_data = torch.tensor(binary_data)
    
# 5. 输出结果
    Y = binary_data
    # 如果 Y 是整数类型的张量
    Y = torch.tensor(Y, dtype=torch.long)

    # 转换为浮点类型
    Y = Y.float()


    # 数据集划分
    total_samples = len(X)-5
    train_size = int(0.7 * total_samples)
    val_size = int(0.2 * total_samples)
    test_size = total_samples - train_size - val_size
    

    
    # 创建数据集
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    dataset = StockDataset(X, Y)
    train_dataset = Subset(dataset, range(train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, total_samples-1))
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    


    # 损失函数
    criterion = torch.nn.BCEWithLogitsLoss()    
    
    model = StockLSTM(
    feature_dim=16,
    latent_dim=64,  
    dropout=0.1,
    num_stocks=29,
    look_back=5
)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.999,
        patience=20,
        verbose=True,
        min_lr=1e-6
    )


    class EarlyStopping:
        def __init__(self, patience=7, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False

        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.counter = 0
            return self.early_stop

    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    print("\nStarting training...")
    trainer = ModelTrainer(model, criterion, optimizer, scheduler, DEVICE)
    train_losses, val_losses = trainer.train(
        train_loader, val_loader, NUM_EPOCHS, PATIENCE
    )

   
    print("\nLoading best model and performing test...")
    current_path= os.path.dirname(os.path.abspath(__file__))

    output_file = os.path.join(current_path, 'best_model.pth')
    checkpoint = torch.load(output_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_predictions, test_actuals = trainer.validate(test_loader)
    test_actuals = np.array(test_actuals) 
    test_predictions = np.array(test_predictions) 
    test_actuals = np.array(test_actuals) 
    test_predictions = np.array(test_predictions) 
    test_predictions = [binary_to_float(row, min_val=-1, max_val=1, precision=0.0001) for row in test_predictions]
    test_actuals = [binary_to_float(row, min_val=-1, max_val=1, precision=0.0001) for row in test_actuals]
    direction_accuracy, sharpe_ratio, ar,position_return = backtest_result(test_predictions,test_actuals)

    ic = stats.spearmanr(test_actuals, test_predictions)[0]  
    window_size = 40
    ic_series = []
    for i in range(len(test_predictions) - window_size + 1):
        window_ic = stats.spearmanr(test_actuals[i:i+window_size], 
                                test_predictions[i:i+window_size])[0]
        ic_series.append(window_ic)

    icir = ic / np.std(ic_series)

    print(f"IC: {ic:.6f}")
    print(f"ICIR: {icir:.6f}")
    print('ic ',ic)
    print('icir ',icir)
    ic = np.corrcoef(test_actuals, test_predictions)[0,1]
    ic2 = np.corrcoef(test_predictions,test_actuals)[0,1]
    print(ic,ic2)
    return ic, icir, direction_accuracy, sharpe_ratio, ar,position_return
import random
import traceback
import csv
import os
if __name__ == "__main__":
    try:
        ic, icir, direction_accuracy, sharpe_ratio, ar,position_return = [],[],[],[],[],[]
        for i in range(10):
            random_seed = random.randint(0, 10000)  # 生成一个 0 到 10000 之间的随机整数
            ic_get, icir_get, direction_accuracy_get, sharpe_ratio_get, ar_get,position_return_get = main()
            if (ic_get != None and icir_get != None):
                set_seed(random_seed)
                
                ic.append(ic_get)
                icir.append(icir_get)
                direction_accuracy.append(direction_accuracy_get)
                sharpe_ratio.append(sharpe_ratio_get)
                position_return.append(position_return_get)
                ar.append(ar_get)

        def calculate_mean_and_std(data):
            cleaned_data = pd.Series(data).dropna()  
            mean = cleaned_data.mean()
            std = cleaned_data.std()
            return mean, std
        current_path= os.path.dirname(os.path.abspath(__file__))

        output_file = os.path.join(current_path, "evaluation.csv")

        print('output_file',output_file)
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
       
            writer.writerow(["ic", "icir", "direction_accuracy", "sharpe_ratio", "ar"])
            

            for row in zip(ic, icir, direction_accuracy, sharpe_ratio, ar):
                writer.writerow(row)
        output_file2 = os.path.join(current_path, "position_return.csv")


        with open(output_file2, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
       
            for row in position_return:
                writer.writerow(row)
                
        ic_mean, ic_std = calculate_mean_and_std(ic)
        icir_mean, icir_std = calculate_mean_and_std(icir)
        direction_accuracy_mean, direction_accuracy_std = calculate_mean_and_std(direction_accuracy)
        sharpe_ratio_mean, sharpe_ratio_std = calculate_mean_and_std(sharpe_ratio)
        ar_mean, ar_std = calculate_mean_and_std(ar)
       
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc() 
