import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def calculate_indicators(data):
    """Calculate technical indicators without TA-Lib"""
    df = data.copy()
    
    # EMA calculations
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility (ATR-like)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    
    # Volume indicators
    df['Volume_EMA'] = df['Volume'].ewm(span=20, adjust=False).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_EMA']
    
    # Price momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['Rate_of_Change'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
    
    # Trend strength
    df['Trend_Strength'] = np.abs(df['EMA9'] - df['EMA21']) / df['ATR']
    
    return df

class PricePrediction:
    def __init__(self, sequence_length=30):  
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def create_model(self):
        """Create a faster and more responsive model"""
        model = Sequential([
            LSTM(128, input_shape=(self.sequence_length, 5), return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss='mean_squared_error')
        self.model = model
        return model

    def prepare_data(self, data):
        """Prepare data with volatility features"""
        try:
            df = pd.DataFrame(data)
            
            # Ensure column names are consistent
            df.columns = [col.lower() for col in df.columns]
            
            # Calculate volatility indicators
            df['hl_diff'] = df['high'] - df['low']
            df['co_diff'] = abs(df['close'] - df['open'])
            df['vol_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
            
            # Calculate fast moving averages
            df['ema5'] = df['close'].ewm(span=5).mean()
            df['ema10'] = df['close'].ewm(span=10).mean()
            
            # Add momentum indicators
            df['momentum'] = df['close'].pct_change(3)
            df['rate_of_change'] = df['close'].pct_change()
            
            # Volatility adjustment factor
            df['volatility'] = df['close'].rolling(5).std()
            df['vol_factor'] = df['volatility'] / df['volatility'].rolling(20).mean()
            
            # Prepare features
            features = np.column_stack((
                df['close'].values,
                df['hl_diff'].values,
                df['vol_ratio'].values,
                df['momentum'].values,
                df['vol_factor'].values
            ))
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            return scaled_features
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None

    def predict(self, data):
        """Make faster predictions with volatility adjustment"""
        try:
            # Prepare data with volatility features
            prepared_data = self.prepare_data(data)
            if prepared_data is None:
                return None
                
            # Get recent volatility
            df = pd.DataFrame(data)
            recent_volatility = df['close'].rolling(5).std().iloc[-1]
            volatility_factor = recent_volatility / df['close'].rolling(20).std().mean()
            
            # Make prediction
            sequence = prepared_data[-self.sequence_length:]
            sequence = sequence.reshape((1, self.sequence_length, 5))
            prediction = self.model.predict(sequence, verbose=0)[0][0]
            
            # Adjust prediction based on volatility
            if volatility_factor > 1.5:  # High volatility
                # Make prediction more conservative
                current_price = df['close'].iloc[-1]
                prediction = current_price + (prediction - current_price) * 0.7
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None

    def predict_trend(self, prediction_data, analysis_data):
        """Short-term prediction focused on quick price movements"""
        try:
            # Get current price and volatility
            current_price = prediction_data['Close'].iloc[-1]
            volatility = analysis_data.get('volatility', 1.0)
            
            # Calculate short-term price changes (1min, 3min, 5min)
            price_changes = []
            for period in [1, 3, 5]:
                change = (current_price - prediction_data['Close'].iloc[-period]) / prediction_data['Close'].iloc[-period] * 100
                price_changes.append(change)
            
            # Get market conditions
            rsi = analysis_data.get('RSI', 50)
            momentum = analysis_data.get('momentum', 'weak')
            
            # Quick reversal detection
            last_candles = prediction_data.tail(3)
            bearish_reversal = (last_candles['High'].iloc[-1] < last_candles['High'].iloc[-2] and 
                              last_candles['Close'].iloc[-1] < last_candles['Open'].iloc[-1])
            bullish_reversal = (last_candles['Low'].iloc[-1] > last_candles['Low'].iloc[-2] and 
                              last_candles['Close'].iloc[-1] > last_candles['Open'].iloc[-1])
            
            # Short-term trend analysis
            short_ema = prediction_data['Close'].ewm(span=5).mean().iloc[-1]
            medium_ema = prediction_data['Close'].ewm(span=8).mean().iloc[-1]
            
            # Aggressive sell conditions
            if ((price_changes[0] < -0.05 and price_changes[1] < -0.1) or  # Quick drop
                (bearish_reversal and rsi > 55) or  # Reversal from high
                (short_ema < medium_ema and rsi > 60)):  # Trend change with high RSI
                logger.info(f"SELL Signal: Changes={price_changes}, RSI={rsi:.2f}")
                return 'SELL'
                
            # Aggressive buy conditions
            elif ((price_changes[0] > 0.05 and price_changes[1] > 0.1) or  # Quick rise
                  (bullish_reversal and rsi < 45) or  # Reversal from low
                  (short_ema > medium_ema and rsi < 40)):  # Trend change with low RSI
                logger.info(f"BUY Signal: Changes={price_changes}, RSI={rsi:.2f}")
                return 'BUY'
            
            return 'NEUTRAL'
            
        except Exception as e:
            logger.error(f"Error predicting trend: {e}")
            return None

    def analyze_market_conditions(self, data):
        """Enhanced market analysis with short-term focus"""
        try:
            df = pd.DataFrame(data)
            df.columns = [col.lower() for col in df.columns]
            
            # Very short-term EMAs
            df['ema5'] = df['close'].ewm(span=5).mean()
            df['ema8'] = df['close'].ewm(span=8).mean()
            df['ema13'] = df['close'].ewm(span=13).mean()
            
            # Fast RSI calculation (7 period)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate 1-minute momentum
            df['momentum'] = df['close'].pct_change(1)
            recent_momentum = df['momentum'].tail(3).mean()
            
            # Quick trend detection
            if df['ema5'].iloc[-1] > df['ema8'].iloc[-1] > df['ema13'].iloc[-1]:
                if recent_momentum > 0.001:
                    trend = 'strong_up'
                else:
                    trend = 'uptrend'
            elif df['ema5'].iloc[-1] < df['ema8'].iloc[-1] < df['ema13'].iloc[-1]:
                if recent_momentum < -0.001:
                    trend = 'strong_down'
                else:
                    trend = 'downtrend'
            else:
                trend = 'neutral'
            
            # Calculate volatility (3-period)
            volatility = df['close'].rolling(3).std().iloc[-1] / df['close'].iloc[-1]
            
            return {
                'trend': trend,
                'RSI': rsi.iloc[-1],
                'momentum': 'strong' if abs(recent_momentum) > 0.001 else 'weak',
                'volatility': volatility
            }
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return None

    def get_price_direction(self, current_price, predicted_price):
        """Get price direction with volatility threshold"""
        try:
            change = (predicted_price - current_price) / current_price
            
            # Adjust threshold based on recent predictions accuracy
            if abs(change) > 0.0005:  # Increased sensitivity
                return 1 if change > 0 else -1
            return 0
            
        except Exception as e:
            logger.error(f"Error getting price direction: {e}")
            return 0

def get_price_direction(current_price, predicted_price, threshold=0.001):
    """Determine price direction with dynamic threshold"""
    price_change = (predicted_price - current_price) / current_price
    
    # Adjust threshold based on recent volatility
    if abs(price_change) > threshold * 2:
        threshold *= 1.5
    
    if price_change > threshold:
        return 1
    elif price_change < -threshold:
        return -1
    else:
        return 0
