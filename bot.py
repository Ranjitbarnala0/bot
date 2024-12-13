import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Layer
from sklearn.preprocessing import MinMaxScaler
import time
import os
import json
import logging
from datetime import datetime
from prediction import PricePrediction
from r import TradingRL
from monitor import TradingMonitor

# Suppress TensorFlow GPU and other unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load configuration settings
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    logger.info("Configuration settings loaded successfully.")
except json.JSONDecodeError as e:
    logger.error(f"Error loading config.json: {e}")
    raise SystemExit(1)

# Initialize MetaTrader 5 connection
if not mt5.initialize():
    logger.error("MetaTrader 5 initialization failed.")
    raise SystemExit(1)
else:
    logger.info("MetaTrader 5 initialized successfully.")

# Validate the availability of BTCUSD symbol
BTC_SYMBOL = "BTCUSD"
symbol_info = mt5.symbol_info(BTC_SYMBOL)
if symbol_info is None or not symbol_info.visible:
    if not mt5.symbol_select(BTC_SYMBOL, True):
        logger.error(f"Failed to select {BTC_SYMBOL}, exiting.")
        raise SystemExit(1)
    else:
        logger.info(f"{BTC_SYMBOL} symbol selected successfully.")
else:
    logger.info(f"{BTC_SYMBOL} symbol is available and visible.")

# Configure trading parameters
LOT_SIZE = 5.0  # Lot size for each trade
HEDGE_LOT_SIZE = 5.0  # Lot size for hedge trades
MAX_TRADES_PER_SESSION = 100
MAX_TRADE_DURATION = 3600  # Maximum trade duration in seconds
TREND_CONFIRMATION_INTERVAL = 60  # Seconds to wait for trend confirmation
TREND_REVERSAL_THRESHOLD = 0.7  # Probability threshold for trend reversal
MAGIC_NUMBER = 234000
INITIAL_STOP_LOSS = 100  # Initial stop loss in dollars
TRAILING_STOP_DISTANCE = 50  # Trailing stop distance in dollars
MIN_PROFIT_TO_TRAIL = 50  # Minimum profit needed to activate trailing stop

# Timeframe configurations
TIMEFRAMES = {
    '1m': mt5.TIMEFRAME_M1,
    '5m': mt5.TIMEFRAME_M5,
    '15m': mt5.TIMEFRAME_M15,
    '30m': mt5.TIMEFRAME_M30,
    '1h': mt5.TIMEFRAME_H1
}

# Initialize prediction model
predictor = PricePrediction(sequence_length=60)
model_path = 'trained_lstm_model.h5'

# Try to load existing model, create new one if not found
try:
    predictor.load_model(model_path)
    logger.info("Loaded existing LSTM model")
except:
    logger.info("Creating new LSTM model")
    predictor.create_model()

# Initialize RL agent
rl_agent = TradingRL()
try:
    if rl_agent.load('trading_rl_model.h5'):
        logger.info("Loaded existing RL model")
    else:
        logger.info("Created new RL model")
except Exception as e:
    logger.error(f"Error loading RL model: {e}")

# Initialize monitor
trade_monitor = TradingMonitor()

def get_model():
    """Get the trained LSTM model"""
    return predictor.model

def prepare_data(df):
    """Prepare data for prediction"""
    try:
        if isinstance(df, pd.DataFrame):
            # Get OHLCV columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df_cols = [col for col in df.columns if any(req.lower() in col.lower() for req in required_cols)]
            if len(df_cols) < 5:
                # If not all OHLCV columns available, use Close for all price columns
                close_col = next(col for col in df.columns if col.lower() == 'close')
                close_prices = df[close_col].values.reshape(-1, 1)
                return np.column_stack([
                    close_prices,  # Open
                    close_prices,  # High
                    close_prices,  # Low
                    close_prices,  # Close
                    np.zeros_like(close_prices)  # Volume (dummy)
                ])
            else:
                return df[df_cols].values
        else:
            close_prices = df.reshape(-1, 1)
            return np.column_stack([
                close_prices,  # Open
                close_prices,  # High
                close_prices,  # Low
                close_prices,  # Close
                np.zeros_like(close_prices)  # Volume (dummy)
            ])
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None

def predict_trend(prediction_data, analysis_data):
    """Improved trend prediction with volatility consideration"""
    try:
        # Get current price and volatility
        current_price = prediction_data['Close'].iloc[-1]
        volatility = analysis_data.get('volatility', 1.0)
        
        # Prepare data for prediction
        prepared_data = prepare_data(prediction_data)
        if prepared_data is None:
            return None
            
        # Get prediction
        predicted_price = predictor.predict(prepared_data)
        if predicted_price is None:
            return None
            
        # Get market conditions
        trend = analysis_data.get('trend', 'neutral')
        rsi = analysis_data.get('RSI', 50)
        momentum = analysis_data.get('momentum', 'weak')
        
        # Adjust RSI thresholds based on volatility
        rsi_upper = 65 - (volatility * 5)  # Lower threshold in high volatility
        rsi_lower = 35 + (volatility * 5)  # Higher threshold in high volatility
        
        # Calculate price change prediction
        price_change = (predicted_price - current_price) / current_price
        
        # Strong trend requirements
        if abs(price_change) > 0.001 * volatility:
            if price_change > 0 and trend in ['uptrend', 'strong_up'] and rsi < rsi_upper:
                if momentum == 'strong':
                    logger.info(f"Strong BUY signal: change={price_change:.4f}, RSI={rsi:.2f}")
                    return 'BUY'
            elif price_change < 0 and trend in ['downtrend', 'strong_down'] and rsi > rsi_lower:
                if momentum == 'strong':
                    logger.info(f"Strong SELL signal: change={price_change:.4f}, RSI={rsi:.2f}")
                    return 'SELL'
        
        return 'NEUTRAL'
        
    except Exception as e:
        logger.error(f"Error predicting trend: {e}")
        return None

def analyze_market(df):
    """Simple market analysis"""
    try:
        # Get market conditions
        market_conditions = predictor.analyze_market_conditions(df)
        if market_conditions is None:
            return None
            
        logger.info(f"Market Analysis - Trend: {market_conditions['trend']}, RSI: {market_conditions['RSI']:.2f}")
        return market_conditions
        
    except Exception as e:
        logger.error(f"Error in market analysis: {e}")
        return None

def detect_reversal(symbol, position_type):
    """Detect potential price reversals using multiple timeframes"""
    try:
        # Check multiple timeframes for confirmation
        timeframes = [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5]
        reversal_signals = 0
        
        for tf in timeframes:
            # Get recent candles
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, 5)
            if rates is not None:
                df = pd.DataFrame(rates)
                
                # Calculate quick momentum
                df['momentum'] = df['close'].pct_change()
                
                # Check for reversal patterns
                if position_type == mt5.POSITION_TYPE_BUY:
                    # Bearish signals for long positions
                    if df['momentum'].iloc[-1] < -0.0005:  # Immediate drop
                        reversal_signals += 1
                    if df['close'].iloc[-1] < df['open'].iloc[-1]:  # Current candle is red
                        reversal_signals += 1
                    if df['high'].iloc[-1] < df['high'].iloc[-2]:  # Lower high
                        reversal_signals += 1
                else:
                    # Bullish signals for short positions
                    if df['momentum'].iloc[-1] > 0.0005:  # Immediate rise
                        reversal_signals += 1
                    if df['close'].iloc[-1] > df['open'].iloc[-1]:  # Current candle is green
                        reversal_signals += 1
                    if df['low'].iloc[-1] > df['low'].iloc[-2]:  # Higher low
                        reversal_signals += 1
        
        # Return true if we have enough reversal signals
        return reversal_signals >= 3
        
    except Exception as e:
        logger.error(f"Error in detect_reversal: {e}")
        return False

def should_take_profit(position, analysis):
    """Improved exit strategy with dynamic profit taking"""
    try:
        # Get current profit
        profit = calculate_profit(position)
        if profit is None:
            return False

        # Get market volatility
        volatility = analysis.get('volatility', 1.0)
        
        # Dynamic thresholds based on volatility
        stop_loss = -30 * volatility  # Tighter stop loss in volatile markets
        take_profit = 50 * volatility  # Higher take profit in volatile markets
        
        # Immediate stop loss if loss exceeds threshold
        if profit < stop_loss:
            logger.info(f"Stopping loss at {profit:.2f}")
            return True
            
        # Take profit conditions
        if profit > 0:
            # Scale trailing stop based on profit level
            if profit > take_profit:
                # Use tight trailing stop for large profits
                if profit - position.get('max_profit', profit) < -5:
                    logger.info(f"Taking profit at {profit:.2f} (trailing stop hit)")
                    return True
            elif profit > 20:
                # Medium trailing stop for moderate profits
                if profit - position.get('max_profit', profit) < -3:
                    logger.info(f"Taking profit at {profit:.2f} (medium trail)")
                    return True
            else:
                # Wide trailing stop for small profits
                if profit - position.get('max_profit', profit) < -2:
                    logger.info(f"Taking profit at {profit:.2f} (wide trail)")
                    return True
        
        # Update maximum profit
        if 'max_profit' not in position or profit > position['max_profit']:
            position['max_profit'] = profit
        
        return False
        
    except Exception as e:
        logger.error(f"Error in take profit logic: {e}")
        return False

def calculate_volatility_atr(symbol, timeframe=mt5.TIMEFRAME_M5, period=14):
    """Calculate Average True Range (ATR) for volatility"""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 1)
        df = pd.DataFrame(rates)
        
        # Calculate True Range
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Calculate ATR
        atr = df['tr'].mean()
        return atr
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return None

def fetch_data(symbol, timeframe, num_bars=100):
    """Fetch and prepare data for prediction"""
    try:
        # Get 1-minute data for short-term analysis
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, num_bars)
        if rates is None:
            logger.error("Failed to get rates data")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

def calculate_volatility(data, window=20):
    """Calculate price volatility using ATR"""
    try:
        df = data.copy()
        df['high'] = df['High']
        df['low'] = df['Low']
        df['close'] = df['Close']
        
        # Calculate True Range
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['tr'].rolling(window=window).mean()
        
        return df['atr'].iloc[-1]
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        return None

def calculate_dynamic_take_profit(entry_price, atr, risk_factor=2.0):
    """Calculate dynamic take profit based on volatility"""
    try:
        return atr * risk_factor
    except Exception as e:
        logger.error(f"Error calculating dynamic take profit: {e}")
        return None

def place_trade(symbol, action, lot_size, set_sl=True):
    """Place a trade with initial stop loss"""
    try:
        symbol_info = mt5.symbol_info_tick(symbol)
        if symbol_info is None:
            logger.error("Failed to get symbol info")
            return None
            
        if action == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = symbol_info.ask
            sl = price - INITIAL_STOP_LOSS  # Initial stop loss for buy
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = symbol_info.bid
            sl = price + INITIAL_STOP_LOSS  # Initial stop loss for sell
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "magic": MAGIC_NUMBER,
            "comment": f"{action} order",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        if set_sl:
            request["sl"] = sl  # Set initial stop loss
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode}")
            return None
            
        logger.info(f"Trade placed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error placing trade: {e}")
        return None

def get_current_position():
    try:
        positions = mt5.positions_get(symbol=BTC_SYMBOL)
        if positions:
            return positions[0]
        return None
    except Exception as e:
        logger.error(f"Error getting current position: {e}")
        return None

def calculate_profit(position):
    """Calculate current profit for a position"""
    try:
        if position is None:
            return 0
        
        current_price = mt5.symbol_info_tick(BTC_SYMBOL).ask if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(BTC_SYMBOL).bid
        profit = position.volume * (current_price - position.price_open) if position.type == mt5.ORDER_TYPE_BUY else position.volume * (position.price_open - current_price)
        return profit
    except Exception as e:
        logger.error(f"Error calculating profit: {e}")
        return 0

def get_all_positions():
    """Get all open positions"""
    try:
        positions = mt5.positions_get(symbol=BTC_SYMBOL)
        if positions is None:
            return []
        return list(positions)
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return []

def check_trade_duration(position, start_time):
    """Check if a trade has exceeded its maximum duration"""
    if start_time is None:
        return False
    try:
        duration = (datetime.now() - start_time).total_seconds()
        if duration > MAX_TRADE_DURATION:
            logger.warning(f"Trade {position.ticket} exceeded maximum duration of {MAX_TRADE_DURATION} seconds. Closing trade.")
            return True
        return False
    except Exception as e:
        logger.error(f"Error checking trade duration: {e}")
        return False

def update_trailing_stop(position):
    """Simple trailing stop implementation"""
    try:
        # Get current profit
        profit = calculate_profit(position)
        if profit is None or profit < MIN_PROFIT_TO_TRAIL:
            return False

        # Get current price
        symbol_info = mt5.symbol_info_tick(position.symbol)
        if symbol_info is None:
            return False

        current_price = symbol_info.bid if position.type == mt5.POSITION_TYPE_BUY else symbol_info.ask
        
        # Calculate trailing stop distance (0.2% of current price)
        trail_distance = current_price * 0.002
        
        # Update stop loss only if we have enough profit
        if position.type == mt5.POSITION_TYPE_BUY:
            new_sl = current_price - trail_distance
            if new_sl > position.sl and new_sl > position.price_open:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": position.ticket,
                    "symbol": position.symbol,
                    "sl": new_sl,
                    "tp": position.tp
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Failed to update trailing stop: {result.comment}")
                    return False
                return True
                
        elif position.type == mt5.POSITION_TYPE_SELL:
            new_sl = current_price + trail_distance
            if new_sl < position.sl and new_sl < position.price_open:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": position.ticket,
                    "symbol": position.symbol,
                    "sl": new_sl,
                    "tp": position.tp
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Failed to update trailing stop: {result.comment}")
                    return False
                return True
                
        return False
        
    except Exception as e:
        logger.error(f"Error updating trailing stop: {e}")
        return False

def close_position(position):
    """Close a trading position"""
    try:
        # Get position details before closing
        entry_price = position.price_open
        position_type = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'
        
        # Close the position
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(close_request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position: {result.comment}")
            return False
            
        # Get exit price and calculate profit/loss
        exit_price = result.price
        profit_loss = position.profit
        
        try:
            # Get current state and action confidence from RL agent
            market_data = fetch_data(BTC_SYMBOL, mt5.TIMEFRAME_M5)
            if market_data is not None:
                current_state = rl_agent.get_state(market_data)
                action_confidence = max(rl_agent.model.predict(np.array([current_state]))[0])
            else:
                logger.warning("Could not fetch market data for metrics")
                current_state = np.zeros(rl_agent.state_size)
                action_confidence = 0
                
            # Calculate reward
            reward = rl_agent.calculate_reward(position_type, None, exit_price, entry_price)
            
            # Record trade in monitor with safe defaults
            trade_monitor.save_trade(
                trade_type=position_type,
                entry_price=entry_price,
                exit_price=exit_price,
                profit_loss=profit_loss,
                reward=reward,
                action_confidence=action_confidence,
                state_values=current_state.tolist() if current_state is not None else []
            )
            
            # Generate and log performance summary every 10 trades
            if len(trade_monitor.trades_data['trades']) % 10 == 0:
                try:
                    summary = trade_monitor.get_performance_summary()
                    logger.info(f"Performance Summary: {summary}")
                    trade_monitor.plot_learning_progress()
                    logger.info("Learning progress plot updated")
                except Exception as e:
                    logger.error(f"Error generating performance summary: {e}")
                    
        except Exception as e:
            logger.error(f"Error recording trade metrics: {e}")
            # Continue execution even if metrics recording fails
        
        logger.info(f"Position closed successfully at {exit_price}")
        return True
        
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        return False

def analyze_trend_strength(df):
    """Analyze short-term trend strength"""
    try:
        # Calculate momentum indicators
        df['ema9'] = df['close'].ewm(span=9).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        
        # Calculate trend strength based on EMAs
        ema_diff = (df['ema9'] - df['ema21']) / df['ema21'] * 100
        
        # Get latest strength value
        current_strength = ema_diff.iloc[-1]
        
        # Convert to float value between 0 and 1
        normalized_strength = min(max(abs(current_strength) / 2, 0), 1)
        
        return normalized_strength
        
    except Exception as e:
        logger.error(f"Error analyzing trend strength: {e}")
        return 0.5  # Return moderate strength on error

def calculate_lot_size(trend_strength, balance, risk_percent=1.0, trade_score=0):
    """Calculate aggressive lot size based on signal strength"""
    try:
        # Ensure trend_strength is a float
        trend_strength = float(trend_strength) if trend_strength is not None else 0.5
        
        # Get account balance if not provided
        if balance is None:
            account_info = mt5.account_info()
            if account_info is None:
                return 1.0  # Minimum lot size as fallback
            balance = float(account_info.balance)

        # Calculate signal strength (0-1)
        signal_strength = min(1.0, trade_score / 100.0) if trade_score else 0.5
        
        # Combine trend and signal strength
        combined_strength = (trend_strength + signal_strength) / 2
        
        # Calculate base lot size (1.0 to 20.0)
        lot_size = 1.0 + (combined_strength * 19.0)  # Maps 0-1 to 1-20
        
        # Adjust based on recent volatility
        volatility = calculate_volatility_atr(BTC_SYMBOL)
        if volatility:
            volatility_factor = min(1.5, max(0.5, volatility))
            lot_size *= volatility_factor
        
        # Round to nearest 0.5
        lot_size = round(lot_size * 2) / 2
        
        # Ensure limits
        lot_size = max(1.0, min(20.0, lot_size))
        
        logger.info(f"Calculated lot size: {lot_size} (Strength: {combined_strength:.2f})")
        return lot_size
        
    except Exception as e:
        logger.error(f"Error calculating lot size: {e}")
        return 1.0  # Return minimum lot size on error

def predict_short_term_trend(data):
    """Predict very short-term trend (1-minute)"""
    try:
        df = data.copy()
        
        # Calculate super short-term indicators
        df['ema3'] = df['close'].ewm(span=3).mean()
        df['ema5'] = df['close'].ewm(span=5).mean()
        df['ema8'] = df['close'].ewm(span=8).mean()
        
        # Calculate price changes and momentum
        df['momentum'] = df['close'].pct_change()
        df['volume_change'] = df['tick_volume'].pct_change()
        
        # Price changes for different timeframes
        df['change_1min'] = df['close'].pct_change(1)
        df['change_2min'] = df['close'].pct_change(2)
        df['change_3min'] = df['close'].pct_change(3)
        
        # Get latest values
        current_price = df['close'].iloc[-1]
        ema3 = df['ema3'].iloc[-1]
        ema5 = df['ema5'].iloc[-1]
        ema8 = df['ema8'].iloc[-1]
        
        # Recent price changes
        changes = {
            '1min': df['change_1min'].iloc[-1] * 100,
            '2min': df['change_2min'].iloc[-1] * 100,
            '3min': df['change_3min'].iloc[-1] * 100
        }
        
        # Detect immediate trend changes
        price_acceleration = changes['1min'] - changes['2min']
        momentum_change = df['momentum'].iloc[-1] - df['momentum'].iloc[-2]
        volume_surge = df['volume_change'].iloc[-1] > 0.1  # Even lower volume threshold
        
        # Calculate trend strength (-3 to +3)
        trend_strength = sum(1 for x in changes.values() if x > 0) - sum(1 for x in changes.values() if x < 0)
        
        logger.info(f"1min Change: {changes['1min']:.3f}%, Acceleration: {price_acceleration:.3f}%, Volume: {df['volume_change'].iloc[-1]:.1f}%, Trend Strength: {trend_strength}")
        
        # Base signal strength calculation (scaled up significantly)
        base_signal = abs(changes['1min']) * 20  # Scale up by 20x instead of 3x
        
        # Add acceleration component
        base_signal += abs(price_acceleration) * 15
        
        # Add trend strength component
        base_signal += abs(trend_strength) * 5
        
        # Quick sell signals - more sensitive
        if ((changes['1min'] < -0.005) or  # Small threshold
            (ema3 < ema5 and price_acceleration < -0.002) or  # Very sensitive to acceleration
            (trend_strength < 0)):  # Any negative trend strength
            if volume_surge:
                base_signal *= 1.2  # 20% volume boost
            return 'SELL', base_signal
            
        # Quick buy signals - more sensitive
        elif ((changes['1min'] > 0.005) or  # Small threshold
              (ema3 > ema5 and price_acceleration > 0.002) or  # Very sensitive to acceleration
              (trend_strength > 0)):  # Any positive trend strength
            if volume_surge:
                base_signal *= 1.2  # 20% volume boost
            return 'BUY', base_signal
            
        return 'NEUTRAL', 0
        
    except Exception as e:
        logger.error(f"Error in short-term prediction: {e}")
        return 'NEUTRAL', 0

def calculate_recovery_lot_size(base_lot_size, initial_position):
    """Calculate appropriate lot size for recovery trade"""
    try:
        # Get symbol info
        symbol_info = mt5.symbol_info(BTC_SYMBOL)
        if symbol_info is None:
            return base_lot_size
            
        # Get maximum allowed lot size
        max_lot = symbol_info.volume_max
        min_lot = symbol_info.volume_min
        
        # Calculate required lot size for recovery
        loss_amount = abs(initial_position.profit)
        current_price = mt5.symbol_info_tick(BTC_SYMBOL).ask
        point_value = loss_amount / (initial_position.volume * 100)  # Approximate points needed
        
        # Calculate recovery lot size (1.5x instead of 2x to be safer)
        recovery_lot_size = base_lot_size * 1.5
        
        # Ensure lot size is within limits
        recovery_lot_size = min(recovery_lot_size, max_lot)
        recovery_lot_size = max(recovery_lot_size, min_lot)
        
        # Round to allowed lot step
        lot_step = symbol_info.volume_step
        recovery_lot_size = round(recovery_lot_size / lot_step) * lot_step
        
        return recovery_lot_size
        
    except Exception as e:
        logger.error(f"Error calculating recovery lot size: {e}")
        return base_lot_size

def run_bot():
    """Main bot loop with minute-by-minute monitoring"""
    try:
        CHECK_INTERVAL = 5  # Check every 5 seconds
        last_state = None
        last_action = None
        trades_this_session = 0
        last_trade_time = None
        training_interval = 300  # Train every 5 minutes
        last_training_time = time.time()
        recovery_trade_placed = False
        initial_position = None
        failed_recovery_attempts = 0
        max_recovery_attempts = 3
        last_trend_reversal = None
        reversal_cooldown = 60  # Wait 60 seconds after reversal before new trades
        model_error_count = 0
        max_model_errors = 3
        
        while True:
            try:
                current_time = time.time()
                
                # Get 1-minute market data
                data = fetch_data(BTC_SYMBOL, mt5.TIMEFRAME_M1, 20)
                if data is None:
                    time.sleep(CHECK_INTERVAL)
                    continue
                
                # Get current state
                try:
                    current_state = rl_agent.get_state(data)
                except Exception as e:
                    logger.error(f"Error getting state: {e}")
                    model_error_count += 1
                    if model_error_count >= max_model_errors:
                        logger.info("Resetting RL model due to repeated errors")
                        rl_agent.reset_model()
                        model_error_count = 0
                    time.sleep(CHECK_INTERVAL)
                    continue
                
                # Get current positions
                current_positions = get_all_positions()
                if current_positions is None:
                    time.sleep(CHECK_INTERVAL)
                    continue
                
                # Predict short-term trend
                trend_prediction, signal_strength = predict_short_term_trend(data)
                
                # Get market analysis
                market_analysis = analyze_market(data)
                if market_analysis is None:
                    time.sleep(CHECK_INTERVAL)
                    continue
                    
                # Get trend confirmation from multiple timeframes
                m1_trend = market_analysis.get('trend', 'neutral')
                m5_data = fetch_data(BTC_SYMBOL, mt5.TIMEFRAME_M5, 20)
                m5_analysis = analyze_market(m5_data) if m5_data is not None else None
                m5_trend = m5_analysis.get('trend', 'neutral') if m5_analysis else 'neutral'
                
                # Strong trend confirmation when both timeframes agree
                strong_uptrend = m1_trend == 'uptrend' and m5_trend == 'uptrend'
                strong_downtrend = m1_trend == 'downtrend' and m5_trend == 'downtrend'
                
                # Get RSI and other indicators
                rsi = market_analysis.get('RSI', 50)
                volume_change = market_analysis.get('volume_change', 0)
                acceleration = market_analysis.get('acceleration', 0)
                
                # Check for extreme RSI conditions
                is_overbought = rsi > 70
                is_oversold = rsi < 30
                
                # Update existing positions with monitoring (no automatic stop loss)
                for position in current_positions:
                    current_profit = calculate_profit(position)
                    if current_profit is None:
                        continue
                        
                    # Take profit on trend reversal with confirmation
                    trend_reversed = False
                    strong_reversal = False
                    
                    if position.type == mt5.POSITION_TYPE_BUY:
                        trend_reversed = trend_prediction == 'SELL' and strong_downtrend
                        strong_reversal = trend_reversed and rsi > 75
                    else:
                        trend_reversed = trend_prediction == 'BUY' and strong_uptrend
                        strong_reversal = trend_reversed and rsi < 25
                    
                    # Dynamic profit taking based on trend strength and profit amount
                    min_profit_target = 100  # Minimum profit to consider taking
                    
                    # Calculate trend strength score
                    trend_strength = market_analysis.get('trend_strength', 0)
                    volume_change = abs(market_analysis.get('volume_change', 0))
                    momentum = abs(market_analysis.get('acceleration', 0))
                    
                    # If profit is good and trend is weakening, take profit
                    should_take_profit = False
                    
                    if current_profit >= min_profit_target:
                        if strong_reversal:
                            # Take profit immediately on strong reversal
                            should_take_profit = True
                            logger.info(f"Taking profit due to strong trend reversal: {current_profit:.2f}")
                        elif trend_reversed and current_profit > 200:
                            # Take larger profits on trend reversal
                            should_take_profit = True
                            logger.info(f"Taking larger profit on trend reversal: {current_profit:.2f}")
                        elif current_profit > 300:
                            # Take profit at significant gain regardless of trend
                            should_take_profit = True
                            logger.info(f"Taking profit at significant gain: {current_profit:.2f}")
                        elif trend_strength < -2 and current_profit > 150:
                            # Take profit when trend is weakening
                            should_take_profit = True
                            logger.info(f"Taking profit on weakening trend: {current_profit:.2f}")
                        elif volume_change > 2 and momentum > 0.1 and current_profit > 250:
                            # Take profit on high volatility
                            should_take_profit = True
                            logger.info(f"Taking profit on high volatility: {current_profit:.2f}")
                    
                    if should_take_profit:
                        close_position(position)
                        recovery_trade_placed = False
                        initial_position = None
                        failed_recovery_attempts = 0
                        last_trend_reversal = current_time
                        continue
                    
                    # If in loss, consider recovery trade
                    if current_profit < -10 and not recovery_trade_placed and failed_recovery_attempts < max_recovery_attempts:
                        initial_position = position
                        recovery_trade_placed = True
                        logger.info(f"Position in loss: {current_profit:.2f}, enabling recovery mode")
                
                # Check if we're in reversal cooldown period
                if last_trend_reversal and current_time - last_trend_reversal < reversal_cooldown:
                    logger.info("In reversal cooldown period, skipping new trades")
                    time.sleep(CHECK_INTERVAL)
                    continue
                
                # Place new trades if allowed
                if len(current_positions) == 0 or (len(current_positions) == 1 and recovery_trade_placed and initial_position is not None):
                    # Get action from RL agent
                    action = rl_agent.act(current_state)
                    
                    # Map RL action to trade decision, but respect strong trends and RSI
                    if action == 1 and not strong_downtrend and not is_overbought:
                        trend_prediction = 'BUY'
                        trade_score = signal_strength * 1.2
                        if strong_uptrend:
                            trade_score *= 1.5  # Boost score in strong uptrend
                    elif action == 2 and not strong_uptrend and not is_oversold:
                        trend_prediction = 'SELL'
                        trade_score = signal_strength * 1.2
                        if strong_downtrend:
                            trade_score *= 1.5  # Boost score in strong downtrend
                    else:
                        trend_prediction = 'NEUTRAL'
                        trade_score = 0
                    
                    # Volume and acceleration adjustments
                    if abs(volume_change) > 1:
                        trade_score *= 1.2
                    if abs(acceleration) < 0.05:
                        trade_score *= 0.8
                    
                    # RSI score adjustments
                    if trend_prediction == 'BUY':
                        if rsi > 65:  # Getting close to overbought
                            trade_score *= 0.5
                        elif rsi < 40:  # Good buying opportunity
                            trade_score *= 1.3
                    else:  # SELL
                        if rsi < 35:  # Getting close to oversold
                            trade_score *= 0.5
                        elif rsi > 60:  # Good selling opportunity
                            trade_score *= 1.3
                    
                    logger.info(f"RL Action: {action}, Prediction: {trend_prediction}, RSI: {rsi:.1f}, Score: {trade_score:.1f}")
                    logger.info(f"M1 Trend: {m1_trend}, M5 Trend: {m5_trend}")
                    
                    # Place trade if score is high enough and market conditions are right
                    if trade_score >= 5:
                        # Additional safety checks
                        if (trend_prediction == 'BUY' and rsi > 75) or (trend_prediction == 'SELL' and rsi < 25):
                            logger.info(f"Skipping trade due to extreme RSI: {rsi:.1f}")
                            continue
                            
                        base_lot_size = calculate_lot_size(0.8, None, trade_score=trade_score)
                        
                        if recovery_trade_placed and initial_position is not None:
                            lot_size = calculate_recovery_lot_size(base_lot_size, initial_position)
                            logger.info(f"Recovery trade with calculated lot size: {lot_size}")
                        else:
                            lot_size = base_lot_size
                        
                        trade_placed = False
                        if trend_prediction == 'BUY' and (strong_uptrend or (m1_trend != 'downtrend' and m5_trend != 'downtrend')):
                            logger.info(f"Opening BUY position (Score: {trade_score:.1f}, Lots: {lot_size})")
                            trade_placed = place_trade(BTC_SYMBOL, 'BUY', lot_size, set_sl=False)
                        elif trend_prediction == 'SELL' and (strong_downtrend or (m1_trend != 'uptrend' and m5_trend != 'uptrend')):
                            logger.info(f"Opening SELL position (Score: {trade_score:.1f}, Lots: {lot_size})")
                            trade_placed = place_trade(BTC_SYMBOL, 'SELL', lot_size, set_sl=False)
                            
                        if trade_placed:
                            last_state = current_state
                            last_action = action
                            last_trade_time = current_time
                        elif recovery_trade_placed:
                            failed_recovery_attempts += 1
                            if failed_recovery_attempts >= max_recovery_attempts:
                                logger.info("Max recovery attempts reached, disabling recovery mode")
                                recovery_trade_placed = False
                                initial_position = None
                
                # Train the model periodically
                if current_time - last_training_time >= training_interval:
                    try:
                        logger.info("Training RL model...")
                        rl_agent.replay(32)
                        rl_agent.update_target_model()
                        last_training_time = current_time
                    except Exception as e:
                        logger.error(f"Error training model: {e}")
                        # Reset model if training fails
                        rl_agent.reset_model()
                
                time.sleep(CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(CHECK_INTERVAL)
                
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        cleanup()
    except Exception as e:
        logger.error(f"Fatal error in bot loop: {e}")
        cleanup()

# Save model on exit
def cleanup():
    try:
        rl_agent.save('trading_rl_model.h5')
        logger.info("Saved RL model on exit")
    except Exception as e:
        logger.error(f"Error saving RL model: {e}")
    finally:
        mt5.shutdown()

# Main trading loop
try:
    run_bot()
except KeyboardInterrupt:
    logger.info("Bot stopped by user")
    cleanup()
except Exception as e:
    logger.error(f"Fatal error: {e}")
    cleanup()
