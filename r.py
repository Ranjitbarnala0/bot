import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
import logging
from datetime import datetime
import MetaTrader5 as mt5
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingRL:
    def __init__(self):
        self.state_size = 10
        self.action_size = 3  # hold, buy, sell
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # start with full exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model = self._build_model()
        self.target_model = self._build_model()  # Target network for stable learning
        self.update_target_every = 5  # Update target model every N episodes
        self.episode_count = 0
        self.min_reward_threshold = -0.5  # Minimum reward before adapting
        self.success_threshold = 0.5  # Reward threshold for successful trades
        
    def _build_model(self):
        """Neural Net for Deep-Q learning Model with advanced architecture"""
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='huber_loss', optimizer=Adam(learning_rate=self.learning_rate))
        return model
        
    def update_target_model(self):
        """Update target model with weights from main model"""
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience with prioritized memory"""
        # Calculate TD error for prioritization
        if state is not None and next_state is not None:
            current_q = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            next_q = self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
            td_error = abs(reward + self.gamma * np.max(next_q) - current_q[action])
            
            # Store experience with priority
            self.memory.append((state, action, reward, next_state, done, td_error))
            
            # Keep memory size under control
            if len(self.memory) > 10000:
                # Remove experiences with lowest TD error
                self.memory.sort(key=lambda x: x[5])  # Sort by TD error
                self.memory = self.memory[-10000:]  # Keep top 10000
        
    def act(self, state):
        """Choose action using epsilon-greedy with adaptive exploration"""
        try:
            if state is None:
                return 0  # Hold if state is invalid
                
            state = np.reshape(state, [1, self.state_size])
            
            # Adaptive exploration based on recent performance
            if len(self.memory) > 100:
                recent_rewards = [mem[2] for mem in self.memory[-100:]]
                avg_reward = np.mean(recent_rewards)
                
                # Increase exploration if performing poorly
                if avg_reward < self.min_reward_threshold:
                    self.epsilon = min(1.0, self.epsilon * 1.1)
                # Decrease exploration if performing well
                elif avg_reward > self.success_threshold:
                    self.epsilon = max(self.epsilon_min, self.epsilon * 0.9)
            
            if np.random.rand() <= self.epsilon:
                return np.random.randint(self.action_size)
                
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0])
            
        except Exception as e:
            logger.error(f"Error in act: {e}")
            return 0
            
    def replay(self, batch_size=32):
        """Replay experiences to train the model"""
        try:
            if len(self.memory) < batch_size:
                return
            
            minibatch = random.sample(self.memory, batch_size)
            states = np.array([i[0] for i in minibatch])
            actions = np.array([i[1] for i in minibatch])
            rewards = np.array([i[2] for i in minibatch])
            next_states = np.array([i[3] for i in minibatch])
            dones = np.array([i[4] for i in minibatch])

            states = np.squeeze(states)
            next_states = np.squeeze(next_states)

            targets = self.model.predict(states, verbose=0)
            next_state_values = self.model.predict(next_states, verbose=0)

            for i in range(batch_size):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_state_values[i])

            self.model.fit(states, targets, epochs=1, verbose=0)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        except Exception as e:
            logger.error(f"Error in replay: {e}")
            
    def calculate_reward(self, action, position=None, next_price=None, current_price=None):
        """Calculate reward for action taken with enhanced market awareness"""
        try:
            if position is None:  # No position exists
                if action == 0:  # Hold
                    return 0.05  # Small reward for being cautious
                return -0.05  # Small penalty for trying to trade without position info
                    
            # Calculate base profit/loss
            if position.type == mt5.POSITION_TYPE_BUY:
                profit = (next_price - current_price) / current_price
            else:
                profit = (current_price - next_price) / current_price
                    
            # Scale up the reward for better learning
            reward = profit * 100
            
            # Get market data for context
            data = mt5.copy_rates_from_pos(position.symbol, mt5.TIMEFRAME_M1, 0, 20)
            if data is not None:
                df = pd.DataFrame(data)
                
                # Calculate market volatility
                volatility = df['high'].max() - df['low'].min()
                avg_volatility = (df['high'] - df['low']).mean()
                volatility_ratio = volatility / avg_volatility if avg_volatility > 0 else 1
                
                # Calculate trend strength
                ema5 = df['close'].ewm(span=5).mean().iloc[-1]
                ema20 = df['close'].ewm(span=20).mean().iloc[-1]
                trend_strength = abs(ema5 - ema20) / ema20
                
                # Calculate volume strength
                volume_ratio = df['tick_volume'].iloc[-1] / df['tick_volume'].mean()
                
                # Adjust reward based on market conditions
                if action > 0:  # Buy or Sell actions
                    # Higher reward for profitable trades in high volatility
                    if profit > 0:
                        reward *= (1 + volatility_ratio * 0.2)  # Up to 20% boost
                        
                        # Additional reward for trading with the trend
                        if (position.type == mt5.POSITION_TYPE_BUY and ema5 > ema20) or \
                           (position.type == mt5.POSITION_TYPE_SELL and ema5 < ema20):
                            reward *= (1 + trend_strength)
                            
                        # Volume confirmation bonus
                        if volume_ratio > 1.2:  # 20% above average volume
                            reward *= 1.1  # 10% bonus
                    else:
                        # Smaller penalty if loss was during high volatility
                        reward *= (1 - volatility_ratio * 0.1)  # Up to 10% reduction
                        
                        # Bigger penalty for trading against the trend
                        if (position.type == mt5.POSITION_TYPE_BUY and ema5 < ema20) or \
                           (position.type == mt5.POSITION_TYPE_SELL and ema5 > ema20):
                            reward *= (1 - trend_strength)
                else:  # Hold action
                    if abs(profit) < volatility_ratio * 0.001:  # Good hold during sideways
                        reward = 0.1 * (1 + trend_strength)  # Small positive reward
                    elif profit < -0.001:  # Bad hold during loss
                        reward = -0.1 * (1 + trend_strength)  # Small negative reward
                    
            # Cap the reward to prevent extreme values
            reward = max(min(reward, 10), -10)
            
            logger.debug(f"Reward: {reward:.4f} for action {action} "
                        f"(profit: {profit:.4f}, volatility: {volatility_ratio:.2f}, "
                        f"trend: {trend_strength:.2f})")
            
            return reward
                
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0

    def get_state(self, df):
        """Create state from market data"""
        try:
            if df is None or len(df) < 2:
                logger.warning("Invalid dataframe provided to get_state")
                return np.zeros(self.state_size)  # Return zero state as fallback
                
            # Get last row of data
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Safely get Close prices with fallback to any available price column
            close_col = None
            for col in ['Close', 'close', 'CLOSE']:
                if col in df.columns:
                    close_col = col
                    break
            if close_col is None:
                # Fallback to first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    close_col = numeric_cols[0]
                else:
                    logger.error("No suitable price column found in dataframe")
                    return np.zeros(self.state_size)
            
            # Calculate features with safe fallbacks
            try:
                price_change = (current[close_col] - prev[close_col]) / prev[close_col]
            except:
                price_change = 0
                
            # Get technical indicators with safe defaults
            rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
            ema9 = df['EMA9'].iloc[-1] if 'EMA9' in df.columns else current[close_col]
            ema21 = df['EMA21'].iloc[-1] if 'EMA21' in df.columns else current[close_col]
            ema50 = df['EMA50'].iloc[-1] if 'EMA50' in df.columns else current[close_col]
            
            # Calculate trend strengths
            short_trend = (ema9 / ema21 - 1) if ema9 != 0 and ema21 != 0 else 0
            long_trend = (ema21 / ema50 - 1) if ema21 != 0 and ema50 != 0 else 0
            
            # Safe momentum and volume calculations
            momentum = df['Momentum'].iloc[-1] if 'Momentum' in df.columns else price_change
            volume = current['Volume'] if 'Volume' in df.columns else 1
            avg_volume = df['Volume'].mean() if 'Volume' in df.columns else 1
            
            # Risk indicators
            is_overbought = 1 if rsi > 70 else 0
            is_oversold = 1 if rsi < 30 else 0
            
            # Create state vector with enhanced risk awareness
            state = np.array([
                price_change * 100,  # Scale up for better learning
                short_trend * 100,
                long_trend * 100,
                rsi / 100,  # Normalize RSI
                momentum * 100,
                volume / avg_volume if avg_volume != 0 else 1,
                is_overbought,
                is_oversold,
                (current[close_col] / ema50 - 1) * 100 if ema50 != 0 else 0,
                1 if volume > avg_volume else 0
            ])
            
            return state
            
        except Exception as e:
            logger.error(f"Error in get_state: {e}")
            return np.zeros(self.state_size)  # Return zero state as fallback

    def adaptive_act(self, state, market_data):
        """Adaptive action selection based on market conditions"""
        try:
            # Get market volatility from ATR
            atr = market_data['ATR'].iloc[-1] if 'ATR' in market_data else 0
            avg_atr = market_data['ATR'].rolling(20).mean().iloc[-1] if 'ATR' in market_data else 0
            
            # Get price data
            current_price = market_data['Close'].iloc[-1]
            prev_price = market_data['Close'].iloc[-2]
            price_change = abs((current_price - prev_price) / prev_price)
            
            # Define volatility thresholds
            LOW_VOLATILITY = 0.0001  # 0.01% price change
            LOW_ATR_RATIO = 0.8  # ATR below 80% of average indicates low volatility
            
            # Check if we're in a low volatility environment
            is_low_volatility = (price_change < LOW_VOLATILITY or 
                               (avg_atr != 0 and atr/avg_atr < LOW_ATR_RATIO))
            
            if is_low_volatility:
                # Scalping Mode
                return self._scalping_strategy(state, market_data)
            else:
                # Normal Trading Mode
                return self._normal_strategy(state)
                
        except Exception as e:
            logger.error(f"Error in adaptive_act: {e}")
            return 0  # Default to hold
            
    def _scalping_strategy(self, state, market_data):
        """Scalping strategy for low volatility conditions"""
        try:
            rsi = state[1] * 100  # Denormalize RSI
            price_change = state[0] / 100  # Denormalize price change
            
            # Get micro-trend indicators
            ema9 = market_data['EMA9'].iloc[-1]
            ema21 = market_data['EMA21'].iloc[-1]
            micro_trend = ema9 / ema21 - 1
            
            # Tighter RSI bounds for scalping
            SCALP_OVERBOUGHT = 65
            SCALP_OVERSOLD = 35
            
            # Smaller price movement thresholds
            MICRO_TREND_THRESHOLD = 0.0001  # 0.01%
            
            if rsi > SCALP_OVERBOUGHT and micro_trend < -MICRO_TREND_THRESHOLD:
                return 2  # Sell
            elif rsi < SCALP_OVERSOLD and micro_trend > MICRO_TREND_THRESHOLD:
                return 1  # Buy
            else:
                return 0  # Hold
                
        except Exception as e:
            logger.error(f"Error in scalping strategy: {e}")
            return 0
            
    def _normal_strategy(self, state):
        """Normal trading strategy with existing risk management"""
        try:
            if np.random.rand() <= self.epsilon:
                # Risk-aware random action selection
                rsi = state[1] * 100
                is_overbought = state[5]
                is_oversold = state[6]
                
                if is_overbought:
                    return np.random.choice([0, 2], p=[0.4, 0.6])
                elif is_oversold:
                    return np.random.choice([0, 1], p=[0.4, 0.6])
                else:
                    return np.random.randint(self.action_size)
            
            # Get Q-values and apply risk management
            act_values = self.model.predict(state.reshape(1, -1), verbose=0)
            action = np.argmax(act_values[0])
            
            # Apply risk management rules
            rsi = state[1] * 100
            is_overbought = state[5]
            is_oversold = state[6]
            
            if is_overbought and action == 1:
                action = 0
            elif is_oversold and action == 2:
                action = 0
                
            return action
            
        except Exception as e:
            logger.error(f"Error in normal strategy: {e}")
            return 0
            
    def reset_model(self):
        """Reset the model to its initial state"""
        self.memory.clear()
        self.epsilon = 1.0
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def save(self, filepath='trading_rl_model.h5'):
        """Save the model and memory"""
        try:
            self.model.save(filepath)
            with open(filepath + '.memory', 'wb') as f:
                pickle.dump({
                    'memory': self.memory,
                    'epsilon': self.epsilon
                }, f)
            logger.info(f"Model and memory saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            
    def load(self, filepath='trading_rl_model.h5'):
        """Load the model and memory"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            with open(filepath + '.memory', 'rb') as f:
                data = pickle.load(f)
                self.memory = data['memory']
                self.epsilon = data['epsilon']
            logger.info(f"Model and memory loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
