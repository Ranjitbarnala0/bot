import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

class TradingMonitor:
    def __init__(self):
        self.performance_file = 'trading_performance.json'
        self.trades_data = self._load_performance_data()
        
    def _load_performance_data(self):
        if os.path.exists(self.performance_file):
            with open(self.performance_file, 'r') as f:
                return json.load(f)
        return {
            'trades': [],
            'learning_metrics': {
                'win_rate': [],
                'avg_reward': [],
                'exploration_rate': []
            }
        }
    
    def save_trade(self, trade_type, entry_price, exit_price, profit_loss, 
                  reward, action_confidence, state_values):
        """Record a completed trade with its details"""
        try:
            # Ensure all numeric values are valid
            trade_data = {
                'timestamp': datetime.now().isoformat(),
                'type': str(trade_type),
                'entry_price': float(entry_price) if entry_price is not None else 0,
                'exit_price': float(exit_price) if exit_price is not None else 0,
                'profit_loss': float(profit_loss) if profit_loss is not None else 0,
                'reward': float(reward) if reward is not None else 0,
                'action_confidence': float(action_confidence) if action_confidence is not None else 0,
                'state_values': state_values if isinstance(state_values, list) else []
            }
            
            # Initialize trades list if it doesn't exist
            self.trades_data.setdefault('trades', [])
            self.trades_data['trades'].append(trade_data)
            
            self._update_learning_metrics()
            self._save_performance_data()
            
        except Exception as e:
            print(f"Error saving trade: {e}")
    
    def _update_learning_metrics(self):
        """Update learning progress metrics"""
        try:
            recent_trades = self.trades_data['trades'][-50:]  # Look at last 50 trades
            if not recent_trades:
                return
                
            # Calculate win rate (safely handle None values)
            wins = sum(1 for trade in recent_trades if trade.get('profit_loss', 0) > 0)
            win_rate = wins / len(recent_trades)
            
            # Calculate average reward (safely handle None/invalid values)
            valid_rewards = [trade.get('reward', 0) for trade in recent_trades if isinstance(trade.get('reward'), (int, float))]
            avg_reward = sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0
            
            # Calculate average action confidence (safely handle None/invalid values)
            valid_confidences = [trade.get('action_confidence', 0) for trade in recent_trades 
                               if isinstance(trade.get('action_confidence'), (int, float))]
            avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0
            
            # Update metrics
            self.trades_data.setdefault('learning_metrics', {})
            for key in ['win_rate', 'avg_reward', 'exploration_rate']:
                self.trades_data['learning_metrics'].setdefault(key, [])
                
            self.trades_data['learning_metrics']['win_rate'].append(float(win_rate))
            self.trades_data['learning_metrics']['avg_reward'].append(float(avg_reward))
            self.trades_data['learning_metrics']['exploration_rate'].append(float(1 - avg_confidence))
            
        except Exception as e:
            print(f"Error updating learning metrics: {e}")
            # Initialize metrics if they don't exist
            self.trades_data.setdefault('learning_metrics', {
                'win_rate': [],
                'avg_reward': [],
                'exploration_rate': []
            })
    
    def _save_performance_data(self):
        """Save performance data to file"""
        with open(self.performance_file, 'w') as f:
            json.dump(self.trades_data, f, indent=2)
    
    def plot_learning_progress(self):
        """Generate plots showing the bot's learning progress"""
        metrics = self.trades_data['learning_metrics']
        
        plt.figure(figsize=(15, 10))
        
        # Win Rate
        plt.subplot(3, 1, 1)
        plt.plot(metrics['win_rate'], label='Win Rate', color='green')
        plt.title('Trading Performance Over Time')
        plt.ylabel('Win Rate')
        plt.grid(True)
        plt.legend()
        
        # Average Reward
        plt.subplot(3, 1, 2)
        plt.plot(metrics['avg_reward'], label='Avg Reward', color='blue')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.legend()
        
        # Exploration Rate
        plt.subplot(3, 1, 3)
        plt.plot(metrics['exploration_rate'], label='Exploration Rate', color='orange')
        plt.xlabel('Updates')
        plt.ylabel('Exploration Rate')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('learning_progress.png')
        plt.close()
    
    def get_performance_summary(self):
        """Generate a summary of the bot's performance"""
        try:
            if not self.trades_data.get('trades'):
                return "No trades recorded yet."
                
            recent_trades = self.trades_data['trades'][-50:]
            all_trades = self.trades_data['trades']
            
            # Safely calculate metrics
            total_trades = len(all_trades)
            recent_wins = sum(1 for t in recent_trades if t.get('profit_loss', 0) > 0)
            recent_win_rate = recent_wins / len(recent_trades) if recent_trades else 0
            total_profit = sum(t.get('profit_loss', 0) for t in all_trades)
            
            valid_rewards = [t.get('reward', 0) for t in recent_trades if isinstance(t.get('reward'), (int, float))]
            avg_reward = sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0
            
            valid_confidences = [t.get('action_confidence', 0) for t in recent_trades 
                               if isinstance(t.get('action_confidence'), (int, float))]
            confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0
            
            summary = {
                'total_trades': total_trades,
                'recent_win_rate': float(recent_win_rate),
                'total_profit': float(total_profit),
                'avg_reward': float(avg_reward),
                'confidence': float(confidence)
            }
            
            return summary
            
        except Exception as e:
            print(f"Error generating performance summary: {e}")
            return "Error generating performance summary"
