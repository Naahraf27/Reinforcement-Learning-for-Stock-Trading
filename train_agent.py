# Import necessary modules
import pandas as pd
# from stable_baselines3 import PPO
from stable_baselines3 import DQN
from trading_environment import TradingEnv  # Assuming this is what you named your custom environment's Python file

# Load training data for Adidas/Nike/Puma
df = pd.read_csv('/Users/farhaanfayaz/Desktop/RL Project/Data/NKE_train.csv')

# Initialize custom trading environment
env = TradingEnv(df=df)

# Initialize PPO model
# model = PPO("MlpPolicy", env, learning_rate=0.0005, verbose=1)

# Initialize DQN model
model = DQN("MlpPolicy", env, learning_rate=0.0003, verbose=1)

# Train the model
# Originally = 20000
model.learn(total_timesteps=100000)

# Save the model
model.save("dqn_trading_model_nke")