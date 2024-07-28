import pandas as pd
# from stable_baselines3 import PPO
from stable_baselines3 import DQN
from trading_environment import TradingEnv

# Load test data for Adidas/Nike/Puma
df_test = pd.read_csv('/Users/farhaanfayaz/Desktop/RL Project/Data/NKE_test.csv')

# Initialize custom trading environment with test data
env_test = TradingEnv(df=df_test)

# Load the trained model for Adidas/Nike/Puma
# model_ads = PPO.load("ppo_trading_model_nke")
model_ads = DQN.load("dqn_trading_model_nke")


# Reset the environment and get the initial observation
obs = env_test.reset()

# Initialize variables to hold evaluation metrics
total_rewards = 0
done = False

# Loop to go through the test data
while not done:
    action, _ = model_ads.predict(obs)
    obs, reward, done, _ = env_test.step(action)
    total_rewards += reward

print(f"Total rewards earned during testing: {total_rewards}")

