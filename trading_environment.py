# Import necessary modules
import gym
import numpy as np
import pandas as pd

# Define the custom trading environment
class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=1000):
        super(TradingEnv, self).__init__()
        
        # Action: 0->Hold, 1->Buy, 2->Sell
        self.action_space = gym.spaces.Discrete(3)
        
        # State: [Balance, Stock Price, Number of Stocks Owned]
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.df = df  # Dataframe containing stock data
        self.initial_balance = initial_balance  # Starting money
        self.balance = initial_balance  # Money left
        self.current_step = 0  # Current time step
        self.stock_price = 0  # Current stock price
        self.stock_owned = 0  # Number of stocks owned

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.stock_price = self.df.loc[self.current_step, 'Normalized_Close']
        self.stock_owned = 0
        # return [self.balance, self.stock_price, self.stock_owned]
        return np.array([self.balance, self.stock_price, self.stock_owned])

    def step(self, action):
        self.current_step += 1
        self.stock_price = self.df.loc[self.current_step, 'Normalized_Close']
        
        done = False
        if self.current_step == len(self.df) - 1:
            done = True
            
        reward = 0
        if action == 1:  # Buy
            self.stock_owned += 1
            self.balance -= self.stock_price
            #  reward based on potential for future profit
            reward = max(0, (self.stock_price - self.df.loc[self.current_step - 1, 'Normalized_Close']) * self.stock_owned) - 0.01 * np.abs(500 - self.balance)
    
        elif action == 2:  # Sell
            self.stock_owned -= 1
            self.balance += self.stock_price
            #  reward based on profit from selling
            reward = max(0, (self.stock_price - self.df.loc[self.current_step - 1, 'Normalized_Close']) * self.stock_owned) - 0.01 * np.abs(500 - self.balance)

        else:
            # Reward for holding can be neutral or slightly positive if the stock price goes up
            reward = max(0, (self.stock_price - self.df.loc[self.current_step - 1, 'Normalized_Close']) * self.stock_owned) - 0.01 * np.abs(500 - self.balance)
     

        # Adding a term that discourages excessive stock holdings (risk management)
        reward -= 0.02 * np.abs(10 - self.stock_owned)
    
        info = {}
        
        # return [self.balance, self.stock_price, self.stock_owned], reward, done, info
        return np.array([self.balance, self.stock_price, self.stock_owned]), reward, done, info

# Load your Adidas training data into a DataFrame
df = pd.read_csv('/Users/farhaanfayaz/Desktop/RL Project/Data/ADS.DE_train.csv')

# Initialize your custom trading environment with the Adidas data
env = TradingEnv(df=df)

# Reset the environment and get the initial state
initial_state = env.reset()
print("Initial State:", initial_state)

# Take a sample "Buy" action (action code: 1)
next_state, reward, done, _ = env.step(1)
print("Next State:", next_state)
print("Reward:", reward)
