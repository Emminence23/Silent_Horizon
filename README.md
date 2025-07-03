# Project Silent Horizon

> Aim : Train a supervised model on a Time Series Classification Task : **buy / hold / sell** decisions on BTC‑USD.

Prototype :

- Let's start with an analytic function called `signal simulator`. Define a horizon (10 minutes) as the information available to the `signal simulator`. 

- If the future prices in [t + 1, t + Horizon] rise above a threshold (0.01%), then our `signal simulator` says it should have been a **buy** signal at time t. Similarly for a supposed **sell** signal.

- We wish to train our model to replicate the activity of our chosen `signal simulator`.

- The fun part is that we do it without actually knowing the future. We only use the lookback data (past 60 minutes).

### Details 

Use-case: Real-time BTC trade decision engine  

Data: [Kaggle BTC-USD 1-minute OHLCV](https://www.kaggle.com/datasets)

Model: 1D CNN + LSTM  

Labels: Future-aware trade opportunities

Input: Last 60 minutes of engineered feature  

Output: 3-class softmax (Buy / Hold / Sell)  

---

### Results

- Accuracy: **~87.6%** on test (with strict label definitions)

> ![Performance Plot](output_0.png)
> The red arrows indicate model suggesting to sell and the green arrows indicate the model suggesting to buy.
> This help in decision making would significantly increase the capture by market micro movements

---

### Beyond the Prototype (Future Plans)

- The prototype shows us that we can build a Model having the ability to act well as a `signal simulator`.  

- Trying some more reliable `signal simulator`s which could essentially generate tradeable signals in real time.

- Trying out different model architectures and tuning the Hyperparameters with optuna.