# Most Churn Models Cheat — Here's How I Made Sure Mine Didn't

*What building a leakage-safe churn pipeline on a million real KKBox subscribers taught me about the gap between a notebook and a product.*

Most churn prediction write-ups follow the same five steps: load one clean CSV, do a random 80/20 split, train a model, report a 90-something percent accuracy, and call it done. It looks impressive right up until you ask one question: would this model still work on a customer it has never seen, on a date it has never seen? For a lot of these projects, the honest answer is no — the accuracy number is mostly measuring how well the model memorized the past, not how well it predicts the future.

I wanted to build something closer to what a churn model actually looks like inside a subscription business, so I used the KKBox WSDM Churn Challenge dataset: roughly a million real subscribers spread across three raw tables — transaction history (plan changes, renewals, cancellations), daily listening logs, and member demographics. There's no flat CSV handed to you here. The first real piece of engineering is joining all three tables per user and turning the result into 34 engineered features. That part is mechanical, though. The part that actually matters is the one most tutorials skip entirely: making sure the evaluation is honest.

### The leak hiding in plain sight

One of the most "predictive" features available in this dataset is `days_until_expiry` — how many days are left on a user's subscription. It's a fantastic predictor of churn, for an uninteresting reason: a user who's about to churn almost by definition has a subscription that's about to expire. Train on that feature with a random split and you get a model that looks outstanding and is nearly useless, because in production you don't get to know a user's exact expiry countdown before deciding whether to intervene — that countdown *is* the thing you're trying to act ahead of. I dropped it.

The second half of the fix is the part that almost never gets discussed: how the data gets split in the first place. A random 80/20 split treats every row as an independent coin flip, which isn't true when rows are tied together in time. So instead of shuffling, I split temporally — by quantile on `days_since_last_txn` — holding out the most recent slice of activity as the test set, the same way the model will actually be judged once it's live: on behavior it hasn't seen yet.

You can see the split doing its job in the data itself. The training set has a 10.4% churn rate; the test set has 3.1%. A random split would have given me nearly identical numbers on both sides, because shuffling erases exactly the kind of time-based shift real deployment has to survive. The fact that they're different isn't a problem to explain away — it's the evidence that the evaluation is actually honest.

### The model bake-off told a more interesting story than "pick the best one"

With a clean setup in place, I trained three candidates — logistic regression as a baseline, XGBoost, and LightGBM — and tracked every run in MLflow instead of eyeballing notebook output. The results weren't a simple ranking:

Logistic regression looked solid on paper (0.94 AUC) but collapsed once forced to make an actual yes/no decision — 0.58 F1, 0.53 recall. A high AUC can hide a model that's bad at the decision you actually need it to make.

XGBoost and LightGBM both crossed 0.96 AUC and landed within a hair of each other on F1 — 0.822 versus 0.819. On F1 alone it's basically a coin flip. LightGBM got the production slot because it pulled ahead on ROC-AUC (0.969 vs 0.960) and average precision (0.837 vs 0.791), the metrics that matter more when you care about ranking *all* users by churn risk rather than just the ones that cross one threshold. At its tuned decision threshold of 0.82 — not the default 0.5 most tutorials never touch — it landed at 83% recall and 81% precision.

### A model isn't a product until something can call it

The last piece is the one I think actually separates a portfolio project from a notebook: the model doesn't live as a `.pkl` file in a folder. It's served behind a FastAPI endpoint, containerized with Docker, returning a churn probability for a given user on request. That's the difference between "I can train a model" and "I can ship one" — and it's usually the part that's hardest to fake in an interview, because it's the part most tutorials never bother to do.

### The actual breakthrough

There's no exotic architecture here and no custom loss function. If there's a breakthrough in this project, it's something more boring and more useful than a clever model: treating the evaluation setup with the same rigor as the model itself. A model that's near-perfect on a leaking, randomly-shuffled split isn't a better model — it's a worse one, because it's lying to you about how it'll perform on the only users that actually matter: the ones you haven't seen yet.

I'm building this as part of a broader ML portfolio while finishing my Master's in Data Science, and it's the project I'd point to first if you want to see how I think about the gap between "the model works" and "the model works in production." Code, notebooks, and the full pipeline are on GitHub: [github.com/classifiedSni/churn-prediction](https://github.com/classifiedSni/churn-prediction)
