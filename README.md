# rao-blasso

Bayesian Lasso with Rao-Blackwellized variance estimator. This
estimator leads to an inference procedure that requires passing
through the base data only once (during initialization), so that each
subsequent iteration of the Gibbs sampler can be done much faster
without degrading the quality of the solution. This becomes
particularly useful with very large datasets (in number of rows) with
a moderate amount of features.

## Requirements

  - numpy
  - scipy
  - tqdm

## Runnable example

```
from rao_blasso import RBLasso 
import pandas as pd
import numpy as np

x = pd.read_csv('https://llperezp-datasets.s3.us-east-2.amazonaws.com/commcrime.csv')
X = x.drop('ViolentCrimesPerPop', axis=1).to_numpy()
y = x['ViolentCrimesPerPop'].to_numpy()

rb = RBLasso(rao_s2=True, alpha=0.1).fit(X, y, num_iter=2000)
rmse = np.sqrt(np.mean((rb.predict(X) - y)**2))
```