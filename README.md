### Improvements of V2:

Possible additional classification
- Predict the percentage of rise/fall

Possible additional features:

- NYSE/NASDAQ(or SP500)kind of reference on that day or days before
    - possibly, use the reference for a kind of stocks, like technology
- The performance prior to the day (2~3 days)
- Quadratic terms
- Other stock performance - in the same category

### 20160722 v1

|        Feature       |      Figure      |
|:--------------------:|:----------------:|
|     Hidden layers    |         1        |
| Hidden layer 1 units |        10        |
|       Features       |         4        |
|    Classifications   |         1        |
|   Training set size  |        415       |
|   Network mapping    |      Direct      |
|      Activation      |      Sigmoid     |
|     Optimization     | Gradient Descent |
|     Learning rate    |       0.005      |
|      Iterations      |      110000      |
|    Final accuracy    |       80%        |


### 20160725 v1.1
|        Feature       |  Figure |
|:--------------------:|:-------:|
|     Hidden layers    |    2    |
| Hidden layer 1 units |    32   |
| Hidden layer 2 units |    16   |
|       Features       |    4    |
|    Classifications   |    1    |
|   Training set size  |   520   |
|   Network mapping    | Sigmoid |
|      Activation      | Sigmoid |
|     Optimization     |   Adam  |
|     Learning rate    |  0.0001 |
|      Iterations      |  30000  |
|    Final accuracy    |   85%   |

### 20160726 Commit 1ddd432
- Need to fine tune the stochastic training


