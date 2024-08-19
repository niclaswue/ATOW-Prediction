# ATOW-Prediction
https://ansperformance.eu/study/data-challenge


# TODOs:
Add features to improve the prediction.
All multiplications and divisions should be features.

Feature ideas:
- zero fuel weight
- estimated engine type by knowing (Austrian, Swiss, Vueling)
- take off speed
- take off climb rate
- taxi out fuel consumption
- landing speed
- weather at destination
- weather at departure airport
- estimated fuel burn for route (need to account for flight phases)
- airline specific differences => cluster similar data per airline
- route schedule (daily, weekly etc.)
- month of year?

further improvements
- feature importance
- normalize features
- bayesian hyperparameter search
- use neural network as predictor (learn JAX?) 
- incorporate uncertainty?!
