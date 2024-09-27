# pip install git+https://github.com/DGAC/Acropole.git
import pandas as pd
from acropole import FuelEstimator

fe = FuelEstimator()

flight = pd.DataFrame(
    {
        "typecode": ["A320", "A320", "A320", "A320"],
        "groundspeed": [400, 410, 420, 430],
        "altitude": [10000, 11000, 12000, 13000],
        "vertical_rate": [2000, 1500, 1000, 500],
        # optional features:
        "second": [0.0, 1.0, 2.0, 3.0],
        "airspeed": [400, 410, 420, 430],
        "mass": [60000, 60000, 60000, 60000],
    }
)

flight_fuel = fe.estimate(flight)  # flight.data if traffic flight
from IPython import embed

embed()
exit()  # TODO: Remove DBG
