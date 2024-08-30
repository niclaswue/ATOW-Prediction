# Import required libraries
from pytrends.request import TrendReq
import pandas as pd

# Initialize PyTrends
pytrends = TrendReq(hl="en-US", tz=360)

# List of cities
cities = ["Dublin", "Los Angeles", "New York", "Tokyo", "London", "Paris", "Sydney"]

# Initialize an empty list to store results
results_list = []

# def get_interest(city):
pytrends.build_payload(
    kw_list=["Dublin airport"],
    timeframe=["2022-01-01 2022-10-20"],
)
multirange_interest_over_time_df = pytrends.interest_over_time()
print(multirange_interest_over_time_df.head())
from IPython import embed

embed()
exit()  # TODO: Remove DBG

# Function to get interest for a specific city
# def get_interest(city):
#     kw_list = [f"Directions to {city} airport"]
#     pytrends.build_payload(kw_list, timeframe="today 5-y")
#     interest_over_time_df = pytrends.interest_over_time()

#     # Calculate average interest
#     avg_interest = interest_over_time_df[kw_list[0]].mean()

#     return avg_interest


# Fetch data for each city
for city in cities:
    interest = get_interest(city)
    results_list.append({"City": city, "Interest": interest})

# Create DataFrame from the list of results
results_df = pd.DataFrame(results_list)

# Sort the dataframe by interest in descending order
results_df = results_df.sort_values("Interest", ascending=False)

# Display the results
print(results_df)
