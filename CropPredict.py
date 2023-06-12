# -*- coding: utf-8 -*-
"""
Created on Tue May 30 08:06:27 2023

@author: ZAM0335
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Stations
from datetime import datetime
from geopy.geocoders import Nominatim

import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import math
from math import sqrt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import requests
import io
from datetime import datetime, date, timedelta
from matplotlib.pyplot import figure
import joblib
import pickle
##PUlLING USDA DATA

def get_usda_data ( state, commodity_name, year):
    api_key = 'B3A5D45D-0A9A-3FA5-869D-15861BD57EF4'
    base_url_api_get = 'http://quickstats.nass.usda.gov/api/api_GET/?key=' + api_key + '&'

    commodity_name = commodity_name #change based on desired commodity
    state = state
     
    parameters =    'source_desc=SURVEY&sector_desc=CROPS&group_desc=FIELD CROPS&commodity_desc=' + commodity_name + '&statisticcat_desc=PROGRESS&' + \
                            'agg_level_desc=STATE&state_alpha=' + state + '&year__GE=' + year + '&format=CSV'
        
    full_url = base_url_api_get + parameters
        
    response = requests.get(full_url)
    content = response.content
    data = pd.read_csv(io.StringIO(content.decode('utf-8')))
 
    data = data[['unit_desc', 'short_desc', 'year','week_ending', 'Value']]
    data = data[data['unit_desc'] == 'PCT PLANTED']
    
    return data

def get_weather_data (latitude, longitude, state):
    start = datetime(2011, 6, 3)
    end = datetime(2022, 6, 10)
    
    stations = Stations()
    Stations.cache_dir = 'Downloads'
    stations = stations.nearby(latitude, longitude)
    stations = stations.inventory('daily')
    station = stations.fetch(200)
    
    station = station[station['region'] == state]

    state_weather = pd.DataFrame()

    #station = station.dropna()
    #station = station.reset_index()
    for i in station.index.values:
        weather_temp = Daily(i, start, end)
        weather_temp = weather_temp.fetch()
        weather_temp = weather_temp.reset_index()
        weather_temp['station'] = i
        state_weather = state_weather.append(weather_temp, ignore_index = True)
    
    return state_weather,station
weather = get_weather_data(41.1158, -98.0017, 'NE')

import statistics
def feature_gen (weather_data, usda_data):
    percip_feature = pd.DataFrame(columns = ['date', 'station', 'num_percip'])
    freezing_days_feature = pd.DataFrame(columns = ['date', 'station', 'num_cold'])
    warm_days_feature = pd.DataFrame(columns = ['date', 'station', 'num_warm'])
    Absolute_Percipitation = pd.DataFrame(columns = ['date', 'station', 'abs_percip'])
    avg_temp_upper = pd.DataFrame(columns = ['date', 'station', 'avg_temp_upper'])
    avg_temp_lower = pd.DataFrame(columns = ['date', 'station', 'avg_temp_lower'])

    weather_data.tavg = weather_data.tavg.fillna((weather_data['tmin'] +weather_data['tmax'])/2)

    usda_data.week_ending = pd.to_datetime(usda_data.week_ending)
    
    start = (usda_data.week_ending.to_numpy())
    end = (usda_data.week_ending - timedelta(weeks=1)).to_numpy()
    station_groups = weather_data.groupby('station')
    
    for name, group in station_groups:
    #print(group.time)
        for i in range(0, len(start)):
            print(str(name) + '_________________________')
            st = start[i]
            en = end[i]
            temp_query = group.query('time >= @en and time <= @st')
            #print(temp_query)
            Num_percip = (temp_query.prcp != 0).sum()
            percip_feature = percip_feature.append({'date':start[i], 'station':name, 'num_percip':Num_percip}, ignore_index = True)
            num_cold = (temp_query.tavg <= 7).sum()
            freezing_days_feature = freezing_days_feature.append({'date':start[i], 'station':name, 'num_cold':num_cold}, ignore_index = True)
            num_warm = (temp_query.tavg >= 20).sum()
            warm_days_feature = warm_days_feature.append({'date':start[i], 'station':name, 'num_warm':num_warm}, ignore_index = True)
            abs_percip = (temp_query.prcp).sum()
            Absolute_Percipitation = Absolute_Percipitation.append({'date':start[i], 'station':name, 'abs_percip':abs_percip}, ignore_index = True)
            try:            
                avg_temp_up = statistics.mean(temp_query.tavg) + (2 * statistics.stdev(temp_query.tavg))
            except:
                avg_temp_up= 0
            avg_temp_upper = avg_temp_upper.append({'date':start[i], 'station':name, 'avg_temp_upper':avg_temp_up}, ignore_index = True)
            try:            
                avg_temp_low = statistics.mean(temp_query.tavg) - (2 * statistics.stdev(temp_query.tavg))
            except:
                avg_temp_low= 0
            avg_temp_lower = avg_temp_lower.append({'date':start[i], 'station':name, 'avg_temp_lower':avg_temp_low}, ignore_index = True)
    
            i = i+1
    
    Complete_Data= pd.DataFrame()

    abs_percip_groups = Absolute_Percipitation.groupby('station')
    for name, group in abs_percip_groups:
        #print(group.abs_percip)
        temp_col = group.abs_percip.reset_index()
        Complete_Data['abs_percip_' + str(name) ] = temp_col.abs_percip
        
    
    warm_days_group = warm_days_feature.groupby('station')
    for name, group in warm_days_group:
        #print(group.num_warm)
        temp_col = group.num_warm.reset_index()
        Complete_Data['warm_days_' + str(name) ] = temp_col.num_warm
    
    freezing_days_groups = freezing_days_feature.groupby('station')
    for name, group in freezing_days_groups:
        #print(group.num_cold)
        temp_col = group.num_cold.reset_index()
        Complete_Data['cold_days_' + str(name) ] = temp_col.num_cold
    
    
    num_percip_groups = percip_feature.groupby('station')
    for name, group in num_percip_groups:
        #print(group.num_percip)
        temp_col = group.num_percip.reset_index()
        Complete_Data['num_percip_' + str(name) ] = temp_col.num_percip
        
    avg_temp_upper__groups = avg_temp_upper.groupby('station')
    for name, group in avg_temp_upper__groups:
        temp_col = group.avg_temp_upper.reset_index()
        Complete_Data['avg_temp_upper' + str(name) ] = temp_col.avg_temp_upper
     
    avg_temp_lower__groups = avg_temp_lower.groupby('station')
    for name, group in avg_temp_lower__groups:
        temp_col = group.avg_temp_lower.reset_index()
        Complete_Data['avg_temp_lower' + str(name) ] = temp_col.avg_temp_lower
        
    Complete_Data = Complete_Data.fillna(0)
    return Complete_Data

def date_features (data, usda):
    usda = usda.reset_index()
    data['Date'] = usda['week_ending']
    data['Percent_Planted'] = usda['Value']   
    data['Date']  = pd.to_datetime(data['Date'] )
    data['month'] = pd.DatetimeIndex(data['Date']).month
    data['week'] =(( (pd.DatetimeIndex(data['Date']).month) * 4) + ((pd.DatetimeIndex(data['Date']).day)/7 ))
    data['week'] = data['week'].astype(int)
    data['day'] = (pd.DatetimeIndex(data['Date']).day)
    
    return data

##############################For Nebraska##############################################
import pandas as pd
import matplotlib.pyplot as plt

usda_data = get_usda_data('NE', 'CORN', '2022')

weather_data, station_data = get_weather_data(41.1158, -98.0017, 'NE')

features_data = feature_gen(weather_data, usda_data)

complete_data = date_features(features_data, usda_data)


##############################For all corn belt##############################################


#NEBRASKA TEST MODEL########

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


usda_data = pd.DataFrame()  

for year in range(2011, 2023):
    data = get_usda_data('NE', 'CORN', str(year))
    usda_data = usda_data.append(data)
weather_data, station_data = get_weather_data(41.1158, -98.0017, 'NE')

features_data = feature_gen(weather_data, usda_data)

complete_data = date_features(features_data, usda_data)



heatmap_data = complete_data.drop(['Date'], axis=1)

correlation_matrix = heatmap_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Variable Correlation Heatmap')
plt.show()

target_variable = 'Percent_Planted' 

correlation_with_target = complete_data.corr()[target_variable]
correlation_with_target = correlation_with_target.drop(target_variable)  
correlation_with_target = correlation_with_target.abs().sort_values(ascending=False)

top_features = correlation_with_target.index[:50] 
top_correlations = correlation_with_target.values[:50]  

correlation_df = pd.DataFrame({'Feature': top_features, 'Correlation': top_correlations})
print(correlation_df)



csv_path = r'\\tedfil01\DataDropDev\PythonPOC\Adam M\E5D96DDD-402A-37EF-8DFB-8A935B6C33C9.csv'
df_csv = pd.read_csv(csv_path)


df_selected = df_csv[["Value", "CV (%)"]]


complete_data = pd.merge(complete_data, df_selected, left_index=True, right_index=True)

print(complete_data)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


X = complete_data.drop(['Value'], axis=1)
y = complete_data['Value']


imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X = imputer.fit_transform(X)
X = scaler.fit_transform(X)


test_sizes = [0.2, 0.3, 0.4]

for test_size in test_sizes:
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    
    model = LinearRegression()
    model.fit(X_train, y_train)

   
    y_pred = model.predict(X_test)

   
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('Test Size:', test_size)
    print('Mean Squared Error:', mse)
    print('R^2 Score:', r2)
    print('------------------------')

########## Random Forrest ################
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X = imputer.fit_transform(X)
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R^2 Score:', r2)

###########XGB  regressor#######
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

X = complete_data.drop(['Value'], axis=1)
y = complete_data['Value']

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X = imputer.fit_transform(X)
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R^2 Score:', r2)



#########IOWA#############

for year in range(2011, 2023):
    data = get_usda_data('IA', 'CORN', str(year))
    usda_data = usda_data.append(data)

weather_data, station_data = get_weather_data(42.4903, -94.2040, 'IA')

iafeatures_data = feature_gen(weather_data, usda_data)

iacomplete_data = date_features(iafeatures_data, usda_data)



for year in range(2011, 2023):
    data  = get_usda_data('IN', 'CORN', str(year))
    usda_data = usda_data.append(data)
weather_data, station_data = get_weather_data(38.0998, -86.1586, 'IN')
infeatures_data = feature_gen(weather_data, usda_data)
incomplete_data =  date_features(infeatures_data, usda_data)

for year in range(2011, 2023):
    data = get_usda_data('MI', 'CORN', str(year))
    usda_data = usda_data.append(data)
weather_data, stattion_data = get_weather_data(44.182205, -84.506836, 'MI')
mifeatures_data = feature_gen(weather_data, usda_data)
micomplete_data = data_features(mifeatures_daya, usda_data)

for year in range(2011, 2023):
    data = get_usda_data('MO', 'CORN', str(year))
    usda_data = usda_data.append(data)
weather_data, stattion_data = get_weather_data(38.573936, -92.603760, 'MI')
mofeatures_data = feature_gen(weather_data, usda_data)
mocomplete_data = data_features(mofeatures_daya, usda_data)

#usda_data = pd.DataFrame()  
#states = {
#    'IL': (38.5489, -89.1270),
#    'IN': (38.0998, -86.1586),
#    'IA': (42.4903, -94.2040),
#    'MI': (44.182205, -84.506836),
#    'MN': (44.9765, -93.2761),
#    'MO': (38.573936, -92.603760),
#    'OH': (40.367474, -82.996216)
}


#for state, coordinates in states.items():
    

#    for year in range(2011, 2023):
#        data = get_usda_data(state, 'CORN', str(year))
#        usda_data = usda_data.append(data)

#    weather_data, station_data = get_weather_data(coordinates[0], coordinates[1], state)

    
#    features_data = feature_gen(weather_data, usda_data)

  
#    complete_data = date_features(features_data, usda_data)





































import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import io
from meteostat import Point, Daily, Stations
import statistics

state_coordinates = {
    'IL': (38.5489, -89.1270),
    'IN': (38.0998, -86.1586),
    'IA': (42.4903, -94.2040),
    'MI': (44.182205, -84.506836),
    'MN': (44.9765, -93.2761),
    'OH': (40.367474, -82.996216),
    'PA': (40.335648, -75.926872)
}

def get_usda_data(state, commodity_name, year):
    api_key = 'B3A5D45D-0A9A-3FA5-869D-15861BD57EF4'
    base_url_api_get = 'http://quickstats.nass.usda.gov/api/api_GET/?key=' + api_key + '&'

    commodity_name = commodity_name
    state = state

    parameters = 'source_desc=SURVEY&sector_desc=CROPS&group_desc=FIELD CROPS&commodity_desc=' + commodity_name + '&statisticcat_desc=PROGRESS&' + \
                 'agg_level_desc=STATE&state_alpha=' + state + '&year__GE=' + year + '&format=CSV'

    full_url = base_url_api_get + parameters

    response = requests.get(full_url)
    content = response.content
    data = pd.read_csv(io.StringIO(content.decode('utf-8')))

    data = data[['unit_desc', 'short_desc', 'year','week_ending', 'Value']]
    data = data[data['unit_desc'] == 'PCT PLANTED']

    return data


def get_weather_data(latitude, longitude, state):
    start = datetime(1979, 6, 3)
    end = datetime(2022, 6, 10)

    stations = Stations()
    Stations.cache_dir = 'Downloads'
    stations = stations.nearby(latitude, longitude)
    stations = stations.inventory('daily')
    station = stations.fetch(200)

    station = station[station['region'] == state]

    state_weather = pd.DataFrame()

    for i in station.index.values:
        weather_temp = Daily(i, start, end)
        weather_temp = weather_temp.fetch()
        weather_temp = weather_temp.reset_index()
        weather_temp['station'] = i
        state_weather = state_weather.append(weather_temp, ignore_index=True)

    return state_weather, station


def feature_gen(weather_data, usda_data):
    # Generate additional features from the data
    percip_feature = pd.DataFrame(columns=['date', 'station', 'num_percip'])
    freezing_days_feature = pd.DataFrame(columns=['date', 'station', 'num_cold'])
    warm_days_feature = pd.DataFrame(columns=['date', 'station', 'num_warm'])
    Absolute_Percipitation = pd.DataFrame(columns=['date', 'station', 'abs_percip'])
    avg_temp_upper = pd.DataFrame(columns=['date', 'station', 'avg_temp_upper'])
    avg_temp_lower = pd.DataFrame(columns=['date', 'station', 'avg_temp_lower'])

    weather_data['tavg'] = weather_data['tavg'].fillna((weather_data['tmin'] + weather_data['tmax']) / 2)
    usda_data['week_ending'] = pd.to_datetime(usda_data['week_ending'])

    start = usda_data['week_ending'].to_numpy()
    end = (usda_data['week_ending'] - timedelta(weeks=1)).to_numpy()
    station_groups = weather_data.groupby('station')

    for name, group in station_groups:
        for i in range(0, len(start)):
            print(str(name) + '_________________________')
            st = start[i]
            en = end[i]
            temp_query = group.query('time >= @en and time <= @st')
            Num_percip = (temp_query['prcp'] != 0).sum()
            percip_feature = percip_feature.append({'date': start[i], 'station': name, 'num_percip': Num_percip},
                                                   ignore_index=True)
            num_cold = (temp_query['tavg'] <= 7).sum()
            freezing_days_feature = freezing_days_feature.append({'date': start[i], 'station': name, 'num_cold': num_cold},
                                                                  ignore_index=True)
            num_warm = (temp_query['tavg'] >= 20).sum()
            warm_days_feature = warm_days_feature.append({'date': start[i], 'station': name, 'num_warm': num_warm},
                                                          ignore_index=True)
            abs_percip = temp_query['prcp'].sum()
            Absolute_Percipitation = Absolute_Percipitation.append({'date': start[i], 'station': name,
                                                                     'abs_percip': abs_percip}, ignore_index=True)
            try:
                avg_temp_up = statistics.mean(temp_query['tavg']) + (2 * statistics.stdev(temp_query['tavg']))
            except:
                avg_temp_up = 0
            avg_temp_upper = avg_temp_upper.append({'date': start[i], 'station': name, 'avg_temp_upper': avg_temp_up},
                                                   ignore_index=True)
            try:
                avg_temp_low = statistics.mean(temp_query['tavg']) - (2 * statistics.stdev(temp_query['tavg']))
            except:
                avg_temp_low = 0
            avg_temp_lower = avg_temp_lower.append({'date': start[i], 'station': name, 'avg_temp_lower': avg_temp_low},
                                                   ignore_index=True)

    Complete_Data = pd.DataFrame()

    abs_percip_groups = Absolute_Percipitation.groupby('station')
    for name, group in abs_percip_groups:
        temp_col = group['abs_percip'].reset_index()
        Complete_Data['abs_percip_' + str(name)] = temp_col['abs_percip']

    warm_days_group = warm_days_feature.groupby('station')
    for name, group in warm_days_group:
        temp_col = group['num_warm'].reset_index()
        Complete_Data['warm_days_' + str(name)] = temp_col['num_warm']

    freezing_days_groups = freezing_days_feature.groupby('station')
    for name, group in freezing_days_groups:
        temp_col = group['num_cold'].reset_index()
        Complete_Data['cold_days_' + str(name)] = temp_col['num_cold']

    num_percip_groups = percip_feature.groupby('station')
    for name, group in num_percip_groups:
        temp_col = group['num_percip'].reset_index()
        Complete_Data['num_percip_' + str(name)] = temp_col['num_percip']

    avg_temp_upper__groups = avg_temp_upper.groupby('station')
    for name, group in avg_temp_upper__groups:
        temp_col = group['avg_temp_upper'].reset_index()
        Complete_Data['avg_temp_upper' + str(name)] = temp_col['avg_temp_upper']

    avg_temp_lower__groups = avg_temp_lower.groupby('station')
    for name, group in avg_temp_lower__groups:
        temp_col = group['avg_temp_lower'].reset_index()
        Complete_Data['avg_temp_lower' + str(name)] = temp_col['avg_temp_lower']

    usda_data['week_ending'] = pd.to_datetime(usda_data['week_ending'])
    usda_data = usda_data.sort_values(by='week_ending')
    usda_data['week_ending'] = usda_data['week_ending'].dt.strftime('%Y-%m-%d')

    Complete_Data = Complete_Data.set_index(usda_data['week_ending'])

    return Complete_Data


def plot_feature(feature_name, state, feature_data):
    plt.figure(figsize=(15, 5))
    for i in range(0, len(state)):
        plt.plot(feature_data.index, feature_data[feature_name + '_' + state[i]], label=state[i])
    plt.xlabel('Year')
    plt.ylabel(feature_name)
    plt.title('USDA Data: ' + feature_name)
    plt.legend(loc='upper right')
    plt.show()


# Example usage
commodity_name = 'CORN'
year = '2019'

for state, coordinates in state_coordinates.items():
    usda_data = get_usda_data(state, commodity_name, year)
    weather_data, stations = get_weather_data(coordinates[0], coordinates[1], state)
    feature_data = feature_gen(weather_data, usda_data)

    # Plotting absolute precipitation feature
    plot_feature('abs_percip', [state], feature_data) 


###############proof of concept##########################
    
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read the weather station and USDA data samples
weather_stations = pd.read_csv('weather_stations.csv')
usda_data = pd.read_csv('usda_data.csv')

# Filter data for a specific state (e.g., Illinois)
state = 'IL'
filtered_stations = weather_stations[weather_stations['region'] == state]
filtered_usda_data = usda_data[usda_data['short_desc'] == 'CORN - PROGRESS, MEASURED IN PCT PLANTED']

# Merge the weather and USDA data based on week and year
merged_data = pd.merge(filtered_stations, filtered_usda_data, left_on=['daily_end', 'year'], right_on=['week_ending', 'year'])

# Prepare the features (weather data) and target variable (crop yield)
features = merged_data[['latitude', 'longitude', 'elevation', 'Value']]
target = merged_data['Value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict crop yield for a new set of features
new_features = pd.DataFrame([[38.5, -89, 163, 2023]], columns=['latitude', 'longitude', 'elevation', 'year'])
predicted_yield = model.predict(new_features)

print(f"Predicted crop yield for the specified state: {predicted_yield[0]}")
