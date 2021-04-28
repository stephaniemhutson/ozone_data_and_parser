import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file_name = "./daily_44201_%d.csv"

years = [i + 1990 for i in range(31)]

files = [file_name % year for year in years]

CITIES_STATES = [
    ("New York", "New York"),
    ("Los Angeles", "California"),
    ("Chicago", "Illinois"),
    ("Houston", "Texas"),
    ("Phoenix", "Arizona"),
    ("Philadelphia", "Pennsylvania"),
    ("San Antonio", "Texas"),
    ("San Diego", "California"),
    ("Dallas", "Texas"),
    ("San Jose","California"),
    ("Austin", "Texas"),
    ("Jacksonville", "Florida"),
    ("Fort Worth", "Texas"),
    ("Columbus", "Ohio"),
    ("Charlotte", "North Carolina"),
    ("San Francisco",   "California"),
    ("Indianapolis", "Indiana"),
    ("Seattle", "Washington"),
    ("Denver", "Colorado"),
    ("Washington", "District Of Columbia"),
    ("Boston", "Massachusetts"),
    ("El Paso", "Texas"),
    ("Nashville", "Tennessee"),
    ("Detroit", "Michigan"),
    ("Oklahoma City", "Oklahoma"),
    ("Portland", "Oregon"),
    ("Las Vegas", "Nevada"),
    ("Memphis", "Tennessee"),
    ("Louisville", "Kentucky"),
    ("Baltimore", "Maryland"),
    ("Milwaukee", "Wisconsin"),
    ("Albuquerque", "New Mexico"),
    ("Tucson", "Arizona"),
    ("Fresno", "California"),
    ("Mesa", "Arizona"),
    ("Sacramento", "California"),
    ("Atlanta", "Georgia"),
    ("Kansas City", "Missouri"),
    ("Colorado Springs", "Colorado"),
    ("Omaha", "Nebraska"),
    ("Raleigh", "North Carolina"),
    ("Miami", "Florida"),
    ("Long Beach", "California"),
    ("Virginia Beach", "Virginia"),
    ("Oakland", "California"),
    ("Minneapolis", "Minnesota"),
    ("Tulsa", "Oklahoma"),
    ("Tampa", "Florida"),
    ("Arlington", "Texas"),
    ("New Orleans", "Louisiana"),
    ("Wichita", "Kansas"),
    ("Bakersfield", "California"),
    ("Cleveland", "Ohio"),
    ("Aurora", "Colorado"),
    ("Anaheim", "California"),
    ("Honolulu", "Hawaii"),
    ("Santa Ana", "California"),
    ("Riverside", "California"),
    ("Corpus Christi", "Texas"),
    ("Lexington", "Kentucky"),
    ("Henderson", "Nevada"),
    ("Stockton", "California"),
    ("Saint Paul", "Minnesota"),
    ("Cincinnati", "Ohio"),
    ("St. Louis", "Missouri"),
    ("Pittsburgh", "Pennsylvania"),
    ("Greensboro", "North Carolina"),
    ("Lincoln", "Nebraska"),
    ("Anchorage", "Alaska"),
    ("Plano", "Texas"),
    ("Orlando", "Florida"),
    ("Irvine", "California"),
    ("Newark", "New Jersey"),
    ("Durham", "North Carolina"),
    ("Chula Vista", "California"),
    ("Toledo", "Ohio"),
    ("Fort Wayne", "Indiana"),
    ("St. Petersburg", "Florida"),
    ("Laredo", "Texas"),
    ("Jersey City", "New Jersey"),
    ("Chandler", "Arizona"),
    ("Madison", "Wisconsin"),
    ("Lubbock", "Texas"),
    ("Scottsdale", "Arizona"),
    ("Reno", "Nevada"),
    ("Buffalo", "New York"),
    ("Gilbert", "Arizona"),
    ("Glendale", "Arizona"),
    ("North Las Vegas", "Nevada"),
    ("Winston–Salem", "North Carolina"),
    ("Chesapeake", "Virginia"),
    ("Norfolk", "Virginia"),
    ("Fremont", "California"),
    ("Garland", "Texas"),
    ("Irving", "Texas"),
    ("Hialeah", "Florida"),
    ("Richmond", "Virginia"),
    ("Boise", "Idaho"),
    ("Spokane", "Washington"),
    ("Baton Rouge", "Louisiana"),
    ("Tacoma", "Washington"),
    ("San Bernardino", "California"),
    ("Modesto", "California"),
    ("Fontana", "California"),
    ("Des Moines", "Iowa"),
    ("Moreno Valley", "California"),
    ("Santa Clarita", "California"),
    ("Fayetteville", "North Carolina"),
    ("Birmingham", "Alabama"),
    ("Oxnard", "California"),
    ("Rochester", "New York"),
    ("Port St. Lucie", "Florida"),
    ("Grand Rapids", "Michigan"),
    ("Huntsville", "Alabama"),
    ("Salt Lake City", "Utah"),
    ("Frisco", "Texas"),
    ("Yonkers", "New York"),
]

CITIES_STATES = [(city, state, city) for city, state in CITIES_STATES]

CBSA_NAMES = [
    ("Portland-Vancouver", "Oregon"),
    ("Boston-Cambridge-Newton, MA-NH", "Massachusetts"),
    ("Seattle-Tacoma-Bellevue, WA", "Washington"),
    ("Oxnard-Thousand Oaks-Ventura, CA", "California")
]

CBSA_NAMES = [(city, state, city) for city, state in CBSA_NAMES]


city_states = pd.DataFrame(CITIES_STATES, columns=['City Name', 'State Name', 'Classifier'])
cbsa_names = pd.DataFrame(CBSA_NAMES, columns=['CBSA Name', 'State Name', 'Classifier'])

all_data = None

for i, csv in enumerate(files):
    print(f"Year: {1990 + i}")
    df_raw = pd.read_csv(csv, header=[0])
    df_raw = df_raw.astype({'Date Local': 'datetime64'})
    df_city = pd.merge(df_raw, city_states, how="inner", on=["City Name", "State Name"])
    df_cbsa = pd.merge(df_raw, cbsa_names, how="inner", on=["CBSA Name", "State Name"])
    if all_data is None:
        all_data = pd.concat([df_city, df_cbsa])
    else:
        all_data = pd.concat([all_data, df_city, df_cbsa])

all_data.insert(1, "Is California", pd.Series(all_data['State Name'] == "California"))
# all_data = all_data.filter(items=["Classifier", "State Name", "AQI", "Date Local", "Arithmetic Mean", '1st Max Value'])
all_data.insert(2, "Year", pd.DatetimeIndex(all_data['Date Local']).year)

city_data = []

for row in CITIES_STATES + CBSA_NAMES:
    _, state, classifier = row
    city_data.append(all_data.where(all_data['Classifier'] == classifier).dropna(subset=['State Name']))

year_data = pd.DataFrame(columns=['Classifier', 'State', 'County', 'Year', 'Mean AQI', 'Mean O3', 'Is California', 'Mean max'])

columns = ['Classifier', 'State'] + years
aqi_df = pd.DataFrame(columns=columns)
o3_df = pd.DataFrame(columns=columns)
max_df = pd.DataFrame(columns=columns)

i = 0
for data in city_data:
    if data.empty:
        continue
    state =  [_ for _ in data.to_dict()['State Name'].values()][0]
    is_california = state == 'California'
    classifier =  [_ for _ in data.to_dict()['Classifier'].values()][0]
    county =  [_ for _ in data.to_dict()['County Name'].values()][0]
    aqi_row = {
        'Classifier': [classifier],
        'State': [state],
    }
    o3_row = {
        'Classifier': [classifier],
        'State': [state],
    }
    max_row = {
        'Classifier': [classifier],
        'State': [state],
    }
    for year in years:
        _d = data.where(data['Year'] == year)
        if not _d.empty:
            mean = _d.mean()
            _df = pd.DataFrame({
                'Classifier': [classifier],
                'Year': [year],
                'Mean AQI': [mean['AQI']],
                'Mean O3': [mean['Arithmetic Mean']],
                'Mean max': [mean['1st Max Value']],
                'State': [state],
                'Is California': [is_california],
                'County': [county]
            })
            year_data = pd.concat([year_data, _df])
            aqi_row[year] = [mean['AQI']]
            o3_row[year] = [mean['Arithmetic Mean']]
            max_row[year] = [mean['1st Max Value']]
        else:
            aqi_row[year] = [None]
            o3_row[year] = [None]
            max_row[year] = [None]
    aqi_df = pd.concat([aqi_df, pd.DataFrame(aqi_row)])
    o3_df = pd.concat([o3_df, pd.DataFrame(o3_row)])
    max_df = pd.concat([max_df, pd.DataFrame(max_row)])

year_data = year_data.dropna(subset=["Mean AQI", "Mean O3"])
year_data.to_csv('./year_city_data_aqi_o3_max.csv')
aqi_df.to_csv('./aqi.csv')
o3_df.to_csv('./o3.csv')
max_df.to_csv('./max.csv')
