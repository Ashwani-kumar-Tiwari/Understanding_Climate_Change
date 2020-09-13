
# coding: utf-8

# # Importing necessary Modules

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py # interactive visualization features
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Is Global Temperature really increasing?

# ### Reading the Global Temperatures Data Set

# In[7]:


global1=pd.read_csv('C:/Users/ANANT SHARMA .LAPTOP-FFV0CT73/Desktop/DS+ML/Project ITS/GlobalTemperatures.csv')
global1=global1[['dt','LandAverageTemperature']]
print(global1)


# ### Cleaning the Data: Handling missing values

# In[8]:


global1.dropna(inplace=True)
print(global1)


# ### Transforming Data: Converting the Date-Time timestamp column into Year only values and averaging out the temperature data for different months of the year to get the average for that year

# In[9]:


global1['dt']=pd.to_datetime(global1.dt).dt.strftime('%d/%m/%Y')
global1['dt']=global1['dt'].apply(lambda x:x[6:])
global1=global1.groupby(['dt'])['LandAverageTemperature'].mean().reset_index()
print(global1)


# ### Visualizing our transformed data

# In[10]:


trace=go.Scatter(
    x=global1['dt'],
    y=global1['LandAverageTemperature'],
    mode='lines',
    )
data=[trace]

py.iplot(data, filename='line-mode')


# ## From the above graph, it is evident that global temperatures are indeed rising year on year.
# #### Noted Anomaly in the data: A huge drop in the temperature for 1750

# ## Comparing our data over any two Months

# ### Following similar procedure as before

# In[11]:


global2=pd.read_csv('C:/Users/ANANT SHARMA .LAPTOP-FFV0CT73/Desktop/DS+ML/Project ITS/GlobalTemperatures.csv')
global2=global2[['dt','LandAverageTemperature']]
global2.dropna(inplace=True)
global2['dt']=pd.to_datetime(global2.dt).dt.strftime('%d/%m/%Y')
global2['month']=global2['dt'].apply(lambda x:x[3:5])
global2['year']=global2['dt'].apply(lambda x:x[6:])
global2=global2[['month','year','LandAverageTemperature']]
print(global2)


# In[12]:


global2['month']=global2['month'].map({'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'})
print(global2)


# ### Now that our data has been transformed as required, let us plot it for two months: August and November

# In[13]:


def plot_month(month1,month2):
    a=global2[global2['month']==month1]
    b=global2[global2['month']==month2]
    trace0 = go.Scatter(
    x = a['year'],
    y = a['LandAverageTemperature'],
    mode = 'lines',
    name = month1
    )
    
    trace1 = go.Scatter(
    x = b['year'],
    y = b['LandAverageTemperature'],
    mode = 'lines',
    name = month2
    )
    data = [trace0,trace1]
    py.iplot(data, filename='line-mode')
plot_month('Aug','Nov')


# ## We see a similar trend for the months also. There is a continous increase in the temperatures in individual months too. We can check for any two months by just replacing the month names in the function.

# # Average Temperature By Country (Visualization: Interactive Map)

# ### Reading dataset for Global Temperatures sorted by Country

# In[16]:


temp_country=pd.read_csv('C:/Users/ANANT SHARMA .LAPTOP-FFV0CT73/Desktop/DS+ML/Project ITS/GlobalLandTemperaturesByCountry.csv')
print(temp_country)


# ### Cleaning our Data

# In[17]:


temp_country['Country'].replace({'Denmark (Europe)':'Denmark','France (Europe)':'France','Netherlands (Europe)':'Netherlands','United Kingdom (Europe)':'Europe'},inplace=True)
temp_country.fillna(0,inplace=True)
print(temp_country)


# ### Transforming our Data

# In[18]:


temp_country1=temp_country.groupby(['Country'])['AverageTemperature'].mean().reset_index()
print(temp_country1)


# #### Creating two lists: 1st for Country Name and 2nd for Country Code

# In[19]:


l1=list(['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra','Angola', 'Anguilla', 'Antigua and Barbuda', 'Argentina', 'Armenia','Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas, The',
       'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize','Benin', 'Bermuda', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina','Botswana', 'Brazil', 'British Virgin Islands', 'Brunei',
       'Bulgaria', 'Burkina Faso', 'Burma', 'Burundi', 'Cabo Verde','Cambodia', 'Cameroon', 'Canada', 'Cayman Islands','Central African Republic', 'Chad', 'Chile', 'China', 'Colombia',
       'Comoros', 'Congo, Democratic Republic of the','Congo, Republic of the', 'Cook Islands', 'Costa Rica',"Cote d'Ivoire", 'Croatia', 'Cuba', 'Curacao', 'Cyprus','Czech Republic', 'Denmark', 'Djibouti', 'Dominica','Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
       'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia','Falkland Islands (Islas Malvinas)', 'Faroe Islands', 'Fiji','Finland', 'France', 'French Polynesia', 'Gabon', 'Gambia, The',
       'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland','Grenada', 'Guam', 'Guatemala', 'Guernsey', 'Guinea-Bissau','Guinea', 'Guyana', 'Haiti', 'Honduras', 'Hong Kong', 'Hungary',
       'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland','Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jersey','Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Korea, North',
       'Korea, South', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia','Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macau', 'Macedonia', 'Madagascar','Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta',
       'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico','Micronesia, Federated States of', 'Moldova', 'Monaco', 'Mongolia',
       'Montenegro', 'Morocco', 'Mozambique', 'Namibia', 'Nepal','Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua','Nigeria', 'Niger', 'Niue', 'Northern Mariana Islands', 'Norway',
       'Oman', 'Pakistan', 'Palau', 'Panama', 'Papua New Guinea','Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal','Puerto Rico', 'Qatar', 'Romania', 'Russia', 'Rwanda','Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin','Saint Pierre and Miquelon', 'Saint Vincent and the Grenadines','Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia','Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore',
       'Sint Maarten', 'Slovakia', 'Slovenia', 'Solomon Islands','Somalia', 'South Africa', 'South Sudan', 'Spain', 'Sri Lanka','Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria',
       'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey','Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine',
       'United Arab Emirates', 'United Kingdom', 'United States','Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam',
       'Virgin Islands', 'West Bank', 'Yemen', 'Zambia', 'Zimbabwe']) #Country names
print(l1)


# In[20]:


l2=list(['AFG', 'ALB', 'DZA', 'ASM', 'AND', 'AGO', 'AIA', 'ATG', 'ARG','ARM', 'ABW', 'AUS', 'AUT', 'AZE', 'BHM', 'BHR', 'BGD', 'BRB','BLR', 'BEL', 'BLZ', 'BEN', 'BMU', 'BTN', 'BOL', 'BIH', 'BWA','BRA', 'VGB', 'BRN', 'BGR', 'BFA', 'MMR', 'BDI', 'CPV', 'KHM',
       'CMR', 'CAN', 'CYM', 'CAF', 'TCD', 'CHL', 'CHN', 'COL', 'COM','COD', 'COG', 'COK', 'CRI', 'CIV', 'HRV', 'CUB', 'CUW', 'CYP','CZE', 'DNK', 'DJI', 'DMA', 'DOM', 'ECU', 'EGY', 'SLV', 'GNQ',
       'ERI', 'EST', 'ETH', 'FLK', 'FRO', 'FJI', 'FIN', 'FRA', 'PYF','GAB', 'GMB', 'GEO', 'DEU', 'GHA', 'GIB', 'GRC', 'GRL', 'GRD','GUM', 'GTM', 'GGY', 'GNB', 'GIN', 'GUY', 'HTI', 'HND', 'HKG',
       'HUN', 'ISL', 'IND', 'IDN', 'IRN', 'IRQ', 'IRL', 'IMN', 'ISR','ITA', 'JAM', 'JPN', 'JEY', 'JOR', 'KAZ', 'KEN', 'KIR', 'KOR',
       'PRK', 'KSV', 'KWT', 'KGZ', 'LAO', 'LVA', 'LBN', 'LSO', 'LBR','LBY', 'LIE', 'LTU', 'LUX', 'MAC', 'MKD', 'MDG', 'MWI', 'MYS','MDV', 'MLI', 'MLT', 'MHL', 'MRT', 'MUS', 'MEX', 'FSM', 'MDA',
       'MCO', 'MNG', 'MNE', 'MAR', 'MOZ', 'NAM', 'NPL', 'NLD', 'NCL','NZL', 'NIC', 'NGA', 'NER', 'NIU', 'MNP', 'NOR', 'OMN', 'PAK','PLW', 'PAN', 'PNG', 'PRY', 'PER', 'PHL', 'POL', 'PRT', 'PRI',
       'QAT', 'ROU', 'RUS', 'RWA', 'KNA', 'LCA', 'MAF', 'SPM', 'VCT','WSM', 'SMR', 'STP', 'SAU', 'SEN', 'SRB', 'SYC', 'SLE', 'SGP',
       'SXM', 'SVK', 'SVN', 'SLB', 'SOM', 'ZAF', 'SSD', 'ESP', 'LKA','SDN', 'SUR', 'SWZ', 'SWE', 'CHE', 'SYR', 'TWN', 'TJK', 'TZA',
       'THA', 'TLS', 'TGO', 'TON', 'TTO', 'TUN', 'TUR', 'TKM', 'TUV','UGA', 'UKR', 'ARE', 'GBR', 'USA', 'URY', 'UZB', 'VUT', 'VEN',
       'VNM', 'VGB', 'WBG', 'YEM', 'ZMB', 'ZWE']) #Country Codes
print(l2)


# ### Creating a DataFrame using these two lists

# In[45]:


df=pd.DataFrame(l1,l2)
df.reset_index(inplace=True)
df.columns=['Code','Country']
print(df)


# ### Merging our DataFrame with our original Data

# In[46]:


temp_country1=pd.merge(temp_country1,df,on='Country')
temp_country1.dropna(inplace=True)
print(temp_country1)


# ### Now, that our necessary data transformation has been carried out, let us plot it on an interactive map

# In[47]:


data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'RdYlGn',
        reversescale = True,
        showscale = True,
        locations = temp_country1['Code'],
        z = temp_country1['AverageTemperature'],
        locationmode = 'Code',
        text = temp_country1['Country'].unique(),
        marker = dict(
            line = dict(color = 'rgb(200,200,200)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Temperature')
            )
       ]
print(data)


# In[48]:


layout = dict(
    title = 'Average Temperature By Country',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(0,255,255)',
        projection = dict(
        type = 'Mercator',
            
        ),
            ),
        )
print(layout)


# In[49]:


fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmap2010')


# ### From the above map, we can easily visualize the average temperature of the countries

# ### Now we will figure out the coldest and the hottest countries and represent them

# In[50]:


hot=temp_country1.sort_values(by='AverageTemperature',ascending=False)[:10]
cold=temp_country1.sort_values(by='AverageTemperature',ascending=True)[:10]
countries=pd.concat([hot,cold])
countries.sort_values('AverageTemperature',ascending=False,inplace=True)


# In[51]:


f,ax=plt.subplots(figsize=(12,8))
sns.barplot(y='Country',x='AverageTemperature',data=countries,palette='cubehelix',ax=ax).set_title('The Hottest And Coldest Countries')
plt.xlabel('Mean Temperture')
plt.ylabel('Country')


# ### Now, let us look at the trends and patterns for temperature variation in the biggest economies in the world

# In[53]:


countries=temp_country.copy()
countries['dt']=pd.to_datetime(countries.dt).dt.strftime('%d/%m/%Y')
countries['dt']=countries['dt'].apply(lambda x: x[6:])
countries=countries[countries['AverageTemperature']!=0]
countries.drop('AverageTemperatureUncertainty',axis=1,inplace=True)
print(countries)


# In[54]:


li=['United States','China','India','Japan','Germany','United Kingdom']
countries=countries[countries['Country'].isin(li)]
countries=countries.groupby(['Country','dt'])['AverageTemperature'].mean().reset_index()
countries=countries[countries['dt'].astype(int)>1850]
print(countries)


# In[55]:


abc=countries.pivot('dt','Country','AverageTemperature')
f,ax=plt.subplots(figsize=(20,10))
abc.plot(ax=ax)


# ### The above graph clearly illustrates the temperature trends of USA, UK, Germany, Japan, China and India

# ### Let us now find out which countries have the maximum temperature differences between their average maximum and minimum temperatures

# In[87]:


try1=temp_country.copy()
try1['dt']=try1['dt'].apply(lambda x:x[6:])
print(try1)


# In[88]:


try2=try1[try1['dt']>'1850'].groupby('Country')['AverageTemperature'].max().reset_index()
try2.columns=[['Country','Avg_Temp_Max']]
print(try2)


# In[89]:


try3=try1[try1['dt']>'1850'].groupby('Country')['AverageTemperature'].min().reset_index()
try3.columns=[['Country','Avg_Temp_Min']]
print(try3)


# In[91]:


try2['Avg_Temp_Min']=try3['Avg_Temp_Min']


# In[92]:


print(try2)


# In[95]:


try2.columns=['Country','Avg_Temp_Max','Avg_Temp_Min']
try2['Difference'] = try2['Avg_Temp_Max'] - try2['Avg_Temp_Min']
print(try2)


# In[96]:


try2=try2.sort_values(by='Difference',ascending=False)


# In[97]:


sns.barplot(x='Difference',y='Country',data=try2[:10],palette='RdYlGn').set_title('Countries with Highest Difference between Max And Min Temperture')
plt.xlabel('Temperature Difference')


# ### The above bar graph illustrates the countries with the highest differences between Maximum and Minimum Temperatures

# ### Illustrating Temperature Difference by Country on an Interactive Map

# In[106]:


try2=try2.merge(df,left_on='Country',right_on='Country',how='left')
try2.dropna(inplace=True)


# In[107]:


data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'Viridis',
        reversescale = True,
        showscale = True,
        locations = try2['Code'],
        z = try2['Difference'],
        locationmode = 'Code',
        text = try2['Country'].unique(),
        marker = dict(
            line = dict(color = 'rgb(200,200,200)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Temperature Difference')
            )
       ]
print(data)


# In[108]:


layout = dict(
    title = 'Temperature Difference By Country',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(0,255,255)',
        projection = dict(
        type = 'Mercator',
            
        ),
            ),
        )
print(layout)


# In[109]:


fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmap2010')


# ### The above map shows the difference between the maximum and minimum temperatures for each country.

# # Temperature Variation by States

# ### Reading the Data Set and Cleaning the Data

# In[110]:


states=pd.read_csv('C:/Users/ANANT SHARMA .LAPTOP-FFV0CT73/Desktop/DS+ML/Project ITS/GlobalLandTemperaturesByState.csv')
states.dropna(inplace=True)
states['dt']=pd.to_datetime(states.dt).dt.strftime('%d/%m/%Y')


# ### Visualizing the Data

# In[111]:


f,ax=plt.subplots(figsize=(15,8))
top_states=states.groupby(['State','Country'])['AverageTemperature'].mean().reset_index().sort_values(by='AverageTemperature',ascending=False)
top_states=top_states.drop_duplicates(subset='Country',keep='first')
top_states.set_index('Country',inplace=True)
top_states['AverageTemperature']=top_states['AverageTemperature'].round(decimals=2)
top_states.plot.barh(width=0.8,color='#0154ff',ax=ax)
for i, p in enumerate(zip(top_states.State, top_states['AverageTemperature'])):
    plt.text(s=p,x=1,y=i,fontweight='bold',color='white')


# ### Visualizing Temperature Trends in States

# In[112]:


top_states1=states.copy()
top_states1['dt']=top_states1['dt'].apply(lambda x:x[6:])
top_states1=top_states1[top_states1['State'].isin(list(top_states.State))]
top_states1=top_states1.groupby(['State','dt'])['AverageTemperature'].mean().reset_index()
top_states1=top_states1[top_states1['dt'].astype(int)>1900]
f,ax=plt.subplots(figsize=(18,8))
top_states1.pivot('dt','State','AverageTemperature').plot(ax=ax)
plt.xlabel('Year')


# ### Illustrating Temperatures for States in the USA

# ### Cleaning the Data

# In[113]:


USA=states[states['Country']=='United States']
USA.dropna(inplace=True)
USA['State'].replace({'Georgia (State)':'Georgia','District Of Columbia':'Columbia'},inplace=True)
USA=USA[['AverageTemperature','State']]
USA=USA.groupby('State')['AverageTemperature'].mean().reset_index()
print(USA)


# ### Transforming Data

# In[122]:


du=['Alabama', 'AL','Alaska', 'AK','Arizona', 'AZ','Arkansas', 'AR','California', 'CA','Colorado', 'CO'
,'Columbia', 'DC','Connecticut', 'CT','Delaware', 'DE','Florida', 'FL','Georgia', 'GA','Hawaii', 'HI'
,'Idaho', 'ID','Illinois', 'IL','Indiana', 'IN','Iowa', 'IA','Kansas', 'KS','Kentucky', 'KY'
,'Louisiana', 'LA','Maine', 'ME','Maryland', 'MD','Massachusetts', 'MA','Michigan', 'MI','Minnesota', 'MN'
,'Mississippi', 'MS','Missouri', 'MO','Montana', 'MT','Nebraska', 'NE','Nevada', 'NV','New Hampshire', 'NH','New Jersey', 'NJ'
,'New Mexico', 'NM','New York', 'NY','North Carolina', 'NC','North Dakota', 'ND'
,'Ohio', 'OH','Oklahoma', 'OK','Oregon', 'OR','Pennsylvania', 'PA','Rhode Island', 'RI'
,'South Carolina', 'SC','South Dakota', 'SD','Tennessee', 'TN','Texas', 'TX'
,'Utah', 'UT','Vermont', 'VT','Virginia', 'VA','Washington', 'WA'
,'West Virginia', 'WV','Wisconsin', 'WI','Wyoming', 'WY']
print(du)


# In[123]:


code=du[1::2]
del du[1::2]


# In[124]:


usa=pd.DataFrame(du,code)
usa.reset_index(inplace=True)
usa.columns=['Code','State']
print(usa)


# In[125]:


USA=pd.merge(USA,usa,on='State')


# In[126]:


print(USA)


# ### Data Visualization

# In[127]:


data = [ dict(
        type='choropleth',
        colorscale = 'Viridis',
        autocolorscale = False,
        locations = USA['Code'],
        z = USA['AverageTemperature'].astype(float),
        locationmode = 'USA-states',
        text =USA['State'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Average Temperature")
        ) ]
print(data)


# In[128]:


layout = dict(
        title = 'Average Temperature for USA States',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
print(layout)


# In[129]:


fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )


# ### The above map clearly illustrates Average Temperatures for all the states in USA

# # Temperature Data by Cities

# ### Reading the Dataset and Cleaning the Data

# In[130]:


cities=pd.read_csv('C:/Users/ANANT SHARMA .LAPTOP-FFV0CT73/Desktop/DS+ML/Project ITS/GlobalLandTemperaturesByCity.csv')
cities.dropna(inplace=True)
cities['year']=cities['dt'].apply(lambda x: x[:4])
cities['month']=cities['dt'].apply(lambda x: x[5:7])
cities.drop('dt',axis=1,inplace=True)
cities=cities[['year','month','AverageTemperature','City','Country','Latitude','Longitude']]
cities['Latitude']=cities['Latitude'].str.strip('N')
cities['Longitude']=cities['Longitude'].str.strip('E')
cities.head()


# ## Visualizing the Hottest Cities by Country

# In[131]:


temp_city=cities.groupby(['City','Country'])['AverageTemperature'].mean().reset_index().sort_values(by='AverageTemperature',ascending=False)
temp_city=temp_city.drop_duplicates(subset='Country',keep='first')
temp_city=temp_city.set_index(['City','Country'])
plt.subplots(figsize=(8,30))
sns.barplot(y=temp_city.index,x='AverageTemperature',data=temp_city,palette='RdYlGn').set_title('Hottest Cities By Country')
plt.xlabel('Average Temperature')


# ## Analyzing and Visualizing Temperature Data for Indian Cities

# ### Working with major Indian Cities' Temperature Data

# In[132]:


indian_cities=cities[cities['Country']=='India']
indian_cities=indian_cities[indian_cities['year']>'1850']
major_cities=indian_cities[indian_cities['City'].isin(['Mumbai','New Delhi','Bangalore','Hyderabad','Calcutta','Pune','Madras','Ahmedabad'])]
heatmap=major_cities.groupby(['City','month'])['AverageTemperature'].mean().reset_index()
trace = go.Heatmap(z=heatmap['AverageTemperature'],
                   x=heatmap['month'],
                   y=heatmap['City'],
                  colorscale='Viridis')
data=[trace]
layout = go.Layout(
    title='Average Temperature Of Major Cities By Month',
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')


# ### The above heatmap visualization clearly illustrates the temperature variation of various major Indian Cities over different months of the year

# ## Now, we will look at temperature change trends in these major cities over the years and visualize it.

# In[133]:


graph=major_cities[major_cities['year']>'1900']
graph=graph.groupby(['City','year'])['AverageTemperature'].mean().reset_index()
graph=graph.pivot('year','City','AverageTemperature').fillna(0)
graph.plot()
fig=plt.gcf()
fig.set_size_inches(18,8)


# ### From the above line graph it is evident that even with fluctuations in temperature over the years, in general the temperature of the major indian cities has been gradually increasing over the years, as seen from the subtle upward trend of the line plots.

# ## Visualizing New Delhi Climate Change Data

# ### Reading the Dataset

# In[137]:


data = pd.read_csv('C:/Users/ANANT SHARMA .LAPTOP-FFV0CT73/Desktop/DS+ML/Project ITS/testset.csv',parse_dates=['datetime_utc'],skipinitialspace=True)


# In[139]:


print(len(data))
data.columns


# ### Transforming the Data

# In[140]:


data['Date'] = pd.to_datetime(data['datetime_utc'])
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
data['hour'] = data['Date'].dt.hour


# ### Visualizing Data

# In[141]:


year_humi = data.groupby(data.year).mean()
pd.stats.moments.ewma(year_humi._hum, 5).plot()
year_humi._hum.plot(linewidth=1)
plt.title('Delhi Average Humidity by Year')
plt.xlabel('Year')


# In[142]:


year_heat = data.groupby(data.year).mean()
pd.stats.moments.ewma(year_heat._heatindexm , 5).plot()
year_heat._heatindexm .plot(linewidth=1)
plt.title('Delhi Average Heat by Year')
plt.xlabel('Year')


# In[143]:


year_rain = data.groupby(data.year).mean()
pd.stats.moments.ewma(year_rain._rain, 5).plot()
year_rain._rain.plot(linewidth=1)
plt.title('Delhi Average Rain by Year')
plt.xlabel('Year')


# ### The above graphs illustrate how the different climate trends such as Humidity, Rain and Heat have varied over the years

# # Time Series Analysis

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
import matplotlib.patches as mpatches
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os


# In[8]:


data = pd.read_csv('C:/Users/ANANT SHARMA .LAPTOP-FFV0CT73/Desktop/DS+ML/Project ITS/testset.csv')


# In[9]:


data['datetime_utc'] = pd.to_datetime(data['datetime_utc'])
data.set_index('datetime_utc', inplace= True)
data =data.resample('D').mean()


# In[10]:


data = data[[' _tempm' ]]


# In[11]:


data[' _tempm'].fillna(data[' _tempm'].mean(), inplace=True)


# In[12]:


plt.figure(figsize=(20,8))
plt.plot(data)
plt.title('Time Series')
plt.xlabel('Date')
plt.ylabel('temperature')
plt.show()


# # Time Series Forecast using LSTM
# ### Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in following work.1 They work tremendously well on a large variety of problems, and are now widely used.
# 
# ### LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!
# 
# ### All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.

# In[13]:


data=data.values
data = data.astype('float32')


# In[14]:


scaler= MinMaxScaler(feature_range=(-1,1))
sc = scaler.fit_transform(data)


# In[15]:


timestep = 30

X= []
Y=[]


for i in range(len(sc)- (timestep)):
    X.append(sc[i:i+timestep])
    Y.append(sc[i+timestep])


X=np.asanyarray(X)
Y=np.asanyarray(Y)


k = 7300
Xtrain = X[:k,:,:]
Xtest = X[k:,:,:]    
Ytrain = Y[:k]    
Ytest= Y[k:]


# In[16]:


print(Xtrain.shape)
print(Xtest.shape)


# ## CNN LSTM Model

# In[5]:


from keras.layers import Dense,RepeatVector
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# In[17]:


model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(30,1)))
model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(RepeatVector(30))
model.add(LSTM(128, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(Xtrain,Ytrain,epochs=100, verbose=0 )


# In[18]:


preds_cnn1 = model.predict(Xtest)
preds_cnn1 = scaler.inverse_transform(preds_cnn1)


Ytest=np.asanyarray(Ytest)  
Ytest=Ytest.reshape(-1,1) 
Ytest = scaler.inverse_transform(Ytest)


Ytrain=np.asanyarray(Ytrain)  
Ytrain=Ytrain.reshape(-1,1) 
Ytrain = scaler.inverse_transform(Ytrain)

mean_squared_error(Ytest,preds_cnn1)


# In[19]:


plt.figure(figsize=(20,9))
plt.plot(Ytest , 'blue', linewidth=5)
plt.plot(preds_cnn1,'r' , linewidth=4)
plt.legend(('Test','Predicted'))
plt.show()


# In[20]:


def insert_end(Xin,new_input):
    for i in range(timestep-1):
        Xin[:,i,:] = Xin[:,i+1,:]
    Xin[:,timestep-1,:] = new_input
    return Xin


# In[21]:


first =0   # this section for unknown future 
future=200
forcast_cnn = []
Xin = Xtest[first:first+1,:,:]
for i in range(future):
    out = model.predict(Xin, batch_size=1)    
    forcast_cnn.append(out[0,0]) 
    Xin = insert_end(Xin,out[0,0]) 


# In[22]:


forcasted_output_cnn=np.asanyarray(forcast_cnn)   
forcasted_output_cnn=forcasted_output_cnn.reshape(-1,1) 
forcasted_output_cnn = scaler.inverse_transform(forcasted_output_cnn)


# In[23]:


plt.figure(figsize=(16,9))
plt.plot(Ytest , 'black', linewidth=4)
plt.plot(forcasted_output_cnn,'r' , linewidth=4)
plt.legend(('test','Forcasted'))
plt.show()

