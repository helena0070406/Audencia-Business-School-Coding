# Audencia Finance Case Study about Blue Corporation 
# Yinli ZHANG/ Haiwei FU

##################################################################################################################################################
# import libraries and database.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataAus = pd.read_csv('C:/Users/ASUS/Desktop/Case Study - Data- Blue Corporation Revenue/Australia.csv', parse_dates=['Date'], index_col='Date')
dataCan = pd.read_csv('C:/Users/ASUS/Desktop/Case Study - Data- Blue Corporation Revenue/Canada.csv', parse_dates=['Date'], index_col='Date')
dataGer = pd.read_csv('C:/Users/ASUS/Desktop/Case Study - Data- Blue Corporation Revenue/Germany.csv', parse_dates=['Date'], index_col='Date') 
dataJap = pd.read_csv('C:/Users/ASUS/Desktop/Case Study - Data- Blue Corporation Revenue/Japan.csv', parse_dates=['Date'], index_col='Date')
dataMex = pd.read_csv('C:/Users/ASUS/Desktop/Case Study - Data- Blue Corporation Revenue/Mexico.csv', parse_dates=['Date'], index_col='Date')
dataNig = pd.read_csv('C:/Users/ASUS/Desktop/Case Study - Data- Blue Corporation Revenue/Nigeria.csv', parse_dates=['Date'], index_col='Date')
dataSal = pd.read_csv('C:/Users/ASUS/Desktop/Case Study - Data- Blue Corporation Revenue/sales.csv', parse_dates=['Date'], index_col='Date')

dataAus.head(3)

# the six countries database together.
mydata = pd.concat([dataAus, dataCan, dataGer, dataJap, dataMex, dataNig])
print(mydata.head(5))
      
# 1.the total renevue ranks divided by countries.
print('the total renevue ranks divided by countries.')
print(mydata.sort_values('Revenue', ascending=False).groupby('Country').sum().head(3))


# 2.select year from 2011 to 2012
data20112012 = mydata[mydata.index.year >= 2011]
data20112012 = data20112012[data20112012.index.year <= 2012]
data20112012.head(10)

# 2.return for revenue
returns = data20112012['Revenue'].pct_change()

# 2.reture for revenue by countries.
print('reture for revenue by countries.')
print(returns.groupby(data20112012['Country']).sum().sort_values(ascending=False))


# 3.select the year from 2011 to 2018
data20112018 = mydata[mydata.index.year >= 2011]
data20112018 = data20112018[data20112018.index.year <= 2018]
data20112018.head(5)

returns03 = data20112018['Revenue'].pct_change()

# 3.cumulative return calculate (use returns03)
cumumlative_returns = (1 + returns03).cumprod() - 1
print(cumumlative_returns)

data03 = cumumlative_returns.groupby(data20112018['Country']).sum()
print('Which countries have the most important cumulative revenue growth (in %) between 2011 to 2018?')
print(data03)

# 4.list five best-selling products.
# 4.1 rank sales unites
print('list five best-selling products.')
print(mydata.groupby('ProductID').count().sort_values(by='ProductID', ascending=False).head(5))
# 4.2 rank revenue
the_top_5=mydata['Revenue'].groupby(mydata['ProductID'] ).sum()
the_top_5_1=the_top_5.reset_index().sort_values('Revenue',ascending=False).set_index('ProductID')
the_top_5_1.head()

# 5.list five best-selling products in Japan.
# 5.1 rank sales unites
print('list five best-selling products in Japan.')
print(dataJap.groupby('ProductID').count().sort_values(by='ProductID', ascending=False).head(5))

# 5.2 rank revenue
the_top_5_japan=dataJap['Revenue'].groupby(dataJap['ProductID'] ).sum()
the_top_5_japan_1=the_top_5_japan.reset_index().sort_values('Revenue',ascending=False).set_index('ProductID')
the_top_5_japan_1.head()

# 6.the most expensive product.
# print('the most expensive product.')
# print(mydata.sort_values(by='Revenue', ascending=False).head(1))a

mydata['unit price']=mydata['Revenue']/mydata['Units']
mydata.head()
mydata.max()


# 7.the biggest [manufacturer] contributor to revenue in the world.
data_GEO = pd.read_excel('C:/Users/ASUS/Desktop/Case Study - Data- Blue Corporation Revenue/MDM_GEO.xlsx')
data_MAN = pd.read_excel('C:/Users/ASUS/Desktop/Case Study - Data- Blue Corporation Revenue/MDM_MANUFACTURER.xlsx')
data_PRO = pd.read_excel('C:/Users/ASUS/Desktop/Case Study - Data- Blue Corporation Revenue/MDM_PRODUCT.xlsx')

manufacturer_all = pd.merge(mydata,data_PRO,left_on='ProductID',right_on='ProductID',how='inner')
manufacturer_all_1 = manufacturer_all['Revenue'].groupby(manufacturer_all['ManufacturerID']).sum()
print('the ranking of manufacturer 1:\n',manufacturer_all_1.reset_index().sort_values('Revenue',ascending=False).set_index('ManufacturerID'))



# 8.the bottom 3 [manufacturer] contributor to revenue in Nigeria.
manufacturer_Nig = pd.merge(dataNig,data_PRO,left_on='ProductID',right_on='ProductID',how='inner')
manufacturer_Nig_1=manufacturer_Nig['Revenue'].groupby(manufacturer_Nig['ManufacturerID']).sum()
print('the ranking of manufacturer nig:\n',manufacturer_Nig_1.reset_index().sort_values('Revenue',ascending=False).set_index('ManufacturerID'))


# 9.the list of products with no revenue in 2017.
# 9.select the year and get ProductID
data2017 = mydata[mydata.index.year == 2017]
dataPoductid = mydata['ProductID'].drop_duplicates()

# 9.get all the item of ProductID in the year2017.
data2017Productid_revenue = data2017['ProductID'].drop_duplicates()
print(data2017Productid_revenue)
print(dataPoductid)       

for i in data2017Productid_revenue:
    if i not in dataPoductid:
        print(i)

# 10.the list of products with no revenue in 2017 & 2018.
data20172018 = mydata[mydata.index.year >= 2017]
data20172018 = data20172018[data20172018.index.year <= 2018]

dataPoductid = mydata['ProductID'].drop_duplicates()

data20172018Productid_revenue = data20172018['ProductID'].drop_duplicates()
print(data20172018Productid_revenue)
print(dataPoductid)

for i in data20172018Productid_revenue:
    if i not in dataPoductid:
        print(i)

# 11.list of the top 3 products with the most important growth (in value) between 2016 to 2017.
data2016 = mydata[mydata.index.year == 2016]
data2017 = data2016[data2016.index.year == 2017]

data2016 = data2016[['ProductID','Revenue']]
data2016.groupby(['ProductID']).sum()
data2016 = data2016.rename(columns={"Revenue": "Revenue_2016"})

data2017 = data2017[['ProductID','Revenue']]
data2017.groupby(['ProductID']).sum()
data2017 = data2017.rename(columns={"Revenue": "Revenue_2017"})

total_2016_2017 = pd.merge(data2016,data2017,left_on='ProductID',right_on='ProductID',how='inner')
total_2016_2017['growth_rate'] = (total_2016_2017['Revenue_2017']-total_2016_2017['Revenue_2016'])/total_2016_2017['Revenue_2016']
total_2016_2017 = total_2016_2017.sort_values(by='growth_rate',ascending=False)
total_2016_2017.head(3)

# 12.Is Blue Corporate a seasonal business? (Y/N), If yes: decompose this time series between seasonality & trend?
mydata.Revenue.plot()
plt.show()

# 13.Provide revenue of USA & Canada in January 2015.
dataCan201501 = dataCan[dataCan.index.year == 2015]
dataCan201501 = dataCan201501[dataCan201501.index.month == 0o1]
print(dataCan201501)

dataCan201501_renevue = dataCan201501['Revenue'].sum()
print(dataCan201501_renevue)

# sales.csv means usa.csv
dataSal201501 = dataSal[dataSal.index.year == 2015]
dataSal201501 = dataSal201501[dataSal201501.index.month == 0o1]
print(dataSal201501)

dataSal201501_renevue = dataSal201501['Revenue'].sum()
print(dataSal201501_renevue)   

# 14.In USA the top 3 states contributor to revenue.
dataSalstates_revenue = dataSal.groupby(dataSal['Zip']).count().sort_values(by='Revenue', ascending=False)
dataSalstates_revenue.head(3)

# 15.In USA the region with the most important share of revenue.
dataSalstates_revenue_most = dataSal.groupby(dataSal['Zip']).count().sort_values(by='Revenue', ascending=False)
dataSalstates_revenue_most.head(3)

# 16.In USA for 2014, the top 3 states with the most important volume of unit.
dataSal2014 = dataSal[dataSal.index.year == 2014]
dataSal2014_unit = dataSal2014.groupby(dataSal2014['Zip'])
dataSal2014_unit.sum().head(3)

# 17.In Canada for 2017, the state with the most important volume of unit and revenue.
dataCan2017 = dataCan[dataCan.index.year == 2017]

dataCan2017_unitrevenue1 = dataCan2017.groupby(dataCan2017['Zip']).count().sort_values(by='Revenue', ascending=False)
dataCan2017_unitrevenue1.head(3)
dataCan2017_unitrevenue2 = dataCan2017_unitrevenue1.sort_values(by='Units', ascending=False)
dataCan2017_unitrevenue2.head(1)


#########################################################################################################################################


# 18.Mockup about 2018 choose one country  Key Message
# we choose USA
dataSal2018=dataSal[dataSal.index.year==2018]
dataSal2018_Revenue=dataSal2018['Revenue'].sum()
print(dataSal2018_Revenue)


the_top_5_usa=dataSal2018['Revenue'].groupby(dataSal2018['ProductID'] ).sum()
the_top_5_usa_1=the_top_5_usa.reset_index().sort_values('Revenue',ascending=False).set_index('ProductID')
the_top_5_usa_1.head()
the_top_5_usa_1.tail()

state2018 = pd.merge(dataSal2018,data_GEO,left_on='Zip',right_on='Zip',how='inner')
state2018_1=state2018[['Zip','Revenue','State']]
TOP3_USA=state2018_1['Revenue'].groupby(state2018_1['State']).sum()
print('the ranking of revenue in usa:\n',TOP3_USA.reset_index().sort_values('Revenue',ascending=False).set_index('State'))


region2018_1=state2018[['Zip','Revenue','Region']]
regaion_revenue2018=region2018_1['Revenue'].groupby(region2018_1['Region']).sum()
print(regaion_revenue2018)


usa = pd.read_csv('C:/Users/ASUS/Desktop/Case Study - Data- Blue Corporation Revenue/sales.csv')
usa_1 = pd.merge(usa,data_GEO,left_on='Zip',right_on='Zip',how='inner')

df_usa_2 = usa_1[(usa_1['Date'] >= '2018-01-01')&(usa_1['Date'] <= '2018-12-31')]
unin_usa=df_usa_2[['Units','State']]
USA2018=unin_usa['Units'].groupby(unin_usa['State']).sum()
print('the ranking of units:\n',USA2018.reset_index().sort_values('Units',ascending=False).set_index('State'))

#########################################################################################################################################

# Pridict use ARIMA Model.
# 19.WW forecast for 2019 using ARIMA Model (Simple Model).
# pm predict
from statsmodels.tsa.arima.model import ARIMA


# fit the data element of revenue.
data_pridct = mydata[mydata.index.year == 2018]
data_pridct = data_pridct[data_pridct.index.month == 12]
data_pridct = data_pridct[data_pridct.index.day >= 25]   # the data is so huge, we use 7 days, but still very big.
data_pridct.head(5)

model = ARIMA(data_pridct[['Revenue']])
model_fit = model.fit()

print(model_fit.summary())


# get the parameters and forecast.
arima_prediction_D = model_fit.forecast(7)



import pmdarima as pm


# auto_arima() uses a stepwise approach to search multiple combinations of p,d,q parameters
model = pm.auto_arima(data_pridct[['Revenue']], start_p=1, start_q=1,
                      test='adf',  # use adftest to find optimal 'd'
                      max_p=3, max_q=3,  # maximum p and q
                      m=12,  # frequency of series - number of periods
                      d=None,  # let model determine 'd', The order of first-differencing
                      seasonal=True,  # Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

print(model.summary())
# get the parameters and forecast values
# return forecast and attached confidence intervals (ignore the latter at this stage)
fc, confint = model.predict(n_periods=7, return_conf_int=True)
index_of_fc = pd.date_range(start='1/1/2019', end='1/7/2019', freq='D') # we experience the date of range is for 7 days, the data so huge.
print(index_of_fc) 
arima_prediction_E = pd.DataFrame(fc, index=index_of_fc, columns=['Revenue'])

arima_prediction_E['Revenue'].plot()
plt.legend()
plt.title('Revenue predict')
plt.ylabel('Time')
plt.xlabel('Revenue')
plt.show()



# 19.WW forecast for 2019 using ARIMA Model (Simple Model)， no perameter p,d,q.
import statsmodels.api as sm

while True:
    path = input('C:/Users/ASUS/Desktop/Case Study - Data- Blue Corporation Revenue/sales.csv')
    name = input("USA_year_2019_revenue_predict")
    data = pd.read_csv(path, parse_dates=['Date'], index_col='Date')

    data_month = dataSal['Revenue'].resample('W-MON').mean()
    data_train = data_month['2011':'2018']

    model = sm.tsa.arima.ARIMA(data_train, order=(1, 0, 30), freq='W-MON')
    result = model.fit()

    pred = result.predict('20190101', '20191231', dynamic=True, typ='levels')
    plt.figure(figsize=(6, 6))
    plt.xticks(rotation=45)
    plt.title('{} Revenue'.format(name))
    plt.plot(pred)
    plt.plot(data_train)
    plt.show()
