'''
Thesis: The outer factors on sustainability-linked bonds
(Thesis python coding)


Audencia Busisness School
MSc in Data management for finance 
By. Haiwei FU
'''
# import libraries.
import pandas as pd
import numpy as np
import seaborn as sns
import eikon as ek
import matplotlib.pyplot as plt
import yfinance as yf
from yahoofinancials import YahooFinancials

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import learning_curve 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree


# open Rifinitiv workplace and Refinitiv Eikon website
# open Yahoo Finance website.

# use eikon API to get the data
ek.set_app_key('6a9c59fdeb3145059ebc8ce7c0d3847b8bcf9deb') 


# get 39 SLBs which published in Luxemburg Stock Exchange before this year. 
df1, err = ek.get_data('XS2239845253',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df2, err = ek.get_data('XS2239845097',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df3, err = ek.get_data('XS2261215011',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df4, err = ek.get_data('USA35155AE99',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df5, err = ek.get_data('US49836AAC80',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})

df6, err = ek.get_data('XS2320746394',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df7, err = ek.get_data('XS2326548562',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df8, err = ek.get_data('XS2326550899',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df9, err = ek.get_data('DE000BHYOSL9',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df10, err = ek.get_data('XS2332306344',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})

df11, err = ek.get_data('XS2338570331',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df12, err = ek.get_data('FR0014003GX7',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df13, err = ek.get_data('XS2335148701',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df14, err = ek.get_data('XS2335148024',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df15, err = ek.get_data('XS2344735811',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})

df16, err = ek.get_data('XS2361358299',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df17, err = ek.get_data('XS2361358539',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df18, err = ek.get_data('XS2364001078',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df19, err = ek.get_data('XS2382209125',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df20, err = ek.get_data('XS2358383896',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})

df21, err = ek.get_data('XS2358383466',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df22, err = ek.get_data('XS2389120325',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df23, err = ek.get_data('XS2389112736',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df24, err = ek.get_data('XS2389984175',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df25, err = ek.get_data('XS2389986899',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})

df26, err = ek.get_data('US79911QAA22',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df27, err = ek.get_data('USP84527AA17',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df28, err = ek.get_data('XS2399933386',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df29, err = ek.get_data('XS2364001078',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df30, err = ek.get_data('USP56226AV89',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})

df31, err = ek.get_data('XS2403428472',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df32, err = ek.get_data('USP3R12FAC46',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df33, err = ek.get_data('USP7S81YAC93',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df34, err = ek.get_data('US68560EAB48',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df35, err = ek.get_data('USP56226AV89',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})

df36, err = ek.get_data('USL79090AD51',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df37, err = ek.get_data('USP56226AV89',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df38, err = ek.get_data('XS2415386726',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})
df39, err = ek.get_data('XS2417499832',['TR.BIDPRICE.date','TR.BIDPRICE'],{"Sdate":'2022-01-01', "Edate":'2022-09-05'})






# show line graph visualization of SLBs in Luxsemburg.
fig = plt.subplots(figsize=(240, 120))


sns.lineplot(x='Date', y='Bid Price', data=df1)
sns.lineplot(x='Date', y='Bid Price', data=df2)
sns.lineplot(x='Date', y='Bid Price', data=df3)
sns.lineplot(x='Date', y='Bid Price', data=df4)
sns.lineplot(x='Date', y='Bid Price', data=df5)

sns.lineplot(x='Date', y='Bid Price', data=df6)
sns.lineplot(x='Date', y='Bid Price', data=df7)
sns.lineplot(x='Date', y='Bid Price', data=df8)
sns.lineplot(x='Date', y='Bid Price', data=df10)

sns.lineplot(x='Date', y='Bid Price', data=df11)
sns.lineplot(x='Date', y='Bid Price', data=df12)
sns.lineplot(x='Date', y='Bid Price', data=df13)
sns.lineplot(x='Date', y='Bid Price', data=df14)
sns.lineplot(x='Date', y='Bid Price', data=df15)

sns.lineplot(x='Date', y='Bid Price', data=df16)
sns.lineplot(x='Date', y='Bid Price', data=df17)
sns.lineplot(x='Date', y='Bid Price', data=df18)
sns.lineplot(x='Date', y='Bid Price', data=df19)
sns.lineplot(x='Date', y='Bid Price', data=df20)

sns.lineplot(x='Date', y='Bid Price', data=df21)
sns.lineplot(x='Date', y='Bid Price', data=df22)
sns.lineplot(x='Date', y='Bid Price', data=df23)
sns.lineplot(x='Date', y='Bid Price', data=df24)
sns.lineplot(x='Date', y='Bid Price', data=df25)

sns.lineplot(x='Date', y='Bid Price', data=df26)
sns.lineplot(x='Date', y='Bid Price', data=df27)
sns.lineplot(x='Date', y='Bid Price', data=df28)
sns.lineplot(x='Date', y='Bid Price', data=df29)

sns.lineplot(x='Date', y='Bid Price', data=df31)
sns.lineplot(x='Date', y='Bid Price', data=df32)
sns.lineplot(x='Date', y='Bid Price', data=df33)
sns.lineplot(x='Date', y='Bid Price', data=df34)
sns.lineplot(x='Date', y='Bid Price', data=df36)
sns.lineplot(x='Date', y='Bid Price', data=df38)
sns.lineplot(x='Date', y='Bid Price', data=df39)


plt.xticks(size=6, rotation=90)
plt.ylabel('Bid Price')
plt.title('SLBs Bid Price in Luxemburg Stock Exchange')
plt.show()


###########################################################################################################################################################
###########################################################################################################################################################
# get infor and data at Yahoo Finance
# get Brent crude oil price(USD), ticker is "CL=F"
yahoo_financials = YahooFinancials('CL=F')
data1 = yahoo_financials.get_historical_price_data(start_date='2022-01-01', 
                                                  end_date='2022-09-05', 
                                                  time_interval='daily')
oil_df = pd.DataFrame(data1['CL=F']['prices'])
oil_df = oil_df.drop('date', axis=1).set_index('formatted_date')
oil_df.head()

# visualize Brent crude oil price
oil_df['close'].plot(title='Brent crude oil price')


###########################################################################################################################################################
# get gas price(USD), ticker is "NG=F"
yahoo_financials = YahooFinancials('NG=F')
data2 = yahoo_financials.get_historical_price_data(start_date='2022-01-01', 
                                                  end_date='2022-09-05', 
                                                  time_interval='daily')
gas_df = pd.DataFrame(data2['NG=F']['prices'])
gas_df = gas_df.drop('date', axis=1).set_index('formatted_date')
gas_df.head()

# visualize Gas price
gas_df['close'].plot(title='Natural gas price')

###########################################################################################################################################################
# get coal price(USD), ticker is "MTF=F"
yahoo_financials = YahooFinancials('MTF=F')
data3 = yahoo_financials.get_historical_price_data(start_date='2022-01-01', 
                                                  end_date='2022-09-05', 
                                                  time_interval='daily')
coal_df = pd.DataFrame(data3['MTF=F']['prices'])
coal_df = coal_df.drop('date', axis=1).set_index('formatted_date')
coal_df.head()

# visualize Coal price
coal_df['close'].plot(title='Coal price')

###########################################################################################################################################################
# import inflation consumer price in Europe data, Excel downloaded in the official website: Eurostat.
inflation = [5.1, 5.9, 7.4, 7.4, 8.1, 8.6, 8.9, 9.1]
y = inflation
x = [1, 2, 3, 4, 5, 6, 7 ,8]

# visualization inflation
x_label = ['2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08']
plt.title('Inflation in Europe')
plt.xticks(x, x_label)
plt.bar(x, y)
plt.show()

###########################################################################################################################################################
# import electricity price in France data, Excel downloaded in the official website: statista.
electricity = [211.58, 185.63, 295.17, 232.92, 197.46, 248.73, 400.95]
y = electricity
x = [1, 2, 3, 4, 5, 6, 7]

# visualization inflation
x_label = ['2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07']
plt.title('Electricity price in France')
plt.xticks(x, x_label)
plt.bar(x, y)
plt.show()

###########################################################################################################################################################
###########################################################################################################################################################
# SLBs Data processing
# create SLBs database 
database_slb = pd.DataFrame([df1['Bid Price'], df2['Bid Price'], df3['Bid Price'], df4['Bid Price'], df5['Bid Price'], df6['Bid Price'], df7['Bid Price'], 
                      df8['Bid Price'], df10['Bid Price'], df11['Bid Price'], df12['Bid Price'], df13['Bid Price'], df14['Bid Price'], 
                      df15['Bid Price'], df16['Bid Price'], df17['Bid Price'], df18['Bid Price'], df19['Bid Price'], df20['Bid Price'], df21['Bid Price'],
                      df22['Bid Price'], df23['Bid Price'], df24['Bid Price'], df25['Bid Price'], df26['Bid Price'], df27['Bid Price'], df28['Bid Price'],
                      df29['Bid Price'], df31['Bid Price'], df32['Bid Price'], df33['Bid Price'], df34['Bid Price'], df36['Bid Price'], df38['Bid Price'],
                      df39['Bid Price']]).T


database_slb.head(5)
database_slb.dropna()

############################################################################################################################################################
# SLBs Correlation with factors (Crude oil, Coal, Natural gas, Electricity, Inflation), I use regression OLS calcutation loss function this method 
# so I can get SLBs the almost influence weight of the factors.

# Oil price
y1 = database_slb
X = oil_df['close']

X, y1, true_coefficient = make_regression(n_samples=5800, n_features=160, n_informative=160, noise=0, coef=True, random_state=None)
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, train_size=0.7)
print(X_train.shape)
print(y1_train.shape)

# OLS
linear_regression = LinearRegression().fit(X_train, y1_train)
print("R^2 on training set: %f" % linear_regression.score(X_train, y1_train))      
print("R^2 on test set: %f" % linear_regression.score(X_test, y1_test))         

print(r2_score(np.dot(X_train, true_coefficient), y1_train))       

plt.figure(figsize=(12, 8))
coefficient_sorting = np.argsort(true_coefficient)[::1]
plt.plot(true_coefficient[coefficient_sorting], "o", label="true")
plt.plot(linear_regression.coef_[coefficient_sorting], "o", label="linear regression")

plt.legend()
plt.title('Oil fit graph')

def plot_learning_curve(est, X_train, y1_train):
    training_set_size, train_scores, test_scores = learning_curve(est, X_train, y1_train, train_sizes=np.linspace(.1, 1, 20))
    estimator_name = est.__class__.__name__
    line = plt.plot(training_set_size, train_scores.mean(axis=1), '--', label="training scores " + estimator_name)
    plt.plot(training_set_size, test_scores.mean(axis=1), '-', label="test scores " + estimator_name, c=line[0].get_color())
    plt.xlabel('Training set size')
    plt.legend(loc='best')
    plt.ylim(-0.1, 1.1)
    
    
plt.figure()    
plot_learning_curve(LinearRegression(), X_train, y1_train)


# ridge regression
ridge_models = {}
training_scores = []
test_scores = []
 
for alpha in [100, 10, 1, .01]:
    ridge = Ridge(alpha=alpha).fit(X_train, y1_train)
    training_scores.append(ridge.score(X_train, y1_train))
    test_scores.append(ridge.score(X_test, y1_test))
    ridge_models[alpha] = ridge

plt.figure()
plt.plot(training_scores, label="training scores")
plt.plot(test_scores, label="test scores")
plt.xticks(range(4), [100, 10, 1, .01])
plt.legend(loc="best")


plt.figure(figsize=(12, 8))
plt.plot(true_coefficient[coefficient_sorting], "o", label="true", c='b')

for i, alpha in enumerate([100, 10, 1, .01]):
    plt.plot(ridge_models[alpha].coef_[coefficient_sorting], "o", label="alpha = %.2f" % alpha, c=plt.cm.summer(i / 3.))
    
plt.legend(loc="best")

plt.figure()
plot_learning_curve(LinearRegression(), X_train, y1_train)
plot_learning_curve(Ridge(alpha=10), X_train, y1_train)

# LASSO
lasso_models = {}
training_scores = []
test_scores = []

# decide the alpha smaller, the hyper-perameter.
for alpha in [30, 10, 1, .01]:
    lasso = Lasso(alpha=alpha).fit(X_train, y1_train)
    training_scores.append(lasso.score(X_train, y1_train))
    test_scores.append(lasso.score(X_test, y1_test))
    lasso_models[alpha] = lasso
    
plt.figure()
plt.plot(training_scores, label="training scores")
plt.plot(test_scores, label="test scores")
plt.xticks(range(4), [30, 10, 1, .01])
plt.legend(loc="best")

plt.figure(figsize=(12, 8))
plt.plot(true_coefficient[coefficient_sorting], "o", label="true", c='b')

for i, alpha in enumerate([30, 10, 1, .01]):
    plt.plot(lasso_models[alpha].coef_[coefficient_sorting], "o", label="alpha = %.2f" % alpha, c=plt.cm.summer(i / 3.))
    
plt.legend(loc="best")


plt.figure()
plot_learning_curve(LinearRegression(), X_train, y1_train)
plot_learning_curve(Ridge(alpha=10), X_train, y1_train)
plot_learning_curve(Lasso(alpha=10), X_train, y1_train)

###############################################################################################################################################################
# Coal price
y2 = database_slb
X = coal_df['close']

X, y2, true_coefficient = make_regression(n_samples=5800, n_features=160, n_informative=160, noise=0, coef=True, random_state=None)
X_train, X_test, y2_train, y2_test = train_test_split(X, y2, train_size=0.7)
print(X_train.shape)
print(y2_train.shape)


# OLS
linear_regression = LinearRegression().fit(X_train, y2_train)
print("R^2 on training set: %f" % linear_regression.score(X_train, y2_train))     
print("R^2 on test set: %f" % linear_regression.score(X_test, y2_test))         

print(r2_score(np.dot(X_train, true_coefficient), y2_train))           

plt.figure(figsize=(12, 8))
coefficient_sorting = np.argsort(true_coefficient)[::1]
plt.plot(true_coefficient[coefficient_sorting], "o", label="true")
plt.plot(linear_regression.coef_[coefficient_sorting], "o", label="linear regression")

plt.title('Coal fit graph')
plt.legend()

def plot_learning_curve(est, X_train, y2_train):
    training_set_size, train_scores, test_scores = learning_curve(est, X_train, y2_train, train_sizes=np.linspace(.1, 1, 20))
    estimator_name = est.__class__.__name__
    line = plt.plot(training_set_size, train_scores.mean(axis=1), '--', label="training scores " + estimator_name)
    plt.plot(training_set_size, test_scores.mean(axis=1), '-', label="test scores " + estimator_name, c=line[0].get_color())
    plt.xlabel('Training set size')
    plt.legend(loc='best')
    plt.ylim(-0.1, 1.1)
    
plt.figure()    
plot_learning_curve(LinearRegression(), X_train, y2_train)


# ridge regression
ridge_models = {}
training_scores = []
test_scores = []
 
for alpha in [100, 10, 1, .01]:
    ridge = Ridge(alpha=alpha).fit(X_train, y2_train)
    training_scores.append(ridge.score(X_train, y2_train))
    test_scores.append(ridge.score(X_test, y2_test))
    ridge_models[alpha] = ridge

plt.figure()
plt.plot(training_scores, label="training scores")
plt.plot(test_scores, label="test scores")
plt.xticks(range(4), [100, 10, 1, .01])
plt.legend(loc="best")


plt.figure(figsize=(12, 8))
plt.plot(true_coefficient[coefficient_sorting], "o", label="true", c='b')

for i, alpha in enumerate([100, 10, 1, .01]):
    plt.plot(ridge_models[alpha].coef_[coefficient_sorting], "o", label="alpha = %.2f" % alpha, c=plt.cm.summer(i / 3.))
    
plt.legend(loc="best")



plt.figure()
plot_learning_curve(LinearRegression(), X_train, y2_train)
plot_learning_curve(Ridge(alpha=10), X_train, y2_train)



# LASSO
lasso_models = {}
training_scores = []
test_scores = []

# decide the alpha smaller, the hyper-perameter.
for alpha in [30, 10, 1, .01]:
    lasso = Lasso(alpha=alpha).fit(X_train, y2_train)
    training_scores.append(lasso.score(X_train, y2_train))
    test_scores.append(lasso.score(X_test, y2_test))
    lasso_models[alpha] = lasso
    
plt.figure()
plt.plot(training_scores, label="training scores")
plt.plot(test_scores, label="test scores")
plt.xticks(range(4), [30, 10, 1, .01])
plt.legend(loc="best")

plt.figure(figsize=(10, 5))
plt.plot(true_coefficient[coefficient_sorting], "o", label="true", c='b')

for i, alpha in enumerate([30, 10, 1, .01]):
    plt.plot(lasso_models[alpha].coef_[coefficient_sorting], "o", label="alpha = %.2f" % alpha, c=plt.cm.summer(i / 3.))
    
plt.legend(loc="best")


plt.figure()
plot_learning_curve(LinearRegression(), X_train, y2_train)
plot_learning_curve(Ridge(alpha=10), X_train, y2_train)
plot_learning_curve(Lasso(alpha=10), X_train, y2_train)

###############################################################################################################################################################
# Gas price
y3 = database_slb
X = gas_df['close']

X, y3, true_coefficient = make_regression(n_samples=5800, n_features=160, n_informative=160, noise=0, coef=True, random_state=None)
X_train, X_test, y3_train, y3_test = train_test_split(X, y3, train_size=0.7)
print(X_train.shape)
print(y3_train.shape)


# OLS
linear_regression = LinearRegression().fit(X_train, y3_train)
print("R^2 on training set: %f" % linear_regression.score(X_train, y3_train))    
print("R^2 on test set: %f" % linear_regression.score(X_test, y3_test))      


print(r2_score(np.dot(X_train, true_coefficient), y3_train))            
plt.figure(figsize=(12, 8))
coefficient_sorting = np.argsort(true_coefficient)[::1]
plt.plot(true_coefficient[coefficient_sorting], "o", label="true")
plt.plot(linear_regression.coef_[coefficient_sorting], "o", label="linear regression")

plt.title('Gas fit graph')
plt.legend()



def plot_learning_curve(est, X_train, y3_train):
    training_set_size, train_scores, test_scores = learning_curve(est, X_train, y3_train, train_sizes=np.linspace(.1, 1, 20))
    estimator_name = est.__class__.__name__
    line = plt.plot(training_set_size, train_scores.mean(axis=1), '--', label="training scores " + estimator_name)
    plt.plot(training_set_size, test_scores.mean(axis=1), '-', label="test scores " + estimator_name, c=line[0].get_color())
    plt.xlabel('Training set size')
    plt.legend(loc='best')
    plt.ylim(-0.1, 1.1)
    
plt.figure()    
plot_learning_curve(LinearRegression(), X_train, y3_train)


# ridge regression
ridge_models = {}
training_scores = []
test_scores = []
 
for alpha in [100, 10, 1, .01]:
    ridge = Ridge(alpha=alpha).fit(X_train, y3_train)
    training_scores.append(ridge.score(X_train, y3_train))
    test_scores.append(ridge.score(X_test, y3_test))
    ridge_models[alpha] = ridge

plt.figure()
plt.plot(training_scores, label="training scores")
plt.plot(test_scores, label="test scores")
plt.xticks(range(4), [100, 10, 1, .01])
plt.legend(loc="best")


plt.figure(figsize=(12, 8))
plt.plot(true_coefficient[coefficient_sorting], "o", label="true", c='b')

for i, alpha in enumerate([100, 10, 1, .01]):
    plt.plot(ridge_models[alpha].coef_[coefficient_sorting], "o", label="alpha = %.2f" % alpha, c=plt.cm.summer(i / 3.))
    
plt.legend(loc="best")



plt.figure()
plot_learning_curve(LinearRegression(), X_train, y3_train)
plot_learning_curve(Ridge(alpha=10), X_train, y3_train)



# LASSO
lasso_models = {}
training_scores = []
test_scores = []

# decide the alpha smaller, the hyper-perameter.
for alpha in [30, 10, 1, .01]:
    lasso = Lasso(alpha=alpha).fit(X_train, y3_train)
    training_scores.append(lasso.score(X_train, y3_train))
    test_scores.append(lasso.score(X_test, y3_test))
    lasso_models[alpha] = lasso
    
plt.figure()
plt.plot(training_scores, label="training scores")
plt.plot(test_scores, label="test scores")
plt.xticks(range(4), [30, 10, 1, .01])
plt.legend(loc="best")

plt.figure(figsize=(10, 5))
plt.plot(true_coefficient[coefficient_sorting], "o", label="true", c='b')

for i, alpha in enumerate([30, 10, 1, .01]):
    plt.plot(lasso_models[alpha].coef_[coefficient_sorting], "o", label="alpha = %.2f" % alpha, c=plt.cm.summer(i / 3.))
    
plt.legend(loc="best")


plt.figure()
plot_learning_curve(LinearRegression(), X_train, y3_train)
plot_learning_curve(Ridge(alpha=10), X_train, y3_train)
plot_learning_curve(Lasso(alpha=10), X_train, y3_train)

##################################################################################################################################################################
##################################################################################################################################################################
# S&P 500 price from 2022-01-01 to 2022-09-05
yahoo_financials = YahooFinancials('^GSPC')
data4 = yahoo_financials.get_historical_price_data(start_date='2022-01-01', 
                                                  end_date='2022-09-05', 
                                                  time_interval='daily')

sp_df = pd.DataFrame(data4['^GSPC']['prices'])
sp_df.head(5)

# visualize Brent crude oil price
sns.lineplot(x='formatted_date', y='close', data=sp_df)
plt.xticks(size=7, rotation=90)
plt.ylabel('close Price')
plt.title('S&P 500 price')
plt.show()

##################################################################################################################################################################
# Dow Jones Industrial Average price from 2022-01-01 to 2022-09-05 ^DJI
yahoo_financials = YahooFinancials('^DJI')
data5 = yahoo_financials.get_historical_price_data(start_date='2022-01-01', 
                                                  end_date='2022-09-05', 
                                                  time_interval='daily')

dj_df = pd.DataFrame(data5['^DJI']['prices'])
dj_df.head(5)

# visualize Brent crude oil price
sns.lineplot(x='formatted_date', y='close', data=dj_df)
plt.xticks(size=7, rotation=90)
plt.ylabel('close Price')
plt.title('Dow Jones Industrial Average price')
plt.show()
##################################################################################################################################################################
##################################################################################################################################################################
# gold price
yahoo_financials = YahooFinancials('GC=F')
data6 = yahoo_financials.get_historical_price_data(start_date='2022-01-01', 
                                                  end_date='2022-09-05', 
                                                  time_interval='daily')
gold_df = pd.DataFrame(data6['GC=F']['prices'])
gold_df = gold_df.drop('date', axis=1).set_index('formatted_date')
gold_df.head()

# visualize Gas price
gold_df['close'].plot(title='Gold price')

###################################################################################################################################################################
# coffee price 
yahoo_financials = YahooFinancials('KC=F')
data7 = yahoo_financials.get_historical_price_data(start_date='2022-01-01', 
                                                  end_date='2022-09-05', 
                                                  time_interval='daily')
coffee_df = pd.DataFrame(data7['KC=F']['prices'])
coffee_df = coffee_df.drop('date', axis=1).set_index('formatted_date')
coffee_df.head()

# visualize Gas price
coffee_df['close'].plot(title='coffee price')

###################################################################################################################################################################
# Cocoa price 
yahoo_financials = YahooFinancials('CC=F')
data7 = yahoo_financials.get_historical_price_data(start_date='2022-01-01', 
                                                  end_date='2022-09-05', 
                                                  time_interval='daily')
Cocoa_df = pd.DataFrame(data7['CC=F']['prices'])
Cocoa_df = Cocoa_df.drop('date', axis=1).set_index('formatted_date')
Cocoa_df.head()

# visualize Gas price
Cocoa_df['close'].plot(title='Cocoa price')

##################################################################################################################################################################
##################################################################################################################################################################
# Gold price with SLBs weight
y1 = database_slb
X = gold_df['close']

X, y1, true_coefficient = make_regression(n_samples=5800, n_features=160, n_informative=160, noise=0, coef=True, random_state=None)
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, train_size=0.7)
print(X_train.shape)
print(y1_train.shape)

# OLS
linear_regression = LinearRegression().fit(X_train, y1_train)
print("R^2 on training set: %f" % linear_regression.score(X_train, y1_train))      
print("R^2 on test set: %f" % linear_regression.score(X_test, y1_test))         

print(r2_score(np.dot(X_train, true_coefficient), y1_train))       

plt.figure(figsize=(12, 8))
coefficient_sorting = np.argsort(true_coefficient)[::1]
plt.plot(true_coefficient[coefficient_sorting], "o", label="true")
plt.plot(linear_regression.coef_[coefficient_sorting], "o", label="linear regression")

plt.title('Gold fit graph')
plt.legend()

def plot_learning_curve(est, X_train, y1_train):
    training_set_size, train_scores, test_scores = learning_curve(est, X_train, y1_train, train_sizes=np.linspace(.1, 1, 20))
    estimator_name = est.__class__.__name__
    line = plt.plot(training_set_size, train_scores.mean(axis=1), '--', label="training scores " + estimator_name)
    plt.plot(training_set_size, test_scores.mean(axis=1), '-', label="test scores " + estimator_name, c=line[0].get_color())
    plt.xlabel('Training set size')
    plt.legend(loc='best')
    plt.ylim(-0.1, 1.1)
    
plt.figure()    
plot_learning_curve(LinearRegression(), X_train, y1_train)

###################################################################################################################################################################
# Coffee price with SLBs
y1 = database_slb
X = coffee_df['close']

X, y1, true_coefficient = make_regression(n_samples=5800, n_features=160, n_informative=160, noise=0, coef=True, random_state=None)
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, train_size=0.7)
print(X_train.shape)
print(y1_train.shape)

# OLS
linear_regression = LinearRegression().fit(X_train, y1_train)
print("R^2 on training set: %f" % linear_regression.score(X_train, y1_train))      
print("R^2 on test set: %f" % linear_regression.score(X_test, y1_test))         

print(r2_score(np.dot(X_train, true_coefficient), y1_train))       

plt.figure(figsize=(12, 8))
coefficient_sorting = np.argsort(true_coefficient)[::1]
plt.plot(true_coefficient[coefficient_sorting], "o", label="true")
plt.plot(linear_regression.coef_[coefficient_sorting], "o", label="linear regression")

plt.title('Coffee fit graph')
plt.legend()

def plot_learning_curve(est, X_train, y1_train):
    training_set_size, train_scores, test_scores = learning_curve(est, X_train, y1_train, train_sizes=np.linspace(.1, 1, 20))
    estimator_name = est.__class__.__name__
    line = plt.plot(training_set_size, train_scores.mean(axis=1), '--', label="training scores " + estimator_name)
    plt.plot(training_set_size, test_scores.mean(axis=1), '-', label="test scores " + estimator_name, c=line[0].get_color())
    plt.xlabel('Training set size')
    plt.legend(loc='best')
    plt.ylim(-0.1, 1.1)
    
plt.figure()    
plot_learning_curve(LinearRegression(), X_train, y1_train)

###################################################################################################################################################################
# Cocoa price with SLBs
y1 = database_slb
X = Cocoa_df['close']

X, y1, true_coefficient = make_regression(n_samples=5800, n_features=160, n_informative=160, noise=0, coef=True, random_state=None)
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, train_size=0.7)
print(X_train.shape)
print(y1_train.shape)

# OLS
linear_regression = LinearRegression().fit(X_train, y1_train)
print("R^2 on training set: %f" % linear_regression.score(X_train, y1_train))      
print("R^2 on test set: %f" % linear_regression.score(X_test, y1_test))         

print(r2_score(np.dot(X_train, true_coefficient), y1_train))       

plt.figure(figsize=(12, 8))
coefficient_sorting = np.argsort(true_coefficient)[::1]
plt.plot(true_coefficient[coefficient_sorting], "o", label="true")
plt.plot(linear_regression.coef_[coefficient_sorting], "o", label="linear regression")

plt.title('Cocoa fit graph')
plt.legend()

def plot_learning_curve(est, X_train, y1_train):
    training_set_size, train_scores, test_scores = learning_curve(est, X_train, y1_train, train_sizes=np.linspace(.1, 1, 20))
    estimator_name = est.__class__.__name__
    line = plt.plot(training_set_size, train_scores.mean(axis=1), '--', label="training scores " + estimator_name)
    plt.plot(training_set_size, test_scores.mean(axis=1), '-', label="test scores " + estimator_name, c=line[0].get_color())
    plt.xlabel('Training set size')
    plt.legend(loc='best')
    plt.ylim(-0.1, 1.1)
    
plt.figure()    
plot_learning_curve(LinearRegression(), X_train, y1_train)

###################################################################################################################################################################
###################################################################################################################################################################
# get other outer factors data
data_other = yf.download('^DJI SI=F BTC-USD ^IXIC BZ=F NG=F ZC=F ZS=F CC=F KC=F CT=F ^N225 ^HSI', 
                   start="2022-01-01", end="2022-09-05")
data_other = data_other['Close']
print(data_other)

data_other.dropna()

###################################################################################################################################################################
# SLBs all fit becoming a portfolio.
database_slb = database_slb
database_slb = database_slb.fillna(method="ffill")

print(database_slb)


slb_mean_lst = database_slb.mean(axis=1)   # calculate the 39 SLBs each time mean price 


# tree models
y = slb_mean_lst
X = data_other

X = pd.DataFrame(X)
X = X.dropna(how='any')
X = X.fillna(method="ffill")
X = X.astype('int')
print(X)

y = y.dropna(how='any')
y = y.fillna(method="ffill")
y = y.astype('int')
print(y)

y = y.drop(y.index[149:], axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=6, min_samples_split=2, min_samples_leaf=1, 
                             min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
                             min_impurity_decrease=0.0)

print(clf.fit(X_train, y_train))
print(clf.score(X_test, y_test))


tree.plot_tree(clf, filled=True, fontsize=5, feature_names=['^DJI', 'SI=F', 'BTC-USD', '^IXIC', 
                                                            'BZ=F', 'NG=F', 'ZC=F', 'ZS=F', 'CC=F', 'KC=F', 'CT=F', '^N225', '^HSI'])









