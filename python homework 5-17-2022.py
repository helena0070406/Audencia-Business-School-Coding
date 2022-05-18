# By. Yujing CHEN & Haiwei FU
# import libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from datetime import date
import requests

# spider and scraw in the yahoo finance. use https://finance.yahoo.com/  (US yahoo finance, and get 'S&P 500'.)
url = 'https://finance.yahoo.com/'
htlm = requests.get(url=url)



data = web.get_data_yahoo('^GSPC', start='1/1/2022', end='15/5/2022')
print(data)

data = data['Close']
############################################################################################################################################
# S&P 500 index discription and features, Bollinger Bands, CCI, MA, ROC. ###################################################################
############################################################################################################################################
# Compute the Bollinger Bands 
def BBANDS(data, ndays):

 MA = pd.Series.rolling(data['Close'], ndays).mean()
 SD = pd.Series.rolling(data['Close'], ndays).std()

 b1 = MA + (2 * SD)
 B1 = pd.Series(b1, name = 'Upper BollingerBand') 
 data = data.join(B1) 
 
 b2 = MA - (2 * SD)
 B2 = pd.Series(b2, name = 'Lower BollingerBand') 
 data = data.join(B2) 
 
 return data
 
# Retrieve the GSPC data from Yahoo finance:
data = web.DataReader('^GSPC',data_source='yahoo',start='1/1/2022', end='15/5/2022')
data = pd.DataFrame(data)

# Compute the Bollinger Bands for GSPC using the 50-day Moving average
n = 50
GSPC_BBANDS = BBANDS(data, n)
print(GSPC_BBANDS)


###########################################################################################################################################
# calculate CCI. 

url = 'https://finance.yahoo.com/'
htlm = requests.get(url=url)



data = web.get_data_yahoo('^GSPC', start='1/1/2022', end='15/5/2022')
print(data)

data = data['Close']



# Commodity Channel Index 
def CCI(data, ndays): 
 TP = (data['High'] + data['Low'] + data['Close']) / 3 
 CCI =(TP - pd.Series.rolling(TP, ndays).mean())/ (0.015 * pd.Series.rolling(TP, ndays).std())
 CCI.name = 'CCI'
 data = data.join(CCI) 
 return data



# Retrieve the GSPC data from Yahoo finance:
data = web.DataReader('^GSPC',data_source='yahoo',start='1/1/2022', end='15/5/2022')
data = pd.DataFrame(data)


# Compute the Commodity Channel Index(CCI) for GSPC based on the 20-day Moving average
n = 20
GSPC_CCI = CCI(data, n)
CCI = GSPC_CCI['CCI']

# Plotting the Price Series chart and the Commodity Channel index below
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
plt.plot(data['Close'],lw=1)
plt.title('GSPC Price Chart')
plt.ylabel('Close Price')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)
plt.plot(CCI,'k',lw=0.75,linestyle='-',label='CCI')
plt.legend(loc=2,prop={'size':9.5})
plt.ylabel('CCI values')
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)


##########################################################################################################################################
# Moving Averages (MA)


# Simple Moving Average 
def SMA(data, ndays): 
 SMA = pd.Series.rolling(data['Close'], ndays).mean()
 SMA.name = 'SMA'
 data = data.join(SMA) 
 return data



# Retrieve the GSPC data from Yahoo finance:
data = web.DataReader('^GSPC',data_source='yahoo',start='1/1/2022', end='15/5/2022')
data = pd.DataFrame(data) 
close = data['Close']

# Compute the 50-day SMA for GSPC
n = 50
SMA_GSPC = SMA(data,n)
SMA_GSPC = SMA_GSPC.dropna()
SMA = SMA_GSPC['SMA']



# Plotting the GSPC Price Series chart and Moving Averages below
plt.figure(figsize=(9,5))
plt.plot(data['Close'],lw=1, label='GSPC Prices')
plt.plot(SMA,'g',lw=1, label='50-day SMA (green)')

plt.legend(loc=2,prop={'size':11})
plt.grid(True)

###############################################################################################
# Rate of Change (ROC)

def ROC(data,n):
 N = data['Close'].diff(n)
 D = data['Close'].shift(n)
 ROC = pd.Series(N/D,name='Rate of Change')
 data = data.join(ROC)
 return data 
 
# Retrieve the GSPC data from Yahoo finance:
data = web.DataReader('^GSPC',data_source='yahoo', start='1/5/2022', end='15/5/2022')
data = pd.DataFrame(data)

# Compute the 5-period Rate of Change for GSPC
n = 5
NIFTY_ROC = ROC(data,n)
ROC = NIFTY_ROC['Rate of Change']

# Plotting the Price Series chart and the Ease Of Movement below
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
plt.plot(data['Close'],lw=1)
plt.title('GSPC Price Chart')
plt.ylabel('Close Price')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)
plt.plot(ROC,'k',lw=0.75,linestyle='-',label='ROC')
plt.legend(loc=2,prop={'size':9})
plt.ylabel('ROC values')
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)

###########################################################################################################################################
# S&P500 index regression, OLS, ridge regression, LASSO. ##################################################################################
###########################################################################################################################################

url = 'https://finance.yahoo.com/'
htlm = requests.get(url=url)



data = web.get_data_yahoo('^GSPC', start='1/1/2022', end='15/5/2022')
print(data)




from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y, true_coefficient = make_regression(n_samples=200, n_features=30, n_informative=10, noise=100, coef=True, random_state=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5, train_size=120)
print(X_train.shape)
print(y_train.shape)

###################################################################################################################################
# OLS

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression().fit(X_train, y_train)
print("R^2 on training set: %f" % linear_regression.score(X_train, y_train))
print("R^2 on test set: %f" % linear_regression.score(X_test, y_test))

from sklearn.metrics import r2_score
print(r2_score(np.dot(X, true_coefficient), y))


plt.figure(figsize=(10, 5))
coefficient_sorting = np.argsort(true_coefficient)[::-1]
plt.plot(true_coefficient[coefficient_sorting], "o", label="true")
plt.plot(linear_regression.coef_[coefficient_sorting], "o", label="linear regression")

plt.legend()


from sklearn.model_selection import learning_curve 


def plot_learning_curve(est, X, y):
    training_set_size, train_scores, test_scores = learning_curve(est, X, y, train_sizes=np.linspace(.1, 1, 20))
    estimator_name = est.__class__.__name__
    line = plt.plot(training_set_size, train_scores.mean(axis=1), '--', label="training scores " + estimator_name)
    plt.plot(training_set_size, test_scores.mean(axis=1), '-', label="test scores " + estimator_name, c=line[0].get_color())
    plt.xlabel('Training set size')
    plt.legend(loc='best')
    plt.ylim(-0.1, 1.1)
    
plt.figure()    
plot_learning_curve(LinearRegression(), X, y)

####################################################################################################################################
# ridge regression
# Its an OLS, but it's a transformer of OLS. better than OLS, we can give a number to hyper-perameter, and see the good varients x, 
# after a fit.
from sklearn.linear_model import Ridge
ridge_models = {}
training_scores = []
test_scores = []

# we give the alpha, and make the perameter in the ridge regression more augment, so that we can see which points are good.  
for alpha in [100, 10, 1, .01]:
    ridge = Ridge(alpha=alpha).fit(X_train, y_train)
    training_scores.append(ridge.score(X_train, y_train))
    test_scores.append(ridge.score(X_test, y_test))
    ridge_models[alpha] = ridge

plt.figure()
plt.plot(training_scores, label="training scores")
plt.plot(test_scores, label="test scores")
plt.xticks(range(4), [100, 10, 1, .01])
plt.legend(loc="best")


plt.figure(figsize=(10, 5))
plt.plot(true_coefficient[coefficient_sorting], "o", label="true", c='b')

for i, alpha in enumerate([100, 10, 1, .01]):
    plt.plot(ridge_models[alpha].coef_[coefficient_sorting], "o", label="alpha = %.2f" % alpha, c=plt.cm.summer(i / 3.))
    
plt.legend(loc="best")



plt.figure()
plot_learning_curve(LinearRegression(), X, y)
plot_learning_curve(Ridge(alpha=10), X, y)


#######################################################################################################################################
# LASSO
# Its an optimization of ridge regression. But when the features is not good, it will remian the '0', not go negative as the RR.
# So based on the feartures, we decide the hyper-perameters smaller. And get rid of the extream features being larger.
from sklearn.linear_model import Lasso

lasso_models = {}
training_scores = []
test_scores = []

# decide the alpha smaller, the hyper-perameter.
for alpha in [30, 10, 1, .01]:
    lasso = Lasso(alpha=alpha).fit(X_train, y_train)
    training_scores.append(lasso.score(X_train, y_train))
    test_scores.append(lasso.score(X_test, y_test))
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
plot_learning_curve(LinearRegression(), X, y)
plot_learning_curve(Ridge(alpha=10), X, y)
plot_learning_curve(Lasso(alpha=10), X, y)

######################################################################################################################################
#Industry reseach, we select the industries in detail in the S&P 500, and use decision tree to see its relationships between the whole
#S&P 500 index，if the S&P 500 index goes up in x%, what is the entropy of selected the indusrties. For indusrties, we based on the WI
#KI, and select its indusrty index in the S&P 500. The industry is the 'information technology' in the S&P 500, and also select its s
#ubset: Software service, Hardware equipment and Semiconductors and with its Semiconductor equipment. 
######################################################################################################################################
wikiurl = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
htlm = requests.get(url=wikiurl)

url = 'https://finance.yahoo.com/'
html = requests.get(url=url)

Information_Technology45 = web.DataReader(name='^SP500-45', data_source='yahoo', start='5/1/2022', end='5/14/2022')
Information_Technology45 = Information_Technology45['Close']
print(Information_Technology45)

Software_Services = web.DataReader(name='^SP500-4510', data_source='yahoo', start='5/1/2022', end='5/14/2022')
Software_Services = Software_Services['Close']
print(Software_Services)

Hardware_Equipment = web.DataReader(name='^SP500-4520', data_source='yahoo', start='5/1/2022', end='5/14/2022')
Hardware_Equipment = Hardware_Equipment['Close']
print(Hardware_Equipment)

Semiconductors_Semiconductor_Equipment = web.DataReader(name='^SP500-4530', data_source='yahoo', start='5/1/2022', end='5/14/2022')
Semiconductors_Semiconductor_Equipment = Semiconductors_Semiconductor_Equipment['Close']
print(Semiconductors_Semiconductor_Equipment)



    
Semiconductors = web.DataReader(name='^SP500-45301020', data_source='yahoo', start='5/1/2022', end='5/14/2022')
Semiconductors = Semiconductors['Close']
print(Semiconductors)

data = web.DataReader('^GSPC', data_source='yahoo', start='5/1/2022', end='5/14/2022')
data = data['Close']

X = [['Information_Technology45', 'Software_Services', 'Hardware_Equipment', 'Semiconductors_Semiconductor_Equipment', 'Semiconductors']]
y = ['data']
















































