from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose

series = Series.from_csv('./airline-passengers.csv', header = 0)
result = seasonal_decompose(series, model='multiplicative') # multiplicative or additive
result.plot()
pyplot.show()
