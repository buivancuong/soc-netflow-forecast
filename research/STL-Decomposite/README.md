<h1>Decomposite of Time Series</h1>

<h2>Summary</h2>

Decomposite is an important technique for Time Series Analysis.

A given time series is thought to consist of three systematic components including: **trend**, **seasonality**, and one non-systematic component called **noise** (**residual/residue**).

1. **Trend**: The increasing or decreasing value in the series; which reflects the long-term progression of the series.
2. **Seasonality**: The repeating short-term cycle in the series; which reflects repeated but non-periodic fluctuations.
3. **Noise**: The random variation in the series. It represents the residuals or remainder of the time series after the other components have been removed.

<h2>Additive or Multiplicative Decomposition</h2>

<h3>Additive Model</h3>

The components are added together as follows:

y(t) = Trend + Sensonality + Noise

<h3>Multiplicative Model</h3>

The components are multiplied together as follows:

y(t) = Trend * Sensonality * Noise


<h2>Reference</h2>

https://anomaly.io/seasonal-trend-decomposition-in-r/

http://www2.hawaii.edu/~fuleky/econ427/6_Time_series_decomposition.html#3

