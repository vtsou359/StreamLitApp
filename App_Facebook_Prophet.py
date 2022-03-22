import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64

st.title('📈 Automated Time Series Forecasting by Billy')

"""
This data app automatically generates future forecasts from stocks. The data are automatically downloaded from Yahoo Finance. 
**Try it on MSFT with the options: data period->2y, forecast period->100, and yearly seasonality.**
In beta mode.
"""


"""
## Step 1: Please write the name of stock you would like to forecast as indicated in Yahoo Finance. (e.g.: for Microsoft write MSFT)
"""
stock_name=st.text_input("Enter the name/symbol of the stock here", value="MSFT")
period_select = st.select_slider(
     'Select the data period that will feed the model',
     options=['1mo', '3mo', '6mo', '1y', '2y', '5y','10y','ytd','max'])

msft = yf.Ticker(stock_name)
# get historical market data
hist = msft.history(period=period_select)
hist['ds'] = hist.index

df=hist[['Close']]
df=df.reset_index()
df=df.rename(columns = {'Date':'ds', 'Close':'y'})
data=df

#df = st.file_uploader(
    #'Import the time series xlsx file here. Columns must be labeled ds (time)  and y (values) and sorted from the oldest time value to the newest. The ds column should be of a format DATE in excel. The y column must be numeric, and represents the measurement we wish to forecast.',
    #type='xlsx')



#encoding='auto'
if df is not None:
    #data = pd.read_excel(df)
    data['ds'] = pd.to_datetime(data['ds'], errors='coerce')

    st.write(data)

    max_date = data['ds'].max()
    # st.write(max_date)

"""
## Step 2: Select Forecast Horizon
Keep in mind that forecasts become less accurate with larger forecast horizons.
"""

periods_input = st.number_input('How many periods would you like to forecast into the future? (min=1 and max=3650->(10 years))',
                                min_value=1, max_value=3650)

confidence_interval_input = st.number_input('Please select the confidence interval from 10% to 99%. For best results use a number from 80 to 99. ',
                                min_value=10, max_value=99)
"""
### Please select (if you want) a yearly, monthly, weekly or daily seasonality. Sometimes the model behaves better with the boxes unselected.
"""
yearlyseas=st.checkbox('YEARLY SEASONALITY? ', value=False)
monthlyseas=st.checkbox('MONTHLY SEASONALITY? ', value=False)
weeklyseas=st.checkbox('WEEKLY SEASONALITY? ', value=False)
dailyseas=st.checkbox('DAILY SEASONALITY? ', value=False)

if df is not None:
    m = Prophet(weekly_seasonality=weeklyseas, daily_seasonality=dailyseas, yearly_seasonality=yearlyseas,interval_width=(confidence_interval_input/100))

    if monthlyseas==True:
        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    m.fit(data)

"""
### Step 3: Visualize Forecast Data
The below visual shows future predicted values. "yhat" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.
"""
if df is not None:
    future = m.make_future_dataframe(periods=periods_input)

    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered = fcst[fcst['ds'] > max_date]
    st.write(fcst_filtered)

    """
    The next visual shows the actual (black dots) and predicted (blue line) values over time.
    """
    fig1 = m.plot(forecast)
    st.write(fig1)

    """
    The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.
    """
    fig2 = m.plot_components(forecast)
    st.write(fig2)

"""
### Step 4: Download the Forecast Data
The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)