
import pmdarima as pm
from pmdarima.model_selection import train_test_split

def get_actuals(series_lst, periods_back):
    '''
    Puts the actual values for the series in a list to compare results against forecast
    '''
    actuals = []
    for i in series_lst:
        if periods_back == 1:
            actuals.append(i[-1:].iloc[0,0])
        else:
            actuals.append(i[-periods_back:-1].iloc[0,0])
    return actuals

def calculate_errors(actual, forecast):
    '''
    Calculates the MAPE (mean absolute percentage error)
    '''
    return abs((actual-forecast)/actual)

def arima_forecast(store_ids, dept_ids, art_dict, holdout_periods, interval):
    '''
    Estimates ARIMA parameters for all 70 groups and then forecasts
    '''
    lst_of_forecasts = []
    for _ in range(holdout_periods):
        lst_of_forecasts.append([]) 
    series_lst = []
    for idx, val in enumerate(series_setup(store_ids, dept_ids)):
        if idx not in art_dict.keys():
            series_lst.append(resample_series(make_series(val[0], val[1]), interval))
        else:
            series_lst.append(art_dict[idx])
    count = 0
    for idx,i in enumerate(series_lst):
        print(count)
        count += 1
        if idx == 68 or idx == 69:
            train, test = train_test_split(i, train_size=len(i)-holdout_periods)
            model = pm.auto_arima(train, seasonal=True)
            forecasts = model.predict(test.shape[0]).tolist()
            for idx, val in enumerate(forecasts):
                lst_of_forecasts[idx].append(val)
        elif len(i) <= 24:
            train, test = train_test_split(i, train_size=len(i)-holdout_periods)
            model = pm.auto_arima(train, seasonal=True)
            forecasts = model.predict(test.shape[0]).tolist()
            for idx, val in enumerate(forecasts):
                lst_of_forecasts[idx].append(val)
        else:
            train, test = train_test_split(i, train_size=len(i)-holdout_periods)
            model = pm.auto_arima(train, seasonal=True, m=12)
            forecasts = model.predict(test.shape[0]).tolist()
            for idx, val in enumerate(forecasts):
                lst_of_forecasts[idx].append(val)
    return lst_of_forecasts, series_lst

def get_6_month_errors(arima_forecasts6):
    '''
    Returns 6 months of ARIMA forecast MAPE (errors). Useful for building histograms
    '''
    arima6_oct = arima_forecasts6[0]
    arima6_nov = arima_forecasts6[1]
    arima6_dec = arima_forecasts6[2]
    arima6_jan = arima_forecasts6[3]
    arima6_feb = arima_forecasts6[4]
    arima6_mar = arima_forecasts6[5]

    oct_actuals = get_actuals(series_lst6, 6)
    nov_actuals = get_actuals(series_lst6, 5)
    dec_actuals = get_actuals(series_lst6, 4)
    jan_actuals = get_actuals(series_lst6, 3)
    feb_actuals = get_actuals(series_lst6, 2)
    mar_actuals = get_actuals(series_lst6, 1)

    arima6_oct_error = []
    for idx, val in enumerate(oct_actuals):
        arima6_oct_error.append(round(calculate_errors(val, arima6_oct[idx]), 2))

    arima6_nov_error = []
    for idx, val in enumerate(nov_actuals):
        arima6_nov_error.append(round(calculate_errors(val, arima6_nov[idx]), 2))

    arima6_dec_error = []
    for idx, val in enumerate(dec_actuals):
        arima6_dec_error.append(round(calculate_errors(val, arima6_dec[idx]), 2))

    arima6_jan_error = []
    for idx, val in enumerate(jan_actuals):
        arima6_jan_error.append(round(calculate_errors(val, arima6_jan[idx]), 2))

    arima6_feb_error = []
    for idx, val in enumerate(feb_actuals):
        arima6_feb_error.append(round(calculate_errors(val, arima6_feb[idx]), 2))

    arima6_mar_error = []
    for idx, val in enumerate(mar_actuals):
        arima6_mar_error.append(round(calculate_errors(val, arima6_mar[idx]), 2))

    arima6_oct_error = np.array(arima6_oct_error)
    arima6_nov_error = np.array(arima6_nov_error)
    arima6_dec_error = np.array(arima6_dec_error)
    arima6_jan_error = np.array(arima6_jan_error)
    arima6_feb_error = np.array(arima6_feb_error)
    arima6_mar_error = np.array(arima6_mar_error)
    return arima6_oct_error, arima6_nov_error, arima6_dec_error, arima6_jan_error, arima6_feb_error, arima6_mar_error

def make_series_list6(store_ids, dept_ids, art_dict, interval):
    '''
    Returns a list of all the 70 series. Useful for building plots
    '''
    series_lst = []
    for idx, val in enumerate(series_setup(store_ids, dept_ids)):
        if idx not in art_dict.keys():
            series_lst.append(resample_series(make_series(val[0], val[1]), interval))
        else:
            series_lst.append(art_dict[idx])
    return series_lst

def updated_six_period_plot_forecast_vs_arima(idx):
    '''
    Builds plots that show the series history and then also 
    the forecast with confidence intervals
    '''
    series = series_lst6[idx]
    if idx == 68 or idx == 69:
        train, test = train_test_split(series, train_size=len(series)-6)
        model = pm.auto_arima(train, seasonal=True)
        forecasts = model.predict(test.shape[0]).tolist()

    elif len(series) <= 24:
        train, test = train_test_split(series, train_size=len(series)-6)
        model = pm.auto_arima(train, seasonal=True)
        forecasts = model.predict(test.shape[0]).tolist()

    else:
        train, test = train_test_split(series, train_size=len(series)-6)
        model = pm.auto_arima(train, seasonal=True, m=12)
        forecasts = model.predict(test.shape[0]).tolist()
    
    forecasts = np.insert(np.array(forecasts), 0 , train.iloc[-1][0]).tolist()
    params = model.get_params()
    SARIMAmodel = SARIMAX(train, order=params['order'], seasonal_order=params['seasonal_order']).fit()
    fcast = SARIMAmodel.get_forecast(6)
    conf_inf = fcast.conf_int()
    print(model)
    fig, ax = plt.subplots(figsize = (18, 12))
    ax.plot(series.index, series, label = 'Actual Sales')
    ax.plot(series[-7:].index, forecasts, label = 'Forecasted Sales')
    ax.fill_between(conf_inf.index, conf_inf['lower TOTAL'].clip_lower(0),
                    conf_inf['upper TOTAL'], color = 'lightgrey',
                    label = '95% Confidence Interval for Forecast')
    ax.axvline(x = series[-6:].index[0], color='k', linestyle='--',
               label = 'End of Historical Sales')
    ax.set_title(f'Comparison of Actual vs Forecasted Sales \n for the 
                {lst_of_stores[idx][0]} Store and {lst_of_stores[idx][1]} Department',
                 fontsize = 20)
    ax.set_xlabel('Time', fontsize = 24)
    ax.set_ylabel('Sales', fontsize = 24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize = 16)
    #ax.set_ylim([None, 5350]) #for the good
    #ax.set_ylim([None, 21000]) #for the bad
    #ax.set_ylim([None, 5350]) #for the ugly
    plt.grid(c='silver')
    #plt.savefig('../images/the_bad2')
    plt.show()

if __name__ == '__main__':
    #run /home/alec/galvanize/capstone/Forecasting_Walmart_Sales/src/script.py
    #other functions and imports are in the script.py

    lst_of_stores = series_setup(store_id, dept_id)

    art_dict = {} #some series do not forecast on the full historical data.
    art_dict[4] = resample_series(make_series('CA_1', 'FOODS_1'),'M')[22:]
    art_dict[12] = resample_series(make_series('CA_2', 'FOODS_2'),'M')[-13:]
    art_dict[15] = resample_series(make_series('CA_3', 'HOBBIES_2'),'M')[38:]
    art_dict[25] = resample_series(make_series('CA_4', 'FOODS_1'),'M')[16:]
    art_dict[30] = resample_series(make_series('TX_1', 'HOUSEHOLD_1'),'M')[16:]
    art_dict[32] = resample_series(make_series('TX_1', 'FOODS_1'),'M')[14:]
    art_dict[39] = resample_series(make_series('TX_2', 'FOODS_1'),'M')[46:]
    art_dict[50] = resample_series(make_series('WI_1', 'HOBBIES_2'),'M')[41:]
    art_dict[60] = resample_series(make_series('WI_2', 'FOODS_1'),'M')[16:]
    art_dict[68] = resample_series(make_series('WI_3', 'FOODS_2'),'M')[34:]
    art_dict[69] = resample_series(make_series('WI_3', 'FOODS_3'),'M')[26:]