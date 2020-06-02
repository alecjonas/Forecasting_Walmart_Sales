run /home/alec/galvanize/capstone/Forecasting_Walmart_Sales/src/script.py

def no_diff_adf_pvalue(series):
    test = sm.tsa.stattools.adfuller(series[list(series)[0]])
    return round(test[1], 2)

def one_diff_adf_pvalue(series):
    test = sm.tsa.stattools.adfuller(series[list(series)[0]].diff()[1:])
    return round(test[1], 2)

def two_diff_adf_pvalue(series):
    test = sm.tsa.stattools.adfuller(series[list(series)[0]].diff().diff()[2:])
    return round(test[1], 2)

if __name__ == '__main__':

    lst= []
    for i in series_setup(store_id, dept_id):
        lst.append(resample_series(make_series(i[0], i[1]), 'M'))
    lst1 = []
    for i in lst:
        lst1.append(no_diff_adf_pvalue(i))        
    lst_of_series_that_need_differencing = []
    lst_of_series_no_differencing = []
    for idx, val in enumerate(series_setup(store_id, dept_id)):
        #print(f"ADF p-value for {val} series: {lst1[idx]}")
        if lst1[idx] <= .05:
            lst_of_series_no_differencing.append(val)
        else:
            lst_of_series_that_need_differencing.append(val)

    lst2 = []
    for i in lst_of_series_that_need_differencing:
        lst2.append(resample_series(make_series(i[0], i[1]), 'M'))
    lst3 = []
    for i in lst2:
        lst3.append(one_diff_adf_pvalue(i))
    lst_of_series_that_need_one_differencing = []
    lst_of_series_two_differencing = []
    for idx, val in enumerate(lst_of_series_that_need_differencing):
        #print(f"ADF p-value for {val} series: {lst3[idx]}")
        if lst3[idx] <= .05:
            lst_of_series_that_need_one_differencing.append(val)
        else:
            lst_of_series_two_differencing.append(val)

    lst4= []
    for i in lst_of_series_two_differencing:
        lst4.append(resample_series(make_series(i[0], i[1]), 'M'))
    lst5 = []
    for i in lst4:
        lst5.append(two_diff_adf_pvalue(i))
    two_diff_lst = []
    lst_of_series_three_differencing = []
    for idx, val in enumerate(lst_of_series_two_differencing):
        #print(f"ADF p-value for {val} series: {lst5[idx]}")
        if lst5[idx] <= .05:
            two_diff_lst.append(val)
        else:
            lst_of_series_three_differencing.append(val)

    no_diff_lst = lst_of_series_no_differencing
    one_diff_lst = lst_of_series_that_need_one_differencing
    two_diff_lst = two_diff_lst