import json
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime,timedelta
import base64
from io import BytesIO
import json
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib
""" matplotlib.use('TKAgg') """
""" matplotlib.use('Agg') """
import matplotlib.dates as mdates
import plotly.graph_objects as go
""" import cufflinks as cf """
import plotly


stock_symbols = None
stock_names_dict = None
stock_names=None
holding=False
isTrading_happened = False
maxNoFilesStored = 5
storeStockHistInBackEnd = False
storeStockHistInFrontEnd = False
priceDatainRequest = False


def load_artifacts():
    global stock_symbols    
    global stock_names_dict  
    global stock_names  
    
    with open('./artifacts/nseStockNamesKeys.json', 'r') as f:
        stock_names_dict = json.load(f)
        stock_symbols = [*stock_names_dict.keys()]
        """ stock_symbols = list(stock_names_dict) """
        stock_names = [*stock_names_dict.values()]
        """ print(type(stock_names))
        print(type(stock_symbols))
        print(stock_names)
        print(stock_symbols) """
        

def get_stockData_frmYahooApi(stockInputInf):
    
    tickerData = yf.Ticker(stockInputInf['ticker_symbol'])
                
    tickerDF = tickerData.history(interval=stockInputInf['history_freq'], start=stockInputInf['history_start'], end = stockInputInf['history_end'])

    priceData = tickerDF[tickerDF.columns[0:4]]
    """ priceData = tickerDF.Open """

    priceData = priceData.asfreq(pd.infer_freq(priceData.index))

    """ except Exception as e:        
        response = '{}, at API while fetching '.format(e) """
    if(priceData.shape[0] == 0):
        raise Exception("the selected frequency history is not available; select '1d' ")
    
    """  print('the length...',priceData.shape[0] )
    print('the non NAn...',priceData.count() ) """
    return priceData

def get_stockData_frmStoreOrYapi(stockInputInf):
    
    global storeStockHistInBackEnd
    
    if (storeStockHistInBackEnd):        
    
        ticker_symbolN = stockInputInf['ticker_symbol'].split('.')[0]
        
        infFilePath = r'./data/stockInputInf_'+ticker_symbolN+'.json'
        stockDataFilePath = r'./data/stockDataPickle_'+ticker_symbolN+'.pkl'
                
        if(os.path.isfile(infFilePath)):
        
            with open(infFilePath, 'r') as f:
                stored_stockInput = json.load(f)
                
        else:stored_stockInput=None
        
        """ print('the stored', stored_stockInput)
        
        print('the if', stored_stockInput==stockInputInf) """

        if(stored_stockInput==stockInputInf):
            print('the input stock data is stored in the local disk')
            priceData = pd.read_pickle(stockDataFilePath)   

        else:            
            json_object = json.dumps(stockInputInf, indent=5)
            with open(infFilePath, 'w') as f:
                f.write(json_object)
                
            print('the input stock data will be fetched from the yahoo finance server')
            
            priceData = get_stockData_frmYahooApi(stockInputInf)

            priceData.to_pickle(stockDataFilePath)
            
         ## Delete(when their no exceeds maxNO) the first stored inf, data files to optimize the space
            global maxNoFilesStored
            
            listStoredIpInfFiles = ['./data/'+filename for filename in os.listdir('./data') if filename.startswith('stockInputInf_')]
            listStoredDataPickleFiles = ['./data/'+filename for filename in os.listdir('./data') if filename.startswith('stockDataPickle_')]
            
            if(len(listStoredIpInfFiles)>maxNoFilesStored):
                sortListInf = sorted(listStoredIpInfFiles, key=lambda file:os.path.getctime(file))[:-maxNoFilesStored]         
                sortListData = sorted(listStoredDataPickleFiles, key=lambda file:os.path.getctime(file))[:-maxNoFilesStored]
                
                for file in sortListInf:                
                        os.remove(file)
                for file in sortListData:                
                        os.remove(file)
                        
    else: priceData = get_stockData_frmYahooApi(stockInputInf)
            
    return priceData   

        
def get_buy_sell_days(price_data, buy_daysCr, sell_daysCr):
    
    pct_change = price_data.pct_change()[1:]
    
    def buying_condition(sub_series):
        return (sub_series > 0).all()
    
    def selling_condition(sub_series):
        return (sub_series < 0).all()
    
    buying_days_data = pct_change.rolling(buy_daysCr).apply(buying_condition)
    
    selling_days_data = pct_change.rolling(sell_daysCr).apply(selling_condition)
    
    return {'potential buying days': buying_days_data, 'potential selling days': selling_days_data}


def check_cumulative_pcnt_cng(price_data, buy_date, potential_selling_day):
    
    pct_cng = price_data.pct_change()[1:]
    
    sub_series = 1+pct_cng[buy_date+timedelta(hours=1):potential_selling_day]
    #sell only when gain is positive
    return sub_series.product() > 1


def get_invest_return(df_stocks, starting_funds, verbose=False):
    
    global holding
    
    price_data = df_stocks.price
    
    holding = False
    
    current_funds = starting_funds
    current_shares = 0
    last_purchase_date = None 
    
    events_list = []
    verbose_list = []
    
    lenth_dfStocks = df_stocks.shape[0]
    """ lenth_dfStocks = len(df_stocks.index) """
    
    for date, data in df_stocks.iterrows():
        
        if (not holding) and data.buying_day:
            # no of shares of possible purchase
            num_share_purchase = int(current_funds/data.price)
            
            current_shares += num_share_purchase
            current_funds -= num_share_purchase*data.price
            current_funds = round(current_funds, 2)
        
            #
            last_purchase_date = date
            events_list.append(('purchase_date', date))
            verbose_list.append('Purchased %s shares at INR %s on %s with available fund INR %s' %(num_share_purchase, round(data.price, 2), date, current_funds))
            holding = True
            
            if verbose:
                print('Purchased %s shares at INR %s on %s with available fund INR %s' %(num_share_purchase, data.price, date, current_funds) )
                
        elif holding and data.potential_selling_day:
            
            if check_cumulative_pcnt_cng(price_data, last_purchase_date, date):
                
                current_funds += round(current_shares*data.price, 2)
                current_funds = round(current_funds, 2)
                
                if verbose:
                    print('Sold %s shares at INR %s on %s totaling current fund INR %s' %(current_shares, round(data.price, 2), date, current_funds) )
                    print('----------------------------------')
                    
                
                events_list.append(('sale_date', date))
                verbose_list.append('Sold %s shares at INR %s on %s totaling current fund INR %s' %(current_shares, data.price, date, current_funds) )
                
                #reset
                current_shares = 0
                holding = False               
    
    
    return current_funds, current_shares, events_list, verbose_list



def plot_stock_Staticdisplay(priceData, ticker_symbolN):
        priceData = priceData.Open
        fig = plt.figure(figsize=(10,5),dpi=175)
        plt.plot(priceData)
        
        if (priceData.index[0].year == priceData.index[-1].year):
            for month in range(priceData.index[0].month+1, priceData.index[-1].month+1):
                plt.axvline(datetime(priceData.index[0].year,month,1), color='k', linestyle='--', alpha=0.2)
                
            # Set the locator
            locator = mdates.MonthLocator()  # every month
            # Specify the format - %b gives us Jan, Feb...
            """ fmt = mdates.DateFormatter('%Y-%m') """
            """ fmt = mdates.DateFormatter('%Y-%b') """
            fmt = mdates.DateFormatter('%b')
            X = plt.gca().xaxis
            X.set_major_locator(locator)
            # Specify formatter
            X.set_major_formatter(fmt)
            
            """ plt.xticks(rotation=10) """
            
            title_string = r'Stock Price Data (year - %s)'%priceData.index[0].year
            
        else:

            for year in range(priceData.index[0].year+1, priceData.index[-1].year+1):
                plt.axvline(datetime(year,1,1), color='k', linestyle='--', alpha=0.2)
                
            title_string = 'Stock Price Data'
        
        # label including this form1 will have these properties
        form1 = {'family': 'serif', 'color': 'blue', 'size': 15}
 
# label including this form2 will have these properties
        form2 = {'family': 'serif', 'color': 'darkred', 'size': 20, 'weight':'bold'}
        plt.xlabel("Date", fontdict=form1)
        plt.ylabel("Value (in INR)", fontdict=form1)
        plt.title(r"'%s' %s" %(ticker_symbolN, title_string), fontdict=form2, alpha=1, loc='center')
        """ plt.savefig(r"stock_data.png") """

        tmpfile = BytesIO()
        """ fig.tight_layout() """
        fig.savefig(tmpfile, format='png',bbox_inches='tight', pad_inches=0.25 )
        
        tmpfile.seek(0)
        encodedStock_display = base64.b64encode(tmpfile.getvalue()).decode('utf-8')  
        
        return encodedStock_display

def plot_stock_Plotlydisplay(tickerDF_plotly, ticker_symbolN):
    
    title = '<b>'+r"'%s' Stock Price Data" %(ticker_symbolN)+'</b>'
    labels = ['Open', 'High', 'Low', 'Close']
    colors = ['rgb(25,25,25)', '#4dac26', '#d7191c', '#ffa600']

    mode_size = [3, 3, 3, 3]
    line_size = [1, 1, 1, 1]

    x_data = tickerDF_plotly.index

    y_data = tickerDF_plotly
    
    """ 'displaylogo':False, """
    
    config = dict({'displaylogo':False, 'scrollZoom': False,'displayModeBar': True,
                   'modeBarButtonsToAdd':['drawopenpath',
                                        'eraseshape'
                                       ],
                   'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'toggleSpikelines']})
    fig = go.Figure()

    for i in range(0, 4):
        if i==0 or i==3:
            fig.add_trace(go.Scatter(x=x_data, y=tickerDF_plotly[tickerDF_plotly.columns[i]], mode='lines',
                name='<b>'+tickerDF_plotly.columns[i]+'</b>',
                text=[" stock"],
                line=dict(color=colors[i], width=line_size[i]),
                connectgaps=True,
            ))
        else:
            fig.add_trace(go.Scatter(x=x_data, y=tickerDF_plotly[tickerDF_plotly.columns[i]], mode='lines',
                name='<b>'+tickerDF_plotly.columns[i]+'</b>',
                text=[" stock"],
                line=dict(color=colors[i], width=line_size[i]),
                connectgaps=True,
                visible='legendonly'
            ))
            

        # endpoints
        fig.add_trace(go.Scatter(
            x=[x_data[0], x_data[-1]],
            y=[tickerDF_plotly[tickerDF_plotly.columns[i]][0], tickerDF_plotly[tickerDF_plotly.columns[i]][-1]],
            mode='markers',
            marker=dict(color=colors[i], size=mode_size[i]),
            showlegend=False
        ))

    fig.update_layout(
        xaxis=dict(        
            title='Date',
            showline=True,
            showgrid=True,
            gridcolor= 'rgba(0,0,0,0.1)',
            color='blue',
            showticklabels=True,
            linecolor='gray',
            linewidth=1,
            mirror = True,
            ticks='inside',
            tickcolor='rgb(204, 204, 204)',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            title='<b>'+"INR"+'</b>',        
            showline=True,
            showgrid=True,
            gridcolor= 'rgba(0,0,0,0.1)',
            color='blue',
            showticklabels=True,
            linecolor='gray',
            linewidth=1.5,
            mirror = True,
            ticks='inside',
            tickcolor='rgb(204, 204, 204)',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),               
        showlegend=True,
        plot_bgcolor='white'
    )

    annotations = []


    # Title
    annotations.append(dict(xref='paper', yref='paper', x=0.31, y=1.02,
                                xanchor='left', yanchor='bottom',
                                text=title,
                                font=dict(family='serif',
                                            size=22,
                                            color='darkred'),
                                showarrow=False))
    # Source
    annotations.append(dict(xref='paper', yref='paper', x=0.95, y=-0.125,
                              xanchor='center', yanchor='top',
                              text='Data Source: Yahoo Finance API',
                              font=dict(family='Arial',
                                        size=10,
                                        color='rgb(150,150,150)'),
                              showarrow=False))

    fig.update_layout(annotations=annotations, yaxis_title="Value (in INR)")

    fig.update_layout(legend=dict(orientation="h",
        yanchor="bottom",
        y=0.915,
        xanchor="left",
        x=-0.004,
        bgcolor="#e9f2ff",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,         
    font=dict(family='Arial',size=13,color='rgb(50,50,50)')))

    fig.update_xaxes(title_font_family="Arial")
    fig.update_yaxes(title={"font": {"size": 15}, "standoff": 0})
    fig.update_xaxes(title={"font": {"size": 15}})
    fig.update_yaxes(title_font_family="Arial")
    fig.update_layout(
        autosize=True,        
    margin=dict(
            autoexpand=True,
            l=20,
            r=50,
            t=60,
            b=15,
            pad = 5
        ))

    dx_xaxes = -(x_data[0]- x_data[-1])

    fig.update_xaxes(range = [x_data[0]-0.02*dx_xaxes, x_data[-1]+0.02*dx_xaxes])
    
    fig.update_layout(    
    newshape_line_color='cyan',
    dragmode = None   
    )
    """ dragmode='drawopenpath', """

    
    """ plotly.offline.plot(fig, filename=r'D:\Digital Platforms\ML\TimeSeries\Stock-Trading-App_finshBundled_toDeploy\server\data\stockDataHtml.html')  """
    
   
    """ width=715,
        height=400, """

    """ fig.show() """
    stockData_plotlyDiv = plotly.offline.plot(fig, config=config, include_plotlyjs=False, output_type='div')
    
    with open(r"D:\Digital Platforms\ML\TimeSeries\Stock-Trading-App_finshBundled_toDeploy\server\data\stockData_Div.html",'w') as f:
        f.write(stockData_plotlyDiv)
        
    return stockData_plotlyDiv


def getStock_acf_pacf_plots(stockInputInf = {}, price_data=False):
    
    global priceDatainRequest 
    
    ticker_symbolN = stockInputInf['ticker_symbol'].split('.')[0]
        
    if( not price_data and stockInputInf):
        priceDatainRequest = False        
        priceData = get_stockData_frmStoreOrYapi(stockInputInf)
        priceDataToClient = priceData.to_json(orient="split")
            
    elif(price_data):
        priceDatainRequest = True
        priceData = pd.read_json(price_data,typ='frame', orient='split') 
        priceData = priceData.asfreq(pd.infer_freq(priceData.index))
        """ 
        df=pd.read_json(data,convert_dates='index',date_unit='ms').set_index('index') 
        df.index=pd.to_datetime(df.index,unit='ms')
        df.drop(['name'], axis=1)
        """
        priceDataToClient = ''
    else: 
        print('The Input request is NUll')
        return 'Error: The Input request is NUll'
    
    #print('The decoded price data ISSSS.....', priceData)
    priceData = priceData.Open
    
    priceData_diff = priceData.pct_change().dropna()
    
    form1 = {'family': 'serif', 'color': 'blue', 'size': 11}
    form2 = {'family': 'serif', 'color': 'darkred', 'size': 13, 'weight':'bold'}      
    
    fig_help, axs = plt.subplots(nrows = 1,
                        ncols = 2,
                        figsize=(10,5),dpi=150
                       )    
    ax1 = axs[0]
    ax2 = axs[1]
    
    plot_acf(priceData_diff, ax=ax1, lags=25, alpha=.1, zero=False)
    #, title=r'%s auto correlation'%ticker_symbol
    
    plt.xticks(list(range(0,26,2)))
    ax1.set_xlabel("ACF Lag (in days)", fontdict=form1)
    ax1.set_ylabel("Correlation Ratio", fontdict=form1)
    ax1.set_ylim([-0.125, 0.125])
    ax1.set_title(r"'%s' Auto Correlation"%ticker_symbolN, fontdict=form2, alpha=1, loc='center')
    
    plot_pacf(priceData_diff, ax=ax2, lags=25, alpha=.1, zero=False)
    #,title=r'%s partial auto correlation'%ticker_symbol
    """  plot_pacf(priceData_diff, ax=ax2, lags=25, alpha=.05, title=r"'%s' Partial Auto Correlation (PACF) Plot for " %ticker_symbol) """
        
    ax2.set_xlabel("PACF Lag (in days)", fontdict=form1)
    ax2.set_ylim([-0.125, 0.125])
    #ax2.set_ylabel("(PACF) Correlation Ratio", fontdict=form1)
    ax2.set_title(r"'%s' Partial Auto Correlation" %ticker_symbolN, fontdict=form2, alpha=1, loc='center') 
    
    """ plt.title(r"ACF and PACF Plots for '%s'" %ticker_symbol, fontdict=form2, alpha=1, loc='center')  """
    
    fig_help.text(0.5,-0.1,r"(ACF, PACF gives the correlation between the stock values at different time periods (refer 'About' section for details), which" "\n" r"guides the user in deciding the input options for min no of days of monotonic increase/decrease in stock price)", ha="center", va="bottom", fontsize=11.2,color='k', alpha =0.85)
    
    plt.savefig(r"stock_acf_pacf.png")

    tmpfile = BytesIO()
    """ fig.tight_layout() """
    fig_help.savefig(tmpfile, format='png',bbox_inches='tight', pad_inches=0.25, dpi=150 )
    
    tmpfile.seek(0)
    encodedStock_acf_pacf = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    
    response_Helpobject = {
        'plot_stockHelp':encodedStock_acf_pacf,
        'priceDatainRequest':priceDatainRequest,
        'priceDataToClient':priceDataToClient
    }
    
    """ print('the help in util is response', response_Helpobject) """

    return response_Helpobject

""" 
 stockInputInf = {'ticker_symbol': 1, 'history_freq':2, 'history_start':3, 'history_end':4}
 new = {key:val for key,val in stockInputInf.items() if key not in ['ticker_symbol', 'history_freq'] 
"""



def display_stock_data(stockInputInf = {}, price_data=False):  
    
        global priceDataToClient                           
               
        """ stockInputInf = {'ticker_symbol': ticker_symbol, 'history_freq':history_freq, 'history_start':history_start, 'history_end':history_end} """
        
        ticker_symbolN = stockInputInf['ticker_symbol'].split('.')[0]
        
        if( not price_data and stockInputInf):
            priceDatainRequest = False        
            priceData = get_stockData_frmStoreOrYapi(stockInputInf)
            priceDataToClient = priceData.to_json(orient="split")
            
        elif(price_data):
            priceDatainRequest = True
            priceData = pd.read_json(price_data,typ='frame', orient='split') 
            priceData = priceData.asfreq(pd.infer_freq(priceData.index))
            priceDataToClient = None
            
        else: 
            print('The Input request is NUll')
            return 'Error: The Input request is NUll'
        #print('the price data', priceData)
       #########
       
        """ encodedStock_display = plot_stock_Staticdisplay(priceData, ticker_symbolN) """
        
        stockData_plotlyDiv = plot_stock_Plotlydisplay(priceData, ticker_symbolN)
        

        response_Displayobject = {
        'plot_stockDisplay':stockData_plotlyDiv,
        'priceDatainRequest':priceDatainRequest,
        'priceDataToClient':priceDataToClient    
        }               

        return response_Displayobject

        ########

    
        
def analyse_stock_return(stockAnalysInput={}, price_data = False):
        print("Inside the predict function")
        
        global priceDatainRequest    
   
        if(stockAnalysInput):
            
            stockInputInf={key:val for key,val in stockAnalysInput.items() if key not in ['init_fund', 'buy_days_condtn', 'sell_days_condtn']} 
            
            init_fund = stockAnalysInput['init_fund']
            buy_days_condtn = stockAnalysInput['buy_days_condtn']
            sell_days_condtn = stockAnalysInput['sell_days_condtn']            
            
            if(not price_data):
                priceDatainRequest = False        
                priceData = get_stockData_frmStoreOrYapi(stockInputInf)
                priceDataToClient = priceData.to_json(orient="split")
            else: 
                priceDatainRequest = True
                priceData = pd.read_json(price_data,typ='frame', orient='split') 
                print('the history from client', priceData)
                priceData = priceData.asfreq(pd.infer_freq(priceData.index))
                priceDataToClient = None
        else: 
            print('The Input request is NUll')
            return 'Error: The Input request is NUll'
        
        """ Analysis on stock open data """
        priceData = priceData.Open
        
        ticker_symbol = stockAnalysInput['ticker_symbol']
        ticker_symbolN = ticker_symbol.split('.')[0]
 
        potential_trade_dict = get_buy_sell_days(priceData, buy_days_condtn, sell_days_condtn) 

        pot_buying_days = potential_trade_dict['potential buying days']   
        pot_selling_days = potential_trade_dict['potential selling days']   

        df_stocks = pd.DataFrame(index=pot_buying_days.index)

        df_stocks['buying_day'] = (pot_buying_days == 1)
        df_stocks['potential_selling_day'] = (pot_selling_days == 1)
        df_stocks['price'] = priceData

        df_stocks = df_stocks[(df_stocks.buying_day | df_stocks.potential_selling_day)]
        
        
       #########
        current_funds, current_shares, events_list, verbose_list = get_invest_return(df_stocks, init_fund)
        
        lastDay_stock_price= priceData[-1]
        
        finalFundValue = current_funds + round(lastDay_stock_price*current_shares, 2)
        
        last_actual_trans_date = events_list[-1][1]
    
        if (current_shares==0):
            last_analysis_date = last_actual_trans_date
            """ last_stock_price = priceData.get(key=last_actual_trans_date) """
            last_stock_price = priceData.loc[last_actual_trans_date]
        else: 
            last_analysis_date = priceData.index[-1]
            last_stock_price = lastDay_stock_price
        
        percntReturnOnInvestment = round((finalFundValue - init_fund)/init_fund,2)
        
        length_events = len(events_list)   
        
        if (length_events>0):  
            global isTrading_happened
            isTrading_happened=True
            

        # plot
        fig_anal = plt.figure(figsize=(10,5),dpi=200)          
        plt.plot(priceData, color='forestgreen')
        
        if (priceData.index[0].year == priceData.index[-1].year):
                           
            # Set the locator
            locator = mdates.MonthLocator()  # every month
            # Specify the format - %b gives us Jan, Feb...
            """ fmt = mdates.DateFormatter('%Y-%m') """
            """ fmt = mdates.DateFormatter('%Y-%b') """
            fmt = mdates.DateFormatter('%b')
            X = plt.gca().xaxis
            X.set_major_locator(locator)
            # Specify formatter
            X.set_major_formatter(fmt)            
            """ plt.xticks(rotation=10) """            
            title_string = r'Stock Analysis (year - %s)'%priceData.index[0].year
            
        else:                
            title_string = 'Stock Analysis'
     
        
        y_text_postn = int(priceData.min())+(int(priceData.max()) - int(priceData.min()))*0.35
        y_annotate_postn = int(priceData.min())+(int(priceData.max()) - int(priceData.min()))*0.75

        y_lims = (int(priceData.min()*.95), int(priceData.max()*1.05))
        shaded_y_lims = (int(priceData.min()*0.5), int(priceData.max()*1.5))
        
        if (isTrading_happened):
            sale_evnt = 0
            purch_evnt = 0 
            bbox_props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='white', pad=1, lw=1.5)           
                   

            for idx, event in enumerate(events_list):
                color = 'indianred' if event[0] == 'purchase_date' else 'blue'                
                plt.axvline(event[1], color =color, linestyle = '--', alpha=0.3)
                
                if (idx==length_events-2 or idx==length_events-1):
                    annotate_text = 'Purchase' if event[0] == 'purchase_date' else 'Sale'
                    plt.annotate(annotate_text,xy=(event[1], priceData.loc[event[1]]), xycoords='data',xytext=(10, -100), textcoords='offset points',fontsize=10,arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=-90,angleB=10,rad=5", color='k', alpha=0.75), color =color, alpha=0.95)
                
                if event[0] == 'sale_date':
                    if sale_evnt == 0:
                        d1 = events_list[idx-1][1]
                        d2= event[1]
                        """ d1 = datetime.strptime(events_list[idx-1][1],"%Y-%m-%d") 
                        d2= datetime.strptime(event[1],"%Y-%m-%d") """ 
                        mid_date = (d2-d1)/2+d1
                        
                        plt.text(mid_date, y_text_postn, 'Stock-holding', ha="center", va="center", fontsize=10, bbox=bbox_props, color='black', alpha=0.85 )
                        sale_evnt = None
                    plt.fill_betweenx(range(*shaded_y_lims), event[1], events_list[idx-1][1], color='k', alpha =0.04)
                    
                                
                              
        plt.ylim(*y_lims)

       # label including this form1 will have these properties "#05DBC4"
        form1 = {'family': 'serif', 'color': 'blue', 'size': 15}
 
# label including this form2 will have these properties
        form2 = {'family': 'serif', 'color': 'maroon', 'size': 20, 'weight':'bold'}
        plt.xlabel("Date", fontdict=form1)
        plt.ylabel("Value (in INR)", fontdict=form1)
        """ plt.ylabel("INR", fontdict=form1) """
        plt.title(r"'%s' %s"%(ticker_symbolN, title_string) , fontdict=form2, loc='center')
        
        fig_anal.text(0.5,0.82,r"(with optimum buy and sell times for given Input)", ha="center", va="bottom", fontsize=15,color='k', alpha =0.85)
        
        """ plt.savefig(r"stock_analysed.png") """

        tmpfile_anal = BytesIO()
        fig_anal.savefig(tmpfile_anal, format='png',bbox_inches='tight', pad_inches=0.25 )
        
        tmpfile_anal.seek(0)
        encoded_analysedStock = base64.b64encode(tmpfile_anal.getvalue()).decode('utf-8')
        
        response_Analysobject = {
            'ticker_symbol':ticker_symbol,
            'plot_stockAnal':encoded_analysedStock,
            'initialFundValue':init_fund,
            'buySellEvents': events_list,
            'buySellInf': verbose_list,
            'tradingBooolean': isTrading_happened,
            'currentFundValue':finalFundValue,
            'current_shares':current_shares,
            'pcntReturn':percntReturnOnInvestment,
            'last_analysis_date':last_analysis_date,
            'last_stock_price':last_stock_price,
            'priceDatainRequest':priceDatainRequest,
            'priceDataToClient':priceDataToClient              
        }  
        

        return response_Analysobject

        
        ########
       
       

def get_stock_dict():
    return stock_names_dict

def get_stock_symbols():
    return stock_symbols

def get_stock_names():
    return stock_names

if __name__ == '__main__':
    load_artifacts()
    #print(predict_hprice('1st block jayanagar',1800, 2, 2, 2))
    #print(get_house_features())

