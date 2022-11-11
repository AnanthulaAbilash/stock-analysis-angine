from flask import Flask, request, jsonify, send_file
import util
""" from datetime import datetime """
from flask_cors import CORS, cross_origin
""" from flask_sqlalchemy import SQLAlchemy
from base64 import b64encode """
from waitress import serve


app = Flask(__name__)
app.secret_key = 'ShivaShakti'
cors = CORS(app, resources={r'/*':{'origins':'*'}})
app.config['CORS_HEADERS'] = 'Content-Type'
""" app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stockImages.sqlite3'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False """

""" db = SQLAlchemy(app) """

""" class stockImages(db.Model):
    _id = db.Column("id", db.Integer, primary_key=True)
    name = db.Column(db.String(200))
    img = db.Column(db.String)
    
    def __init__(self, name, img) -> None:
        super().__init__()
        self.name = name
        self.img = img """

@app.route('/get_stocks', methods = ['GET'])
@cross_origin()
def get_stock_dict():
    response = jsonify(
         util.get_stock_dict()
    )
    
    """ response.headers.add('Access-Control-Allow-Origin', '*') """
    
    return response

@app.route('/get_stocksSymbols', methods = ['GET'])
@cross_origin()
def get_stock_symbols():
    response = jsonify({
        "symbols":util.get_stock_symbols()
    }        
    )
    
    return response

""" @cross_origin(origins=['http://localhost:3000']) """

@app.route('/get_stocksNames', methods = ['GET'])

def get_stock_names():
    response = jsonify({
        "names":util.get_stock_names()
    })
    
    """ response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*") """
    
    return response




@app.route('/analyse_return', methods = ['GET', 'POST'])
@cross_origin()
def analyse_stock_returnServer():
    
    print("Inside the analyse route")
    
    try:
    
        init_fund = float(request.json['init_fund'])
        ticker_symbol = request.json['ticker_symbol']
        history_freq = request.json['history_freq']
        history_start = request.json['history_start']    
        history_end = request.json['history_end']    
        buy_days_condtn = int(request.json['buy_days_condtn'])
        sell_days_condtn = int(request.json['sell_days_condtn'])
        if('price_data' in request.json):
            price_data = request.json['price_data']  
        else: price_data = False
        
        print('predict req is', init_fund, ticker_symbol, history_freq, history_start, history_end, buy_days_condtn, sell_days_condtn)
        
        stockAnalysInput = {'ticker_symbol': ticker_symbol, 'history_freq':history_freq, 'history_start':history_start, 'history_end':history_end, 'init_fund':init_fund, 'buy_days_condtn':buy_days_condtn, 'sell_days_condtn':sell_days_condtn}
            
        response = jsonify(
            util.analyse_stock_return(stockAnalysInput, price_data)
        )      
            
        """ print("response is......", util.predict_stock_return(init_fund, ticker_symbol, history_freq, history_start, history_end, buy_days_condtn, sell_days_condtn)) """
    
    except Exception as e:        
        response = jsonify('{}, at API while fetching '.format(e))
    
    return response


@app.route('/display_stock', methods = ['GET', 'POST'])
@cross_origin()
def display_stock_data(): 
    print("Inside the display route")   
    try:
        ticker_symbol = request.json['ticker_symbol']
        history_freq = request.json['history_freq']
        history_start = request.json['history_start']    
        history_end = request.json['history_end']
        price_data = request.json['price_data']  
        
        """ print('the price data from req is',price_data) """
            
        print('req is', ticker_symbol, history_freq, history_start, history_end)
        
        stockInputInf = {'ticker_symbol': ticker_symbol, 'history_freq':history_freq, 'history_start':history_start, 'history_end':history_end}
        
        
        
        response = jsonify(util.display_stock_data(stockInputInf=stockInputInf, price_data=price_data))   
            
        """ print("response in try is......", response) """
        
        """ imager = stockImages(name='stockDisplay',img=img_base64)
        db.session.add(imager)
        db.session.commit() 
        print('the db img', imager) """
        """ 
        mylist1 = stockImages.query.filter_by(name='stockDisplay').all()
        event = stockImages.query.get_or_404(id)
        image = b64encode(event.img)
        
        """
    except Exception as e:        
        response = jsonify('{}, at API while fetching '.format(e))       
   
    
    return response


@app.route('/plot_help', methods = ['GET', 'POST'])
@cross_origin()
def display_stock_help():  
    print("Inside the help route")  
    try:
        ticker_symbol = request.json['ticker_symbol']
        history_freq = request.json['history_freq']
        history_start = request.json['history_start']    
        history_end = request.json['history_end']  
        price_data = request.json['price_data']  
            
        print('req is', ticker_symbol, history_freq, history_start, history_end)
        
        stockInputInf = {'ticker_symbol': ticker_symbol, 'history_freq':history_freq, 'history_start':history_start, 'history_end':history_end}
        
        
        response_util = util.getStock_acf_pacf_plots(stockInputInf=stockInputInf, price_data=price_data) 
        
        """ print("response is......", response_util) """
        
        response = jsonify(response_util) 
        
    
    except Exception as e:        
        response = jsonify('{}, at API while fetching '.format(e))
    """ except ValueError as e:    
        response = jsonify('{}, at API while fetching '.format(e)) """     
    
    return response


if __name__ == '__main__':
    print('Server ready..')
    """ db.create_all() """
    util.load_artifacts()
    """ app.run(port=5000) """
    serve(app, port=5000)
    """ serve(app, port=5000, host="127.0.0.1") """