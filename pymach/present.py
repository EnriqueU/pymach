# Standard Libraries
import os
import subprocess
# Local Libraries
import define
import analyze
import prepare
import fselect
import evaluate
import improve
import tools

import pandas as pd

from flask import Flask, render_template, \
        redirect, request, url_for, jsonify, flash
from werkzeug.utils import secure_filename
from collections import OrderedDict

import sys # print en consola


app = Flask(__name__)
app.secret_key = 'some_secret'

APP_PATH = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_DIR'] = os.path.join(APP_PATH, 'uploads')
app.config['MODELS_DIR'] = os.path.join(APP_PATH, 'models')
app.config['MARKET_DIR'] = os.path.join(APP_PATH, 'market')
ALLOWED_EXTENSIONS = ['txt', 'csv', 'ml', 'html']


def report_analyze(figures, response, data_path, data_name):

    definer = define.Define(
            data_path=data_path,
            header=None,
            response=response).pipeline()
            # Here read and save the description of the data

    analyzer = analyze.Analyze(definer)
    dict_figures = OrderedDict()
    table = analyzer.description()

    for fig in figures:
        data_name = data_name.replace(".csv", "")
        plot_path = os.path.join(app.config['MARKET_DIR'], data_name, 'analyze')
        tools.path_exists(plot_path)
        plot_path_plot = os.path.join(plot_path, fig+'.html')

        dict_figures[fig] = analyzer.plot(fig)
        analyzer.save_plot(plot_path_plot)

    dict_report = {'plot': dict_figures, 'table': table}

    return dict_report


def report_model(response, data_path, data_name, problem_type):
    definer = define.Define(data_path=data_path,header=None,
            response=response,problem_type=problem_type).pipeline()
    preparer = prepare.Prepare(definer).pipeline() # scaler
    selector = fselect.Select(definer).pipeline() # pca
    evaluator = evaluate.Evaluate(definer, preparer, selector).pipeline()

    plot = evaluator.plot_models()
    table = evaluator.report

    data_name = data_name.replace(".csv", "")
    plot_path = os.path.join(app.config['MARKET_DIR'], data_name, 'model')
    tools.path_exists(plot_path)
    plot_path_plot = os.path.join(plot_path, 'boxplot.html')
    evaluator.save_plot(plot_path_plot)
    plot_path_report = os.path.join(plot_path, 'report.csv')
    evaluator.save_report(plot_path_report)

    dict_report = {'plot': plot, 'table': table}

    return dict_report


def report_improve(response, data_path):
    definer = define.Define(data_path=data_path,header=None,response=response).pipeline()
    preparer = prepare.Prepare(definer).pipeline()
    selector = fselect.Select(definer).pipeline()
    evaluator = evaluate.Evaluate(definer, preparer, selector)
    improver = improve.Improve(evaluator).pipeline()

    plot = improver.plot_models() # aqui esta el error
    table = improver.report
    dict_report = {'plot': plot, 'table': table}

    return dict_report

def report_market(data_name):

    # analyze_report = OrderedDict()
    # model_report = OrderedDict()

    data_name = data_name.replace(".csv", "")
    app_path = os.path.join(app.config['MARKET_DIR'], data_name)
    # app_dirs = os.listdir(app_path)

    # Show Model info
    try:
        model_path = os.path.join(app_path, 'model')
        plot_model = ''
        with open(os.path.join(model_path, 'boxplot.html')) as f:
            plot_model = f.read()

        table_model = pd.read_csv(os.path.join(model_path, 'report.csv'))
        dict_report_model = {'plot':plot_model, 'table':table_model}  # return 1
    except:
        dict_report_model = {'plot':None, 'table':None}  # return 1


    # Show Analyze info
    try:
        analyze_path = os.path.join(app_path, 'analyze')
        plot_analyze = OrderedDict()
        for plot in os.listdir(analyze_path):
            with open(os.path.join(analyze_path, plot)) as f:
               fig = plot.replace('.html', '')
               plot_analyze[fig] = f.read()

        # Join full report: model and analyze
        dicts_market = {'model':dict_report_model, 'analyze':plot_analyze}
    except:
        dicts_market = {'model':dict_report_model, 'analyze':None}  # return 2


    return dicts_market

def allowed_file(file_name):
    return '.' in file_name and file_name.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


########################### Start Upload Button ##################################
@app.route('/')
@app.route('/defineData', methods=['GET', 'POST'])
def defineData():
    """  Show the files that have been uploaded """
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    dirs.sort(key=str.lower)
    return render_template('uploadData.html', files=dirs)


@app.route('/storeData', methods=['GET', 'POST'])
def storedata():
    """  Upload a new file """
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    dirs.sort(key=str.lower)

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template(
                'uploadData.html',
                infoUpload='Chosse a file .csv',
                files=dirs)

        file = request.files['file']
        file_name = ''
        data_name = ''

        if file.filename == '':
            return render_template(
                'uploadData.html',
                infoUpload='file not selected',
                files=dirs)

        if file and allowed_file(file.filename):
            file_name = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_DIR'], file_name)
            file.save(file_path)
            dirs = os.listdir(app.config['UPLOAD_DIR'])
            dirs.sort(key=str.lower)
            return render_template(
                'uploadData.html',
                infoUpload='Uploaded!! '+file_name,
                files=dirs)

        return render_template(
            'uploadData.html',
            infoUpload='Error',
            files=dirs)
    else:
        return redirect(url_for('defineData'))


@app.route('/chooseData', methods=['GET', 'POST'])
def chooseData():
    """  choose a file and show its content """
    from itertools import islice
    # tools.localization()

    file_name = ''
    data_name = ''
    data_path = ''
    dire = ''
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    if request.method == 'POST':
        file_name = request.form['submit']
        data_name = file_name.replace(".csv", "")
        data_path = os.path.join(app.config['UPLOAD_DIR'], data_name+'.html')

    try: # si existe el archivo .html
        dataset = None
        with open(data_path) as f:
            dataset = f.read()
    except: # si no existe el archivo .html
        data_path = os.path.join(app.config['UPLOAD_DIR'], file_name)
        dire = open(data_path)
        return render_template(
                'uploadData.html',
                files=dirs,
                dataset2 = dire,
                data_name=data_name)

    return render_template(
            'uploadData.html',
            files=dirs,
            dataset=dataset, # se pasa matrix o html
            data_name=data_name)


# ########################## Start Analyze Button ##################################
@app.route('/analyze_base', methods=['GET', 'POST'])
def analyze_base():
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    dirs.sort(key=str.lower)
    return render_template('analyzeData.html', files=dirs)


@app.route('/analyze_app', methods=['GET', 'POST'])
def analyze_app():
    figures = ['Histogram', 'Boxplot', 'Correlation']
    response = "class"
    data_name = ''
    data_path = ''
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    dirs.sort(key=str.lower)
    if request.method == 'POST':
        data_name = request.form['submit'] # pide el nombre
        # se busca el archivo en el directorio y se guarda la ruta
        data_path = os.path.join(app.config['UPLOAD_DIR'], data_name)

    return render_template(
            'analyzeData.html',
            files=dirs,
            figures=report_analyze(figures, response, data_path, data_name),
            data_name=data_name)

########################### End Analyze Button ##################################

########################### Start Model Button ##################################
@app.route('/model_base', methods=['GET', 'POST'])
def model_base():
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    dirs.sort(key=str.lower)
    return render_template('models.html', files=dirs)


@app.route('/model_app', methods=['GET', 'POST'])
def model_app():
    response = "class"
    data_name = ''
    data_path = ''
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    dirs.sort(key=str.lower)
    if request.method == 'POST':
        problem_type = request.form['typeModel']
        data_name = request.form['submit']
        data_path = os.path.join(app.config['UPLOAD_DIR'], data_name)

    return render_template(
            'models.html',
            files=dirs,
            report=report_model(response, data_path, data_name, problem_type),
            data_name=data_name)

########################### End Model Button ##################################

########################### Start Improve Button ##################################
@app.route('/improve_base', methods=['GET', 'POST'])
def improve_base():
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    dirs.sort(key=str.lower)
    return render_template('improve.html', files=dirs)

@app.route('/improve_app', methods=['GET', 'POST'])
def improve_app():
    response = "class"
    data_name = ''
    data_path = ''
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    dirs.sort(key=str.lower)
    if request.method == 'POST':
        data_name = request.form['submit']
        data_path = os.path.join(app.config['UPLOAD_DIR'], data_name)

    return render_template(
            'improve.html',
            files=dirs,
            report=report_improve(response, data_path),
            data_name=data_name)

########################### End Improve Button ##################################

########################### Start Model Button ##################################
@app.route('/market_base', methods=['GET', 'POST'])
def market_base():
    dirs = os.listdir(app.config['MARKET_DIR'])
    dirs.sort(key=str.lower)
    return render_template('market.html', files=dirs)


@app.route('/market_app', methods=['GET', 'POST'])
def market_app():
    response = "class"
    data_name = ''
    data_path = ''
    dirs = os.listdir(app.config['MARKET_DIR'])
    dirs.sort(key=str.lower)
    if request.method == 'POST':
        data_name = request.form['submit']
        # data_path = os.path.join(app.config['MARKET_DIR'], data_name)
    return render_template(
            'market.html',
            files=dirs,
            report=report_market(data_name),
            data_name=data_name)

########################### End Market Button ##################################

# @app.route('/prediction', methods=['GET', 'POST'])
# def prediction():
    # attributes = []
    # dirs = os.listdir(app.config['UPLOAD_DIR'])
    # data_class = 'class'
    # file_name = 'iris.csv'
    # filepath = os.path.join(app.config['UPLOAD_DIR'], file_name)
    # model = 'Naive Bayes'
    # f = open(filepath, 'r')
    # g = open(filepath, 'r')
    # for item in g.readline().split(','):
        # if item.strip() != data_class:
            # attributes.append(item)
    # print(attributes, ' this is something')
    # return render_template('showPrediction.html', file = f, attributes = attributes, data_class = data_class, model = model)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8002)
    #falta: para mensaje flush
        #app.secret_key = 'some_secret'
