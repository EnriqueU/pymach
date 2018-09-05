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

from flask_login import LoginManager
#from flask_googlelogin import GoogleLogin

app = Flask(__name__)
app.secret_key = 'some_secret'

APP_PATH = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_DIR'] = os.path.join(APP_PATH, 'uploads')
app.config['MODELS_DIR'] = os.path.join(APP_PATH, 'models')
app.config['MARKET_DIR'] = os.path.join(APP_PATH, 'market')
ALLOWED_EXTENSIONS = ['txt', 'csv', 'ml', 'html']

login_manager = LoginManager()
login_manager.init_app(app)
#googlelogin = GoogleLogin(app, login_manager)


def report_analyze(figures, data_path, data_name,tipo='normal'):
    # tipo indica normalizar o no los datos
    # Here read and save the description of the data
    definer = define.Define(data_path=data_path,data_name=data_name).pipeline()
    if tipo=='normal':
    	analyzer = analyze.Analyze(definer).pipeline()
    elif tipo=='real':
        analyzer = analyze.Analyze(definer).pipelineReal()

    table1 = definer.describe
    table2 = analyzer.describe
    dict_figures = OrderedDict()

    for fig in figures:
        data_name = data_name.replace(".csv", "")
        plot_path = os.path.join(app.config['MARKET_DIR'], data_name, 'analyze')
        tools.path_exists(plot_path)
        plot_path_plot = os.path.join(plot_path, fig+'.html')

        dict_figures[fig] = analyzer.plot(fig)
        analyzer.save_plot(plot_path_plot)

    dict_report = {'plot': dict_figures, 'table1': table1, 'table2' :  table2}
    return dict_report

def report_model(response, data_path, data_name, problem_type):
    definer = define.Define(data_path=data_path,data_name=data_name,problem_type=problem_type).pipeline()
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


def report_improve(data_path, data_name, problem_type, optimizer, modelos):
    definer = define.Define(data_path=data_path,data_name=data_name,problem_type=problem_type).pipeline()
    preparer = prepare.Prepare(definer).pipeline()
    selector = fselect.Select(definer).pipeline()
    evaluator = evaluate.Evaluate(definer, preparer, selector)
    improver = improve.Improve(evaluator, optimizer, modelos).pipeline()

    plot = improver.plot_models()
    table = improver.report
    dict_report = {'plot': plot, 'table': table}
    #dict_report = {'table': table}

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
@app.route("/home")
def dashboard():
    return render_template("home.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Here we use a class of some kind to represent and validate our
    # client-side form data. For example, WTForms is a library that will
    # handle this for us, and we use a custom LoginForm to validate.
    form = LoginForm()
    if form.validate_on_submit():
        # Login and validate the user.
        # user should be an instance of your `User` class
        login_user(user)

        flask.flash('Logged in successfully.')

        next = flask.request.args.get('next')
        # is_safe_url should check if the url is safe for redirects.
        # See http://flask.pocoo.org/snippets/62/ for an example.
        if not is_safe_url(next):
            return flask.abort(400)

        return flask.redirect(next or flask.url_for('index'))
    return flask.render_template('login.html', form=form)

@app.route('/defineData', methods=['GET', 'POST'])
@login_required
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
            flash('Chosse a file .csv',"alert alert-danger")
            return render_template(
                'uploadData.html',
                infoUpload='Chosse a file .csv',
                files=dirs)

        file = request.files['file']
        file_name = ''
        data_name = ''

        if file.filename == '':
            flash('file not selected',"alert alert-danger")
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
            flash('Uploaded!! '+file_name,"alert alert-success")
            return render_template(
                'uploadData.html',
                infoUpload='Uploaded!! '+file_name,
                files=dirs)

        flash('Error',"alert alert-danger")
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
    dirs.sort(key=str.lower)
    if request.method == 'POST':
        file_name = request.form['submit']
        data_name = file_name.replace(".csv", "")
        data_path = os.path.join(app.config['UPLOAD_DIR'], file_name)
        dire = open(data_path)
        return render_template(
                'uploadData.html',
                files=dirs,
                dataset = dire,
                data_name=data_name)
    else:
        return render_template(
            'uploadData.html',
            infoUpload='Error',
            files=dirs)

########################### End Upload Button ##################################

# ########################## Start Analyze Button ##################################
@app.route('/analyze_base', methods=['GET', 'POST'])
def analyze_base():
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    dirs.sort(key=str.lower)
    return render_template('analyzeData.html', files=dirs)


@app.route('/analyze_app', methods=['GET', 'POST'])
def analyze_app():
    figures = ['Histogram', 'Boxplot', 'Correlation']
    data_name = ''
    data_path = ''
    archivo = ''
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    dirs.sort(key=str.lower)
    if request.method == 'POST':
        data_name = request.form['submit']
        data_path = os.path.join(app.config['UPLOAD_DIR'], data_name)
        tipo = request.args.get('tipo', default = 'real', type = str)
        #if tipo=='normal':
        figures1=report_analyze(figures, data_path, data_name)
        #elif tipo=='real':
        figures2=report_analyze(figures,data_path, data_name,tipo='real')
    else:
        return redirect(url_for('analyze_base'))

    return render_template(
            'analyzeData.html',
            files=dirs,
	        figures1=figures1,
            figures2=figures2,
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
    data_name = ''
    data_path = ''
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    dirs.sort(key=str.lower)
    if request.method == 'POST':
        optimizer = request.form['search']
        problem_type = request.form['typeModelRC']
        modelos = request.form.getlist('typeModel')
        # ---------------------------------------------------------------------
        data_name = request.form['submit'] # choosed data
        data_path = os.path.join(app.config['UPLOAD_DIR'], data_name)

    return render_template(
            'improve.html',
            files=dirs,
            report=report_improve(data_path, data_name, problem_type, optimizer, modelos),
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
