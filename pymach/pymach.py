#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

"""
This module provides the logic of the whole project.

"""
import define
import analyze
import prepare
import feature_selection
import evaluate

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn import cross_validation
import pandas as pd


#name = "inputLocalizacion.csv"
name = "iris.csv"
#name = "inputBus.csv"
# className = "Ruta"
#className = "position"
className = "class"

def main():
    #STEP 0: Define workflow parameters
    definer = define.Define(nameData=name, className=className).pipeline()

    #STEP 1: Analyze data by ploting it
    #analyze.Analyze(definer).pipeline()

    #STEP 2: Prepare data by scaling, normalizing, etc. 
    preparer = prepare.Prepare(definer).pipeline()

    #STEP 3: Feature selection
    featurer = feature_selection.FeatureSelection(definer).pipeline()

    #STEP4: 
    evaluator = evaluate.Evaluate(definer, preparer, featurer).pipeline()

if __name__ == '__main__':
    main()
