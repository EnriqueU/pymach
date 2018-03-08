#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides a few of useful functions (actually, methods)
for describing the dataset which is to be studied.

"""
from __future__ import print_function
import tools

import pandas as pd
import numpy as np
import plotly.plotly as py
import cufflinks as cf # Needed

from plotly.offline.offline import _plot_html
from plotly.offline import iplot
from collections import namedtuple

__all__ = [
    'read', 'description', 'classBalance', 'hist', 'density']

class Analyze():
    """ A class for data analysis """
    def __init__(self, definer):
        """The init class.

        Parameters
        ----------
        typeModel : string
            String that indicates if the model will be trained for clasification
            or regression.
        className : string
            String that indicates which column in the dataset is the class.
        """
        self.problem_type = definer.problem_type
        self.response = definer.response
        self.data = definer.data
        self.plot_html = None

    def description(self):
        """Shows a basic data description ."""

        description = self.data.describe().round(3)
        description = description.reset_index()
        return description

    def classBalance(self):
        """Shows how balanced the class values are."""

        return self.data.groupby(self.response).size()

    def plot_to_html(self, fig):
        plotly_html_div, plotdivid, width, height = _plot_html(
                figure_or_data=fig,
                config="",
                validate=True,
                default_width='90%',
                default_height="100%",
                global_requirejs=False)

        return plotly_html_div

    def histogram(self):
        fig = self.data.iplot(
                kind="histogram",
                asFigure=True,
                xTitle="Features",
                yTitle="Frequency",
                theme="white")

        self.plot_html = self.plot_to_html(fig)
        return self.plot_html

    def boxplot(self):
        fig = self.data.iplot(
                kind="box",
                asFigure=True,
                xTitle="Features",
                yTitle="Values",
                boxpoints="outliers",
                mode='markers',
                text=['Text A', 'Text B', 'Text C'],
                theme="white")

        self.plot_html = self.plot_to_html(fig)
        return self.plot_html

    def correlation(self):
        corr_data = self.data.corr()
        fig = corr_data.iplot(
                kind="heatmap",
                asFigure=True,
                xTitle="Features",
                yTitle="Features",
                theme="white")

        self.plot_html = self.plot_to_html(fig)
        return self.plot_html

    def scatter(self):
        columns = [x for x in self.data.columns if x != self.response]

        fig = self.data[columns].scatter_matrix(
                asFigure=True,
                title="Scatter Matrix",
                xTitle="Features",
                yTitle="Features",
                mode='markers',
                columns=columns,
                theme="white")

        self.plot_html = self.plot_to_html(fig)
        return self.plot_html

    def plot(self, name):
        if name == "Histogram":
            return self.histogram()
        elif name == "Boxplot":
            return self.boxplot()
        elif name == "Correlation":
            return self.correlation()
        elif name == "scatter":
            return self.scatter()

    def save_plot(self, name):
        with open(name, "w") as plot:
            plot.write(self.plot_html)
