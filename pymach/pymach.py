# -*- coding: utf-8 -*-
from analyze import Analyze
from prepare import Prepare

obj = Analyze(className='species')
obj.read("iris.csv")
#print obj.data
#print obj.description()
#print obj.classBalance()
#print obj.hist()
#print obj.density()
#print obj.corr()
print(obj.scatter())

