import pandas as pd
import numpy as np
from titianic.model import Tmodel
#'C:/Users/ezen/PycharmProjects/tensorflow20190824/titiani/data/'
class Tcontroller:
    def __init__(self):
        self._m = Tmodel()
        self._context = './data/'
        self._train = self.create_train()

    def create_train(self) -> object:
        m = self._m
        m.context = self._context
        m.fname = 'train.csv'
        t1 = m.new_dfame()
        print('-'*12+'train head & column'+'-'*12)
        print(t1.head())
        print(t1.columns)

        m.fname = 'test.csv'
        t2=m.new_dfame()

        train = m.hook_process(t1,t2)
        # print('-' * 12 + 'train head & column' + '-' * 12)
        # print(train.head())
        # print(train.columns)