import pandas as pd
import numpy as np
from titianic.model import Tmodel

class Tcontroller:
    def __init__(self):
        self.m = Tmodel()
        self.context = './data/'
        # self.train =

    def creat_train(self) -> object:
        m = self.m
        m.context = self.context
        m.fname = 'train.csv'
        t1 = m.new_dfame()
        print('-'*12+'train head & column'+'-'*12)
        print(t1.head())
        print(t1.columns)