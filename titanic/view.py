import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from titanic.model import Tmodel
class TitanicView:
    def __init__(self):
        self._m = Tmodel
        self._context = './data/'
    def create_train(self) -> object:
        m = self._m
        m.context = self._context
        m.fname = 'train.csv'
        t = m.new_dfame()
        return t
    @staticmethod
    def plot_survived_dead(train):
        # print('* 전달된 t : \n')
        # print(train)
        f, ax = plt.subplots(1, 2, figsize=(18,8))
        # subplot 자동완성 오류 -> subplots 으로
        trian['Survived'].value_counts().plot.pie(explode=[0,0.1],
                                                  autopct='%1.1f%%',
                                                  ax =ax[0],
                                                  shadow=True)
        ax[0].set_title('Survived')
        ax[1].set_ylabel('')

        sns.countplot('Survived', data=train, ax=ax[1])
        ax[1].set_title('Survived')
        plt.show()

    @staticmethod
    def plot_sex(train):
        f, ax = plt.subplots(1,2, figsize=(18,8))
        train['Survived'][train['Sex']=='male'].value_counts().plot.pie(explode=[0,0.1],
                                                                        autopct='%1.1f%%',
                                                                        ax = ax[0],
                                                                        shadow=True)
        train['Survived'][train['Sex'] == 'female'].value_counts().plot.pie(explode=[0, 0.1],
                                                                          autopct='%1.1f%%',
                                                                          ax=ax[1],
                                                                          shadow=True)

        ax[0].set_title('Male')
        ax[1].set_title('Female')

        plt.show()

    @staticmethod
    def bar_chart(train, feature):
        survived = train[train['Survived']==1][feature].value_counts()
        dead = train[train['Survived']==1][feature].value_counts()
        df = pd.DataFrame([survived,dead])
        df.index = ['survived', 'dead']
        df.plot(kind='bar', stacked=True, figsize=(110,1))
        plt.show()


