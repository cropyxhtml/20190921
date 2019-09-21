import pandas as pd
import numpy as np
from titanic.model import Tmodel
from sklearn.svm import SVC
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
        return train
        # print('-' * 12 + 'train head & column' + '-' * 12)
        # print(train.head())
        # print(train.columns)

    def create_model(self) -> object:
        train = self._train
        model = train.drop('Survived', axis=1)
        print('----- Model Info -----')
        print(model.info)
        return model

    def create_dummy(self) ->object:
        train =self._train
        dummy = train['Survived']
        return dummy

    def test_all(self):
        model = self.create_model()
        dummy = self.create_dummy()
        m = self._m
        m.hook_test(model, dummy)

    def submit(self):
        m = self._m
        model = self.create_model()
        dummy = self.create_dummy()
        test = m._test
        test_id = m.test_id

        clf = SVC()
        clf.fit(model,dummy)
        prediction = clf.predict(test)
        submission = pd.DataFrame(
            {'PassengerId': test_id,'Survived':prediction}
        )
        print('='*30)
        print(len(test_id),test_id)

        # print(submission.head())
        # print('m.context :',m.context)
        # submission.to_csv('./data/submission.csv',index=False)

