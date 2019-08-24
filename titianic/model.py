'''
Variable	Definition	Key

Survival 생존여부 	0 = No, 1 = Yes
pclass 승선권	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
Sex	성별
Age 나이
sibsp	동반한 형제 자매 배우자# of siblings / spouses aboard the Titanic
parch	동반한 부모 자식# of parents / children aboard the Titanic
ticket	티켓번호Ticket number
fare	티켓의 요금Passenger fare
cabin	객실번호Cabin number
embarked	승선한 항구명Port of Embarkation	C = Cherbourg(쉐부로), Q = Queenstown(퀸즈타운), S = Southampton(사우스햄톤)
'''
import pandas as pd
import numpy as np
class Tmodel:
    def __init__(self):
        self._context = None
        self._fname = None
        self._train = None
        self._test = None
        self._test_id = None

    @property#=feature(set_name of dims)
    def context(self) -> object:return self._context
    @context.setter
    def context(self,context): self._context = context

    @property
    def context(self) -> object: return self._fname
    @context.setter
    def context(self, context): self._fname = fname

    @property
    def context(self) -> object: return self._train
    @context.setter
    def context(self, context): self._train = train

    @property
    def context(self) -> object: return self._test
    @context.setter
    def context(self, context): self._test = test

    @property
    def context(self) -> object: return self._test_id
    @context.setter
    def context(self, context): self._test_id = test_id

    def new_file(self) -> str: return self.context + self.fname

    def new_dfame(self) -> object:
        file = self.new_file()
        return pd.read_csv(file)

    def hook_process(self,train,test) -> object:
        print('-'*12,1,'-'*12)
        print('-'*12,2,'-'*12)
        print('-'*12,3,'-'*12)

