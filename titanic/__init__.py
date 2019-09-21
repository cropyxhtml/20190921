from titanic.controller import Tcontroller
from titanic.view import TitanicView
if __name__ == '__main__':
    # ctrl = Tcontroller()
    # t = ctrl.creat_train()
    def print_menu():
        print('0. EXIT')
        print('1. LEARNING MACHINE')
        print('2. VIEW : plot_survived_dead')
        print('3. TEST ACCURACY')
        print('4. SUBMIT')
        return input('CHOOSE ONE \n')
    while 1:
        menu = print_menu()
        print('MENU : %s' % menu)

        if menu =='0':
            print('** EXIT **')
            break

        elif menu =='1':
            print('** CREATE TRIAN **')
            ctrl = Tcontroller()
            t = ctrl.create_train()
            print('** T 모델 **')
            print(t)
            break

        elif menu =='2':
            view = TitanicView()
            t = view.create_train()
            # view.plot_survived_dead()
            # view.plot_sex()
            view.bar_chart(t,'Pclass')
            break

        elif menu =='3':
            ctrl = Tcontroller()
            t = ctrl.create_train()
            ctrl.test_all()
            break

        elif menu =='4':
            ctrl = Tcontroller()
            t = ctrl.create_train()
            ctrl.submit()
            break
