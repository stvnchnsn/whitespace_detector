import pickle
import os

def test():
    conn = API_Connector('reddit_pickle')
    conn.pickle_sign_in()
protocol = test

class API_Connector():
    def __init__(self,):
        pass
    def pickle_sign_in(self,pickle_name):
        self.pickle_name = pickle_name
        if not os.path.exists(self.pickle_name+'.pkl'):
            self.pickle_maker()
        else:
            self.your_pickle=pickle.load(open(self.pickle_name+'.pkl','rb'))
            print('found your pickle!')
    def pickle_maker(self):
        '''will vary by API and likely need to be frequently updated, so the pickle_maker
        will be maintained at the child class level'''
        pass

if __name__ == '__main__':
    protocol()