import requests
import pickle
import os
import pandas as pd
import numpy as np
import datetime as dt
from api_connector import API_Connector



class Reddit_API(API_Connector):
    def __init__(self,):
        super().__init__()
    def pickle_maker(self,pickle_name):
            self.your_pickle={}
            self.your_pickle['personal use script'] = input('personal use script: ')
            self.your_pickle['Token'] = input('Token')
            self.your_pickle['username'] = input('username')
            self.your_pickle['password'] = input('password')
            self.your_pickle['app name'] = input('app name')
            with open(self.pickle_name,'wb') as f:
                pickle.dump(self.your_pickle, f)
            self.your_pickle=pickle.load(open(self.pickle_name+'.pkl','rb'))
    
    def connect_reddit(self,pickle_name = None):
        if pickle_name is not None:
            self.your_pickle = pickle.load(open(pickle_name+'.pkl','rb'))
        auth = requests.auth.HTTPBasicAuth(self.your_pickle['personal use script'], self.your_pickle['Token'])
        data = {'grant_type': 'password',
            'username': self.your_pickle['username'],
            'password': self.your_pickle['password']}
        headers = {'User-Agent': self.your_pickle['app name']+'/0.0.1'}
        self.res = requests.post('https://www.reddit.com/api/v1/access_token',
                    auth=auth, data=data, headers=headers)
        TOKEN = self.res.json()['access_token']
        self.headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}
        print(requests.get('https://oauth.reddit.com/api/v1/me', headers=self.headers))
    
    def sub_reddit_pull(self,subreddit,search="top/?t=all",features = 'choice',params = {'limit': 100}):
        res = requests.get("https://oauth.reddit.com/r/"+subreddit+ "/"+search,params = params,
                     headers=self.headers)
        df = pd.DataFrame()
        for i, post in enumerate(res.json()['data']['children']):
            post_data = post['data']
            if features == 'choice':
                features = ['title','selftext','gilded','ups','upvote_ratio','created','num_comments','downs','likes','media',
                        'author_premium','is_original_content','author','id']
            if features == 'all':
                features = list(res.json()['data']['children'][0]['data'].keys())
            
            for feature in features:
                try:
                    df.loc[i,feature] = post_data[feature]
                except:
                    df.loc[i,feature] = 'Not found'
            df.loc[i,'kind_id'] = res.json()['data']['children'][i]['kind'] +'_'+ post_data['id']
        time_converter =lambda x: dt.datetime.fromtimestamp(x).strftime('%d-%B-%Y %H:%M') 
        try:
            df['created'] = pd.to_datetime(list(map(time_converter,df['created'])))
        except: pass

        return df
    def user_comment_pull(self,userid,params = {'limit': 100},return_userid = False):
        res = requests.get("https://oauth.reddit.com/user/"+userid+'/comments/',params = params,
                    headers=self.headers)
        df = pd.DataFrame()
        for i,_ in enumerate(res.json()['data']['children']):
            comment = res.json()['data']['children'][i]['data']
            df.loc[i,'created_utc'] = dt.datetime.fromtimestamp(comment['created_utc'])
            df.loc[i,'created_cst'] = dt.datetime.fromtimestamp(comment['created_utc']) - dt.timedelta(seconds = 18000)
            df.loc[i,'subreddit'] = comment['subreddit']
            df.loc[i,'ups'] = comment['ups']
            df.loc[i,'kind'] = res.json()['data']['children'][i]['kind']
            df.loc[i,'id'] = comment['id']
            df.loc[i,'body'] = comment['body']
            df.loc[i,'comment_length'] = len(comment['body'])
        if return_userid: return df,userid
        else: return df
    def user_post_pull(self,userid,params = {'limit':100},return_userid = False):
        res = requests.get("https://oauth.reddit.com/user/"+userid+'/submitted/', params = params,
                        headers = self.headers)
        df = pd.DataFrame()
        for i,_ in enumerate(res.json()['data']['children']):
            post = res.json()['data']['children'][i]['data']
            df.loc[i,'created_utc'] = dt.datetime.fromtimestamp(post['created_utc'])
            df.loc[i,'created_cst'] = dt.datetime.fromtimestamp(post['created_utc']) - dt.timedelta(seconds = 18000)
            features = ['subreddit','ups','upvote_ratio','url','title','selftext','num_comments','category',
                    'link_flair_type','is_video']
            for f in features: df.loc[i,f] = post[f]
        if return_userid: return df,userid
        else: return df
    def other_subreddits_active_on(self,subreddit,search = 'top',type_of_activity = 'post',params = {'limit':100}):
        ''' Returns dataframe of the other subreddits posters of the subject subreddit post to'''
        subreddit_df = self.sub_reddit_pull(subreddit,search = search)
        df = pd.DataFrame()
        for i,user in enumerate(subreddit_df.author):
            if type_of_activity =='post':
                user_hist, userid = self.user_post_pull(user,params = params,return_userid = True)
            if type_of_activity == 'comment':
                user_hist, userid = self.user_comment_pull(user,params = params,return_userid = True)
            df.loc[i,'user'] = user
            try:
                for user_subreddit, frame in user_hist.groupby(by='subreddit'):
                    if user_subreddit == subreddit:
                        continue
                    df.loc[i,user_subreddit] = len(frame)
            except:pass
        post_sum_subreddits_df = pd.DataFrame()
        for i, col in enumerate(df.columns):
            if col == 'user':continue
            if col.lower() == subreddit.lower(): continue
            post_sum_subreddits_df.loc[i,'subreddit'] = col
            post_sum_subreddits_df.loc[i,'num_of_posts'] = np.nansum(df[col])
            post_sum_subreddits_df.loc[i,'num_of_users_contributed'] = sum(~df[col].isna())
        
        return post_sum_subreddits_df.sort_values(by = 'num_of_users_contributed',ascending=False)\
            .reset_index(drop=True)  
    def post_search(self, search_term, limit = 1000):
        '''
        returns an array of all the subreddits with posts containing the search_term.
        
        Purpose: The end goal is to be able to search comments containing a search term in order to 
        perform a sentiment and topic analysis; however, after an exhaustive search through Reddit API and
        r/redditdev it does not appear that there is a built in method for doing this.  
        '''
        cols = ['created','created_utc','subreddit','subreddit_id', 'selftext','author_fullname', 'title','name',
            'id','author','num_comments','url', 'upvote_ratio','ups', ]
        # format search term
        search_term = search_term.replace(' ','%')
        list_of_dfs = [] 
        for i in range(int(limit/100)):
            # this loop updates the 'after' parameter to the last post of the previous search
            if i == 0:
                params = {'limit':100}
            else:
                try: 
                    params = {'limit':100, 'after':list(new_df.name)[-1]}
                except AttributeError: # new_df is empty which probably means their is no more data
                    return pd.concat(list_of_dfs)[cols]
            res = requests.get(f"https://oauth.reddit.com/search/?q={search_term}&include_over_18=1&type=comment",
                    params = params,
                        headers=self.headers)
            new_df = pd.DataFrame([res.json()['data']['children'][j]['data'] for j in range(len(res.json()['data']['children']))])
            try: list_of_dfs.append(new_df)
            except: pass
        df = pd.concat(list_of_dfs)[cols].reset_index(drop=True)
        df['created'] = pd.to_datetime(df['created'], unit = 's')
        df['created_utc'] = pd.to_datetime(df['created_utc'], unit = 's')
        return df

def test():
    reddit = Reddit_API()
    reddit.connect_reddit('./monkey')
    subreddit = 'homesteading'
    df = reddit.sub_reddit_pull(subreddit,search="top/?t=all",features = 'choice',params = {'limit': 100})
    print(df.head())

def test_search():
    reddit = Reddit_API()
    reddit.connect_reddit('./monkey')
    search_term = 'alcohol'
    df = reddit.post_search(search_term=search_term)
    print(df.shape)
    print(df.head())


if __name__ == '__main__':
    test_search()