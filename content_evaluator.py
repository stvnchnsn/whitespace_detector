# standard Python Libraries
import pickle
import os
import datetime as dt
import sqlite3
import string
# Common Python Libraries
import numpy as np
import nltk

# Other Python Libraries
import openai
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
NLP_SPACY = spacy.load("en_core_web_sm")
# Custom Python Libraries
from chatbot import ChatBot



class Content_Evaluator(ChatBot):
    def __init__(self,search_term = 'alcoholic beverages', api_file='api_key.pkl') -> None:
        super().__init__(api_file)
        self.analyzer = SentimentIntensityAnalyzer()
        self.search_term = search_term
        self.categories = ['Cocktail recipes and tutorials','Wine and beer recommendations','Party and event photos',
                      'Alcohol-related memes and humor','Reviews of bars and restaurants','Alcohol-related news and trends',
                      'Personal stories and experiences','Celebrity endorsements and sponsorships',
                    'Health and wellness tips','Advocacy and activism', f"Not related to {search_term}"]
        self.questions = {
            'category': f'''Which one of these categories does this post best fit into? {self.categories}''',
            'summary':'''Summarize the content in 3 words or less'''
                    }
    def get_completion(self,prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
        )
        return response.choices[0].message["content"]
    def relevance_prompt(self, content):
        prompt = f'''
        Does the content below (delimited by triple backticks) contain references to alcoholic consumption, alcoholic beverages or marketing? 
        Return either 'Yes' or 'No'.

        ```{content}```
        
        Answer:
        '''
        return prompt
    def vader_sentiment(self, content):
        '''
        Uses VADER to determine sentiment
        '''
        scores = self.analyzer.polarity_scores(content)
        sentiment = scores['compound']
        return sentiment
    def named_entity(self,content):
        doc = NLP_SPACY(content)
        entities = [entity.text for entity in doc.ents]
        if len(entities)==0: return None
        else: return entities

    def sentiment_prompt(self, content):
        prompt = f"""What is the sentiment of this post? (respond with "positive", "negative", or "neutral" only): {content}"""
        return prompt
    def evaluate_prompt(self, prompt,model = "text-ada-001",temperature = 0.02,
                        max_tokens = 1000, frequency_penalty = 0,top_p = 0.1 ):
        response = openai.Completion.create(
                    model=model,
                    prompt = prompt,
                    temperature = temperature,
                    max_tokens = max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=0,
                    top_p = top_p,
                    n=1)
            
        result = response.choices[0].text.strip()
        
        return result
    def strip_punctuation(self, input_string, punctuation=string.punctuation):
        """Removes all occurrences of the specified punctuation from the input string."""
        return input_string.translate(str.maketrans('', '', punctuation))
    def category_prompt(self,content):
        prompt = f'''
        Instructions: Return the category that the content (delimited by triple backticks) fits into best.


        Content: ```{content}```
        
        
        
        Categories: 'Cocktail recipes and tutorials','Wine and beer recommendations','Party and event photos',
                      'Alcohol-related memes and humor','Reviews of bars and restaurants','Alcohol-related news and trends',
                      'Personal stories and experiences','Celebrity endorsements and sponsorships',
                    'Health and wellness tips','Advocacy and activism', 'Not related to Alcohol Beverages'.
        Category:
        '''
        return prompt
    def evaluate_questions(self, content):
        responses = {}
        model = "text-davinci-002"
        #model = "text-ada-001"

        for feature_name, question in self.questions.items():
            prompt =  f'''
                    You're an evaluator that only returns the answer to questions about content. 
                    Evaluate the following social media content to answer this question "{question}".
                    Return only the answer.
                    Content: {content}

                    '''
            response = openai.Completion.create(
                    model=model,
                    prompt = prompt,
                    temperature = 0.1,
                    max_tokens = 100,
                    frequency_penalty=0,
                    presence_penalty=0,
                    top_p = 1,
                    n=1)
            
            result = response.choices[0].text.strip()
            responses[feature_name] = result
        return responses
    
        

if __name__ == '__main__':
    #content = 'Hey San Francisco - how do you "Spritz"? Do it the @lillet way. Cheers 🥂#drinkresponsibly #lillet #thelilletway #dessertwine #SF #sanfrancisco #outdoor#ooh #advertising #wallscapes #marketing \n\nhttps://preview.redd.it/lu81ozglb3e31.jpg?width=3061&amp;format=pjpg&amp;auto=webp&amp;v=enabled&amp;s=6e83c5670262eb164940d38ab6a5f188aa38ea92'
    #content = 'Barack Obama and Anthony Bourdain enjoying a $6 dollar meal and a beer in Hanoi, Vietnam. The table they ate at was later enclosed in glass and put in display.'
    #content = 'TIL that alcohol consumption in the U.S. was almost 300% higher in the 1800s, and that whiskey at the time was cheaper than beer, coffee or milk.'
    #content = 'Is there any way to fix my series X’s case? Rubbing alcohol went on it and took off some of the paint making it look oily'
    #content = '''25 Employees Walk Out After Pennsylvania Restaurant Owner Names Drinks ‘The Negro’ And ‘The Caucasian’'''
    #content = '''by the Ohio Governor and EPA Chief to prove the tap water was safe by pretending to drink it.'''
    content = '''Wife and I went to Mexico for our honeymoon. I had the idea of buying a bottle of tequila and drinking a shot every anniversary. It broke on the way home.'''
    #content = '''Why does alcohol still taste like shit? I'm 21, the drinking age is 18 where I am. All my peers already drink. I find beer and wine disgusting and can barely tolerate the taste of vodka and lady drinks (as a dude.) When does it get better?'''

    evaluator = Content_Evaluator()

    test_prompt = f'''
    You will be provided the Title and text of a Reddit post deliminated below with triple backticks.
    Evaluate the post and return a Python Dictionary containing the keys 'relevant_to_alcohol_consumption','sentiment',
    'named_entities','category_of_post','consumer_packaged_goods_mentioned','three_word_summary'. The options for each key are 'relevant_to_alcohol_consumption': True/False, 'sentiment': ['positive,'neutral','negative'],
    'named_entites': list and choose a 'category_of_post' from this list ['Cocktail recipes and tutorials','Wine and beer recommendations','Party and event photos',
                      'Alcohol-related memes and humor','Reviews of bars and restaurants','Alcohol-related news and trends',
                      'Personal stories and experiences','Celebrity endorsements and sponsorships',
                    'Health and wellness tips','Advocacy and activism', 'Other']

    
    
    Content: ```{content}```


    Output (Python Dictionary):'''

    test_response = evaluator.get_completion(test_prompt)

    prompt = evaluator.relevance_prompt(content)
    relevance = evaluator.evaluate_prompt(prompt, model="text-curie-001")
    
    #sentiment = evaluator.vader_sentiment(content)

    #named_entity = evaluator.named_entity(content)

    prompt = evaluator.category_prompt(content)
    #category = evaluator.evaluate_prompt(prompt,model='text-davinci-002')
    #category = evaluator.strip_punctuation(category, "'',.")
    print(f'''

    content: {content}
    lenght_of_prompt: {len(prompt)}
    relevance: {relevance}

    length_of_prompt: {len(test_prompt)}
    test_response: {test_response}
    ''')
    
    # 
    # sentiment (VADER): {sentiment}
    # named_entity: {named_entity}
    # category: {category}\n''')
