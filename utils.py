import os
import re 
import openai 
 
class config:
    YOUR_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = config.YOUR_API_KEY 
 
chat_starter = f'''
The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.
 
 
'''
 
def simple_clean(_text):
    _text = str(_text).strip().replace('\n\n', ' ').replace('\\n', ' ').replace('\n', ' ')
    _text = str(_text).strip().replace('\t\t', ' ').replace('\\t', ' ').replace('\t', ' ')
    _text = re.sub('http\S+','', _text)  
    _text = re.sub('www\S+','', _text)  
    _text = re.sub('@[A-Za-z0-9]+','', _text)
    
    _text = re.sub('[#$%&@]', '', _text)  
    _text = re.sub(r"'s\b", " is", _text)      
    _text = re.sub(r"'ll\b", " will", _text)   
    _text = re.sub('[^a-zA-Z0-9.?,:;!]', ' ', _text) 
    _text =  _text.replace(' n ', '')
    
    _text = re.sub(',+  ', ' ', _text)
    _text = re.sub(' + ', ' ', _text)
    return _text.strip()


def generate_ai_answer(prompt):
    response = openai.Completion.create(
                        model="text-davinci-002",
                        prompt=prompt,
                        temperature=0.9,
                        max_tokens=150,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.6,
                        stop=[" Human:", " AI:"]
                )
    ans = response.choices[0].text
    return ans


