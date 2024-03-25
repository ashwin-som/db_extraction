import numpy as np 
from googleapiclient.discovery import build
#from bs4 import BeautifulSoup
from bs4 import BeautifulSoup
#from sklearn.feature_extraction.text import TfidfVectorizer 
#import heapq
import sys
import spacy
#import create_entity_pairs
from spanbert import SpanBERT
from spacy_help_functions import extract_relations,get_entities,create_entity_pairs
from example_relations import get_all_entities
import os
import google.generativeai as genai
import ast
import requests
# Apply Gemini API Key
GEMINI_API_KEY = 'AIzaSyCSF9KInhX1u1vaLSrv-MCPHOCI0aCqVzQ'  # Substitute your own key here
genai.configure(api_key=GEMINI_API_KEY)

#nlp = spacy.load("en_core_web_lg")
#entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]

'''relations = {
    1:'Schools_Attended',
    2:'Work_For',
    3:'Live_In',
    4:'Top_Member_Employees',
}'''
'''def extract_tuples(input_text,entities_of_interest,spanbert):
    
    #using spacy -> convert text to possible tuples
    # can identify numeric entities->including companies, locations, organizations and products
    
    #first, turn text into sentences -> # Apply spacy model to raw text (to split to sentences, tokenize, extract entities etc.)
    nlp = spacy.load("en_core_web_lg")  
    sentences = nlp(input_text).sents
    #now, I think depending on if spanbert or dictionary, we build them 
    new_tuples = set()
    for sentence in sentences: 
        #sentence_tuples = create_entity_pairs(sentence, existing_entities, window_size=40)
        relations = extract_relations(sentences, spanbert, entities_of_interest)
        print("Relations: {}".format(dict(relations)))
        for tup in relations:
            if tup not in new_tuples:
                new_tuples.add(tup)  
    return list(new_tuples)'''


def scrape_web(query, key, id):
    service = build(
        "customsearch", "v1", developerKey=key
    )

    res = (
        service.cse()
        .list(
            q=query,
            cx=id,
        )
        .execute()
    )
    
    links = []
    for result in res['items']:
        links.append(result.get('link'))
    return links


def gemini_get_candidate_pairs(sent,entities_of_interest,r):
    ents = get_entities(sent, entities_of_interest)
    candidate_pairs = []
    sentence_entity_pairs = create_entity_pairs(sent, entities_of_interest)
    for ep in sentence_entity_pairs:
        if r==1 or r==2:
            if ep[1]=='PERSON' and ep[2]=='ORGANIZATION':
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  # e1=Subject, e2=Object
            elif ep[2]=='PERSON' and ep[1]=='ORGANIZATION':
                candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})  # e1=Object, e2=Subject
        elif r==3:
            if ep[1]=='PERSON' and ep[2] in set(['LOCATION', 'CITY', 'STATE_OR_PROVINCE','COUNTRY']):
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
            elif ep[2]=='PERSON' and ep[1] in set(['LOCATION', 'CITY', 'STATE_OR_PROVINCE','COUNTRY']):
                candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
        elif r==4:
            if ep[2]=='PERSON' and ep[1]=='ORGANIZATION':
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  # e1=Subject, e2=Object
            elif ep[1]=='PERSON' and ep[2]=='ORGANIZATION':
                candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})  # e1=Object, e2=Subject
    
    return candidate_pairs

def get_gemini_completion(prompt, model_name, max_tokens, temperature, top_p, top_k):

    # Initialize a generative model
    model = genai.GenerativeModel(model_name)

    # Configure the model with your desired parameters
    generation_config=genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

    # Generate a response
    response = model.generate_content(prompt, generation_config=generation_config)

    return response.text
def gemini_api(sent,r):
    prompt_text = """Given a sentence, extract all instances of the following relationship type you can find in the sentence. Do not provide any explanation except the output.

Relationship Type: {0}

Output Format:
[(RELATIONSHIP TYPE, SUBJECT, OBJECT),...]

Sentence: {1}""".format(r,sent)

    # Feel free to modify the parameters below.
    # Documentation: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
    model_name = 'gemini-pro'
    max_tokens = 100
    temperature = 0.2
    top_p = 1
    top_k = 32

    response_text = get_gemini_completion(prompt_text, model_name, max_tokens, temperature, top_p, top_k)
    print(response_text)

def process_tuples(sent):
    tuples = ast.literal_eval(sent)
    return tuples


def main():
    #/home/gkaraman/run <google api key> <google engine id> <precision> <query>
    #key = "AIzaSyC0vz_nYIczwBwNupqMrNhmBm4dQbX5Pbw"
    #id = "7260228cc892a415a"
    i = 1 
    google_api = sys.argv[0+i]
    google_engine = sys.argv[1+i]
    google_gemini_id = sys.argv[2+i]
    gem_span = sys.argv[3+i] #indiicates whether we are using span or bert 
    #for r: 1 indicates Schools_Attended, 2 indicates Work_For, 3 indicates Live_In, and 4 indicates Top_Member_Employees
    r = int(sys.argv[4+i]) 
    #t is a real num from 0 to 1 of extraction confidence threshold -> only used if BERT 
    t = float(sys.argv[5+i]) 
    #random seed query 
    q= sys.argv[6+i]
    #k is numper of tuples we want to output 
    k = int(sys.argv[7+i]) 

    #keeps track of urls already looked at 
    explored_urls = set()
    #tuples to be generated starts empty -> use a dictionary to hold onto highest value 
    X_extracted_tuples = {}
    count = 0
    entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    nlp = spacy.load("en_core_web_lg")
    spanbert = SpanBERT("./pretrained_spanbert") 
    print("gemini or spam is", gem_span)
    while len(X_extracted_tuples)<k and count<k:
        count+=1
        links = scrape_web(q,google_api, google_engine)
        #just get links that we have not looked at yet
        #desired_links = []
        for link in links:
            if link not in explored_urls:
                explored_urls.add(link)
                #now extract webpage, as long as no timeoute 
                #with open(link) as fp:
                content = requests.get(link)
                if content.status_code != 200: #check to make sure html 
                    print("not status 200")
                html_stuff = content.text
                soup = BeautifulSoup(html_stuff, 'html.parser')
                #use beautiful soup to get text (only first 10,000 chars)
                text = soup.get_text()[0:10000]
                #nlp = spacy.load("en_core_web_lg")
                doc = nlp(text)
                for sent in doc.sents:
                    #split the text into sentences and extract named entities -> use spaCy
                    if gem_span == '-spanbert':
                        #spanbert = SpanBERT("./pretrained_spanbert") 
                        entities_of_interest_schools = ["ORGANIZATION", "PERSON"]
                        entities_of_interest_employee = ["ORGANIZATION", "PERSON"]
                        entities_of_interest_residence = ["PERSON", "LOCATION", "CITY","STATE_OR_PROVINCE", "COUNTRY"]
                        entities_of_interest_top_employee = ["PERSON","ORGANIZATION"]
                        subjects = set()
                        objects = set()
                        if r==1:#schools 
                            entities_of_interest = entities_of_interest_schools
                            subjects.add("PERSON")
                            objects.add("ORGANIZATION")
                            goal_relation = "per:schools_attended"
                        elif r==2:#works 
                            entities_of_interest = entities_of_interest_employee
                            subjects.add("PERSON")
                            objects.add("ORGANIZATION")
                            goal_relation = "per:employee_of"
                        elif r==3:#lives 
                            entities_of_interest = entities_of_interest_residence
                            subjects.add("PERSON")
                            objects.add("LOCATION")
                            objects.add("CITY")
                            objects.add("STATE_OR_PROVINCE")
                            objects.add("COUNTRY")
                            goal_relation = "org:top_members/employees"
                        elif r==4:#top employee 
                            entities_of_interest = entities_of_interest_top_employee
                            subjects.add("ORGANIZATION")
                            objects.add("PERSON")
                            goal_relation = "per:schools_attended"
                        else:
                            print("invalid input")

                        #nlp = spacy.load("en_core_web_lg")  
                        #doc = nlp(text)  
                         
                        sentence = sent
                        new_tuples = extract_relations(sentence,spanbert,entities_of_interest,subjects, objects, t)
                        #now with new tuples add them to the dictionary
                        #currently, new tuples are some sort of default dictioany 

                        #for now may just gonna print new_tuples to see the format 
                        #format is res[(subj, relation, obj)] = confidence -> dictionary of tuple 
                        print(new_tuples)
                        for tag,confidence in new_tuples.itemize(): #want it to be in format of tuple -> ((entity1,entity2),confidence)
                            subject, relation, obj = tag[0],tag[1],tag[2]
                            if relation == goal_relation and confidence > t: #can add 
                                label = (subject,obj)
                                reversed_label = (obj, subject)
                                if label in X_extracted_tuples: #if label in 
                                    X_extracted_tuples[label] = max(X_extracted_tuples[label],confidence)
                                elif reversed_label in X_extracted_tuples:#if reverse label in 
                                    X_extracted_tuples[reversed_label] = max(X_extracted_tuples[reversed_label],confidence) 
                                else:#add in 
                                    X_extracted_tuples[label] = confidence 
                    #now all new tuples added if confidence threshold met -> repeats handles by storing that with the highest confidence 
                    #if spanbert bert do: 

                    #if gemini do: 
                    elif gem_span == '-gemini':
                        count+=1
                        print("gemini")
                        candidate_pairs = gemini_get_candidate_pairs(sent,entities_of_interest,r) 
                        if len(candidate_pairs)==0:
                            continue
                        target_tuples_sent = gemini_api(sent,relations[r])
                        result_tuples = process_tuples(target_tuples_sent)
                        print(result_tuples)
                    else:
                        print("wrong type input")

    print(X_extracted_tuples)






        
        


if __name__ == "__main__":
    main()
