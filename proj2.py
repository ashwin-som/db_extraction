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
import time
import ast
import re
import requests
from collections import defaultdict
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
    #print('sentence entity pairs:',sentence_entity_pairs)
    #print("sentence entity pairs: ", sentence_entity_pairs)
    for ep in sentence_entity_pairs:
        type1 = ep[1][1] #so can be something like person, location ,etc. 
        type2 = ep[2][1]
        if r==1 or r==2:
            if type1=='PERSON' and type2=='ORGANIZATION':
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  # e1=Subject, e2=Object
            elif type2=='PERSON' and type1=='ORGANIZATION':
                candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})  # e1=Object, e2=Subject
        elif r==3:
            if type1=='PERSON' and type2 in set(['LOCATION', 'CITY', 'STATE_OR_PROVINCE','COUNTRY']):
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
            elif type2=='PERSON' and type1 in set(['LOCATION', 'CITY', 'STATE_OR_PROVINCE','COUNTRY']):
                candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
        elif r==4:
            if type2=='PERSON' and type1=='ORGANIZATION':
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  # e1=Subject, e2=Object
            elif type1=='PERSON' and type2=='ORGANIZATION':
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

    out = ''
    for candidate in response.candidates:
        out = [part.text for part in candidate.content.parts]

    return ''.join(out)
def gemini_api(sent,r,ex_sent,ex_output,meaning):
    if r==4:
        prompt_text ="""Given the TARGET sentence below, extract all instances of the following relationship type you can find in the sentence. Do not provide any explanation except the output. If you are not able to parse any relationships, then return this string: 'NOTHING'. The SUBJECT should only be proper nouns.

Meaning of this Relationship Type: The OBJECT works for or contributes to the SUBJECT."

Example Sentence: Jensen Huang is the head of Nvidia
Example Relationship Type: Top_Member_Employees
Example Output: [('Nvidia','Jensen Huang')]

Output Format:
[('SUBJECT', 'OBJECT'),...]

TARGET: {0}""".format(sent)
    else:
        prompt_text = """Given the TARGET sentence below, extract thenrelationship type you can find in the sentence. Do not provide any explanation except the output. If you are not able to parse any relationships, then return this string: 'NOTHING'. 


Relationship: {0}
Output Format:
[('SUBJECT', 'OBJECT'),...]

TARGET: {1}""".format(r,sent)
    #print("THIS IS THE PROMPT:", prompt_text)

    # Feel free to modify the parameters below.
    # Documentation: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
    model_name = 'gemini-pro'
    max_tokens = 100
    temperature = 0.2
    top_p = 1
    top_k = 32

    response_text = get_gemini_completion(prompt_text, model_name, max_tokens, temperature, top_p, top_k)
    #print('THE GEMINI MODEL HAS RETURNED: ',response_text)
    return response_text

def process_tuples(sent):
    #print('TRYING TO PROCESS THIS: ',sent)
    try:
        tuples = ast.literal_eval(sent)
        return tuples
    except:
        return 'NOTHING'


def main():
    #print("hello world")
    #hello this is a commentgoo
    #/home/gkaraman/run <google api key> <google engine id> <precision> <query>
    #key = "AIzaSyC0vz_nYIczwBwNupqMrNhmBm4dQbX5Pbw"
    #id = "7260228cc892a415a"
    relations = {
    1:'school attended',
    2:'Work_For',
    3:'live in',
    4:4,
    }
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
    #X_extracted_tuples = {}
    X_extracted_tuples = defaultdict()
    #stores the used qs for web api call
    used_qs = set()
    used_qs.add(q)
    count = 0
    entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    nlp = spacy.load("en_core_web_lg")
    spanbert = SpanBERT("./pretrained_spanbert") 
    output_tuples = set()
    #print("gemini or spam is", gem_span)
    numIterations = 0
    prevQuery = ''
    actualRelations = {
        1:'Schools_Attended',
        2:'Work_For',
        3:'Live_In',
        4:'Top_Member_Employees',
    }
    print('____')
    print('Parameters:')
    print('Client key	= ',google_api)
    print('Engine key	= ',google_engine)
    print('Gemini key	= ',google_gemini_id)
    print('Method	= ',gem_span)
    print('Relation	= ',actualRelations[r])
    print('Threshold	= ',t)
    print('Query		= ',q)
    print('# of Tuples	= ',k)
    while (count<k and gem_span == '-gemini') or  (len(X_extracted_tuples) < k and gem_span == '-spanbert'):
        print("=========== Iteration: {0} - Query: {1} ===========".format(numIterations,q))
        numIterations+=1
        if prevQuery==q:
            break
        else:
            prevQuery=q
        links = scrape_web(q,google_api, google_engine)
        link_count = 0 
        link_total = len(links)
        #extractions per webiste -> get difference 
        
        for link in links:
            link_count += 1 
            print("URL (",link_count,"/",link_total,"): ", link)
            if link in explored_urls:
                print("\t\tlink has not been used already!")
            else:
                #print("adding in a new link")
                explored_urls.add(link)
                #now extract webpage, as long as no timeoute 
                #with open(link) as fp:
                print("\tFetching text from url ...")
                content = requests.get(link)
                if content.status_code != 200: #make sure not nothing 
                    print("Unable to fetch URL. Continuing.")
                    continue
                #if content
                html_stuff = content.text
                html_stuff = re.sub(r'\s+', ' ',html_stuff)
                html_stuff = re.sub(r'[^\x00-\x7F]+', ' ',html_stuff)
                soup = BeautifulSoup(html_stuff, 'html.parser')
                #use beautiful soup to get text (only first 10,000 chars)
                text = soup.get_text()[0:10000]
                print("\tAnnotating the webpage using spacy...")
                doc = nlp(text)
                sent_count = 0
                sent_total = sum(1 for _ in doc.sents)
                print('\tExtracted {0} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...'.format(sent_total))
                old_val_span = len(X_extracted_tuples)
                old_val_gem = len(output_tuples)
                #new_tup_count = 0
                print()
                for sent in doc.sents:
                    #split the text into sentences and extract named entities -> use spaCy
                    sent_count += 1 
                    if sent_count%5 ==0:
                        print("\tProcessing sentence ",sent_count," of ", sent_total, " total number of sentences")
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
                            goal_relation = "per:cities_of_residence"
                        elif r==4:#top employee 
                            entities_of_interest = entities_of_interest_top_employee
                            subjects.add("ORGANIZATION")
                            objects.add("PERSON")
                            goal_relation = "org:top_members/employees"
                        else:
                            print("\t\tinvalid input")

                        #nlp = spacy.load("en_core_web_lg")  
                        #doc = nlp(text)  
                         
                        sentence = sent
                        #print("sentence is: ",sentence)
                        new_tuples = extract_relations(sentence,spanbert,entities_of_interest,subjects, objects, t)
                        #print("printing extracted relations", new_tuples)
                        #now with new tuples add them to the dictionary
                        #currently, new tuples are some sort of default dictioany 

                        #for now may just gonna print new_tuples to see the format 
                        #format is res[(subj, relation, obj)] = confidence -> dictionary of tuple 
                        #print("new tuples ",new_tuples)
                        for tag,confidence in new_tuples.items(): #want it to be in format of tuple -> ((entity1,entity2),confidence)
                            subject, relation, obj = tag[0],tag[1],tag[2]
                            if relation == goal_relation and confidence > t: #can add 
                                #"\tConfidence: {0}      \t | Subject: {1}      \t | Object: {2}".format(confidence,tag[0], tag[1])
                                #print("=== Extracted Relation ===")
                                #Input tokens: ['Google', 'CEO', 'Sundar', 'Pichai', ',']
                                #print("Output Confidence: {} ;  Subject: {} ; Object: {} ;".format(confidence,subject,obj))
                                #print("Adding to set of extracted relations")
                                #print("==========")
                                label = (subject,obj)
                                reversed_label = (obj, subject)
                                if label in X_extracted_tuples: #if label in 
                                    X_extracted_tuples[label] = max(X_extracted_tuples[label],confidence)
                                elif reversed_label in X_extracted_tuples:#if reverse label in 
                                    X_extracted_tuples[reversed_label] = max(X_extracted_tuples[reversed_label],confidence) 
                                else:#add in 
                                    X_extracted_tuples[label] = confidence 
                                count+=1
                    #now all new tuples added if confidence threshold met -> repeats handles by storing that with the highest confidence 
                    #if spanbert bert do: 

                    #if gemini do: 
                    elif gem_span == '-gemini':
                        #print("gemini")
                        candidate_pairs = gemini_get_candidate_pairs(sent,entities_of_interest,r) 
                        #print("cnadidate pairs: ",candidate_pairs)
                        if len(candidate_pairs)==0:
                            #print()
                            continue
                        #print(candidate_pairs)
                        #print('Processing Sentence: ',sent)
                        #print('calling gemini')
                        ex_sent,ex_output,meaning = '','',''
                        if r==1:
                            ex_sent = """Sergey Brin attended Stanford University"""
                            ex_output = """[('Sergey Brin','Princeton University')]"""
                            meaning = """The SUBJECT attends the OBJECT."""
                        elif r==2:
                            ex_sent = """Alec Radford works for OpenAI"""
                            ex_output = """[('Alec Radford','OpenAI')]"""
                            meaning = """The SUBJECT works at or contributes something to the OBJECT."""
                        elif r==3:
                            ex_sent = """Mariah Carey lives in New York City"""
                            ex_output = """[('Mariah Carey','New York City')]"""
                            meaning = """The SUBJECT has lived or resided in OBJECT."""
                        elif r==4:
                            ex_sent = """Jensen Huang is the head of Nvidia"""
                            ex_output = """[('Nvidia','Jensen Huang')]"""
                            meaning = """The OBJECT works for the SUBJECT."""
                        try:
                            target_tuples_sent = gemini_api(sent,relations[r],ex_sent,ex_output,meaning)
                        except:
                            print('SOMETHING WAS MESSED UP WHILE CALLING THE API')
                            continue
                        #time.sleep(1)
                        if target_tuples_sent=='NOTHING':
                            continue
                        result_tuples = process_tuples(target_tuples_sent)
                        if result_tuples=='NOTHING':
                            continue
                        print()
                        for tup in result_tuples:
                            print('\t\t=== Extracted Relation ===')
                            print('\t\tProcessing Sentence: ',sent)
                            print('\t\tSubject: {0} ; Object: {1} ;'.format(tup[0],tup[1]))
                            if tup not in output_tuples:
                                q = tup[0]+' '+tup[1]
                                output_tuples.add(tup)
                                count+=1
                                print('\t\tAdding to set of extracted relations')
                            else:
                                print('\t\tDuplicate. Ignoring this.')
                            print()
                        #print(target_tuples_sent)
                    else:
                        print("\t\twrong type input")
                #add in tuple increase 
                if gem_span == '-spanbert':
                    new_tuples = max(len(X_extracted_tuples) - old_val_span ,0)
                else: #is gemini
                    new_tuples = max(len(output_tuples) - old_val_gem,0)
                print("\tNew relations extracted from this website: ",new_tuples)
                print()
        #this should be end of links 
        
        if gem_span == '-spanbert':
            #sort all the element in dictionary 
            X_extracted_tuples = dict(sorted(X_extracted_tuples.items(), key=lambda item: item[1],reverse=True))
            #print("printing extracted tuples",X_extracted_tuples)
            #get the next q 
            if len(X_extracted_tuples) < k:
                found = False
                for tag,confidence in X_extracted_tuples.items():
                    if tag not in used_qs:
                        q = tag[0]+' '+tag[1]
                        #print("printing q, ", q)
                        found = True
                        used_qs.add(tag)
                        break
                if not found: #exit because no more tuples to be found 
                    print("\t\t there are no more seed tuples to be generated. Exiting program...")
                    break
                        

    #print(X_extracted_tuples)
    #for ex, pred in list(zip(candidate_pairs, relation_preds)):
            #print("\tSubject: {}\tObject: {}\tRelation: {}\tConfidence: {:.2f}".format(ex["subj"][0], ex["obj"][0], pred[0], pred[1]))
    if gem_span=='-gemini':
        print('================== ALL RELATIONS for {0} ( {1} ) ================='.format(relations[r],count))
        #print('\t\tAll relations for: ',relations[r],'(',count,')')
        for tup in output_tuples:
            print('Subject: {0} 		| Object: {1}'.format(tup[0],tup[1]))
    else:
        print("\t\tAll relations for: ", goal_relation)
        print("\t\t",len(X_extracted_tuples), "relations generated.")
        count = 0
        for tag,confidence in X_extracted_tuples.items():
            count += 1
            #if count > k: #so that we only print out k tuples
                #break
            #print("\tSubject: {}\tObject: {}\tRelation: {}\tConfidence: {:.2f}".format(tag[0], tag[1], confidence))
            print("\tConfidence: {0}      \t | Subject: {1}      \t | Object: {2}".format(confidence,tag[0], tag[1]))
            #print("subject is ",tag[0], "and object is ", tag[1], "with a confidence of ", confidence) 

    
    print("Total # of iterations = ",numIterations)

    






        
        


if __name__ == "__main__":
    main()
