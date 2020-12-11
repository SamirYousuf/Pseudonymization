# Modules
import nltk
import argparse, json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET


def getData(sentence):
    
    #Get data from url
    
    url = 'https://ws.spraakbanken.gu.se/ws/sparv/v2/'
    values = {'text':sentence}

    d = urllib.parse.urlencode(values)
    d = d.encode('utf-8')
    req = urllib.request.Request(url, d)
    resp = urllib.request.urlopen(req)
    respData = resp.read()

    return respData
    
    
def parseXML(data):
    
    #Parse XML

    words_with_pos = []
    root = ET.fromstring(data)
        
    for w in root.iter('w'):
        attributes = w.attrib
        pos = attributes['pos']
        words_with_pos.append(w.text+'//'+pos)
        
    new_sentence = ' '.join(words_with_pos)
    
    return new_sentence
        
        
def annotate(data):

    #Main function for annotating the data with POS tags from Sparv
            
    new_sentences = []
    
    data = nltk.sent_tokenize(data)
    
    for line in data:
        annotated = getData(line)
        parsed_sentence = parseXML(annotated)
        new_sentences.append(parsed_sentence)
                
    return new_sentences #returning list so that you don't have to tokenize it again in the next step
    
if __name__ == '__main__':
    
    output_data = annotate(data)
    
