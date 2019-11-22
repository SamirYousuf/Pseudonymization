# Modules
import nltk
import re, random
import pandas as pd
import numpy as np
from Levenshtein import distance
import argparse

# Randomly change the school information with respect to the place mentioned if any
def university(data, list_universitys):
  temp_list = []
  for x in data:
    for y in list_universitys:
      if y in x:
        x = x.replace(y, random.choice(list_universitys))
    temp_list.append(x)
  return(temp_list)

# Read a csv file that contails the data for all the professional jobs in sweden (total of 8695)
def profession(data):
  list_job_title = pd.read_csv('data/Prof_dataset.csv')
  list_title = list_job_title['Yrkesbenämning'].tolist()
  list_title = list_title[:-7]
  temp_list1 = []
  temp_list = []
  for x in list_title:
    if ',' in x:
      y = x.split(',')
      temp_list1.append(y[0].lower())
    else:
      temp_list1.append(x.lower())
  for x in data:
    for y in temp_list1:
      if y in x:
        x = x.replace(y, random.choice(temp_list1))
    temp_list.append(x)
  return(' '.join(temp_list))


if __name__ == '__main__':
  
  list_universities = ['Stockholms Universitet','Göteborgs universitet','Uppsala universitet','Lunds universitet','Linnéuniversitetet','Umeå universitet','Linköpings universitet','Malmö högskola','Kungliga Tekniska högskolan','Högskolan i Gävle','Mittuniversitetet','Högskolan Dalarna','Örebro universitet','Karlstads universitet','Mälardalens högskola','Luleå tekniska universitet','Högskolan Kristianstad','Högskolan i Jönköping','Chalmers tekniska högskola','Södertörns högskola','Karolinska institutet','Högskolan i Borås','Högskolan i Halmstad','Högskolan Väst','Högskolan i Skövde','Blekinge tekniska högskola','Sveriges lantbruksuniversitet','Handelshögskolan i Stockholm','Ersta Sköndal Bräcke högskola','Kungliga Musikhögskolan i Stockholm','Gymnastik- och idrottshögskolan','Sophiahemmet Högskola','Röda Korsets högskola','Konstfack','Stockholms dramatiska högskola','Försvarshögskolan','Teologiska Högskolan','Örebro teologiska högskola','Kungliga Konsthögskolan','Johannelunds teologiska högskola','Stockholms Musikpedagogiska Institutet','Newmaninstitutet','Beckmans designhögskola','Ericastiftelsen','Svenska institutet för kognitiv psykoterapi','Högskolan Evidens','Gammelkroppa skogsskola','Skandinaviens akademi för psykoterapiutveckling']
  
  parser = argparse.ArgumentParser(description='Program takes an input text file and output text file')
  
  parser.add_argument('--input', '-i', type=str, required=True)
  parser.add_argument('--output', '-o', type=str, required=True)
  args = parser.parse_args()
  
  # Read a given file and save the tokenized sentence data in a list
  with open(args.input) as file:
    data = file.read()

  data = re.sub(r'\n\n', ' $$$$ . ', data)
  data = nltk.sent_tokenize(data)
  data = university(data, list_universities)
  output_file = profession(data)
  output_file = output_file.replace(' $$$$ . ', '\n\n')

  if args.output:
    file = open(args.output, 'w')
    file.write(output_file)
