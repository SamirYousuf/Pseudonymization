# Modules
import nltk
import re, random
import pandas as pd
import numpy as np
import argparse

# Names in the data are psuedonymised
def name_male_female(data):
  temp_data = data
  list_name = pd.read_csv('/data/last_name.csv')
  last_name = list_name['Last Name'].tolist()
  last_name = last_name[:-1]
  temp_name_list = []
  for x in data:
    if len(re.findall(r' (kallas för|heter|namn är) (\w+) ', x)) > 0:
      y = re.findall(r' (kallas för|heter|namn är) (\w+) ', x)
      y1 = list(y[0])
      temp_name_list.append(y1[1])
  data = ' '.join(data)
  for x in set(temp_name_list):
    if x in last_name:
      if x in data:
        data = re.sub(x, random.choice(last_name), data)
  return ' '.join(nltk.sent_tokenize(data))

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='Program takes an input text file and output text file')
  parser.add_argument('--input', '-i', type=str, required=True)
  parser.add_argument('--output', '-o', type=str, required=True)
  args = parser.parse_args()
  
  # Read a given file and save the tokenized sentence data in a list
  with open(args.input) as file:
    data = file.read()

  data = re.sub(r'\n\n', ' $$$$ . ', data)
  data = nltk.sent_tokenize(data)
  output_file = name_male_female(data)
  output_file = output_file.replace(' $$$$ . ', '\n\n')

  if args.output:
    file = open(args.output, 'w')
    file.write(output_file)
