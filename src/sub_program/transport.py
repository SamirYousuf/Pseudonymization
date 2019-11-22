# Modules
import nltk
import re, random
import pandas as pd
import numpy as np
from Levenshtein import distance
import argparse

# Transport function pseudonymise private vehicles as well as the other mean of transportation
def transport(data, list_transports):
  temp_list = []
  for x in data:
    y = x.split(' ')
    for i, j in enumerate(y):
      for k in list_transports:
        if j == k:
          y[i] = random.choice(list_transports)
    temp_list.append(' '.join(y))
  return(temp_list)

# Changes the stations according to the vehicle mentioned for transportation
def stations(data, list_stations):
  temp_list = []
  for x in data:
    y = x.split(' ')
    for i, j in enumerate(y):
      for k in list_stations:
        if j == k:
          y[i] = random.choice(list_stations)
    temp_list.append(' '.join(y))
  data = temp_list
  temp_list = []
  for x in data:
    y = x.split(' ')
    for i, j in enumerate(y):
      for k in list_stations_en:
        if j == k:
          y[i] = random.choice(list_stations_en)
    temp_list.append(' '.join(y))
  return(' '.join(temp_list))

if __name__ == '__main__':
  
  list_transports = ['buss', 'bil', 'personbil', 'motorcykel', 'lastbil', 'cykel', 'moped', 'helikopter', 'traktor']
  
  list_stations = ['busshållplats', 'spårvagnhållplats', 'tågstation', 'flygplats', 'tunnelbanestation']
  list_stations_en = ['busshållplatsen', 'spårvagnhållplatsen', 'tågstationen', 'flygplatsen', 'tunnelbanestationen']

  parser = argparse.ArgumentParser(description='Program takes an input text file and output text file')
  
  parser.add_argument('--input', '-i', type=str, required=True)
  parser.add_argument('--output', '-o', type=str, required=True)
  args = parser.parse_args()
  
  # Read a given file and save the tokenized sentence data in a list
  with open(args.input) as file:
    data = file.read()

  data = re.sub(r'\n\n', ' $$$$ . ', data)
  data = nltk.sent_tokenize(data)
  data = transport(data, list_transports)
  output_file = stations(data, list_stations, list_stations_en)
  output_file = output_file.replace(' $$$$ . ', '\n\n')

  if args.output:
    file = open(args.output, 'w')
    file.write(output_file)
