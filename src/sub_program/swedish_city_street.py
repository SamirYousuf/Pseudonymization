# Modules
import nltk
import re, random
import pandas as pd
import numpy as np
import argparse

def swedish_cities_streets(data):
  list_swedish_cities_streets = pd.read_csv('/data/city_street.csv')
  list_swedish_cities_streets = list_swedish_cities_streets.drop(columns=['Unnamed: 0'])
  list_swedish_cities_streets = list_swedish_cities_streets[list_swedish_cities_streets.Streets != 'Streets']
  list_swedish_cities_streets = list_swedish_cities_streets[list_swedish_cities_streets.Cities != 'Cities']
  street_list1 = list_swedish_cities_streets['Streets'].tolist()
  city_list1 = list_swedish_cities_streets['Cities'].tolist()
  street_list = []
  city_list = []
  for i in range(len(street_list1)):
    if type(street_list1[i]) is not str:
      pass
    else:
      city_list.append(city_list1[i])
      street_list.append(street_list1[i])
  city_in_data = []
  street_in_data = []
  for i in data:
    for j in set(city_list):
      if str(j) in i:
        city_in_data.append(j)
    for k in set(street_list):
      if k in i.split(' '):
        street_in_data.append(k)
  text = ' '.join(data)
  if len(city_in_data) > 0:
    if len(street_in_data) > 0:
      for i in street_in_data:
        index = street_list.index(i)
        if city_list[index] in city_in_data:
          x = random.choice(city_list)
          text = text.replace(city_list[index], x)
          inde = city_list.index(x)
          text = text.replace(street_list[index], street_list[inde])
        else:
          text = text.replace(i, random.choice(street_list))
      for i in city_in_data:
        index = city_list.index(i)
        if street_list[index] not in street_in_data:
          x = random.choice(street_list)
          text = text.replace(street_list[index], x)
          inde = street_list.index(x)
          text = text.replace(city_list[index], city_list[inde])
        else:
          text = text.replace(i, random.choice(city_list))
  elif len(street_in_data) > 0:
    for i in street_in_data:
      text = text.replace(i, random.choice(street_list))

  data = nltk.sent_tokenize(text)
  temp_list = []
  for x in data:
    if re.search(r' \d{3} \d{2} ', x):
      x = re.sub(r'(\d{3} \d{2})', '000 00', x)
      temp_list.append(x)
    else:
      temp_list.append(x)
  return(' '.join(temp_list))


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
  output_file = swedish_cities_streets(data)
  output_file = output_file.replace(' $$$$ . ', '\n\n')

  if args.output:
    file = open(args.output, 'w')
    file.write(output_file)
