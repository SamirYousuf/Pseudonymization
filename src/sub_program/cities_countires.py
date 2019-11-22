# Modules
import nltk
import re, random
import pandas as pd
import numpy as np
import argparse

# Pseudonymise the countries and cities in the text
# Countries and cities are randomised using conditions given below
# "I live in Paris, France." is replaced to "I live in London, England."
# "I live in Delhi." is replaced to other random city only if the country "India" is not mentioned in the data
# If it is mentioned then it will replace both the entites
# otherwise only the city
def country_city(data):
  country_in_data = []
  cities_in_data = []
  list_data = pd.read_csv('/data/city_country.csv')
  list_country = list_data['Countries'].tolist()
  list_city = list_data['Cities'].tolist()
  for i in data:
    for j in set(list_country):
      if str(j) in i:
        country_in_data.append(j)
    for k in set(list_city):
      if k in i.split(' '):
        cities_in_data.append(k)
  text = ' '.join(data)
  if len(country_in_data) > 0:
    if len(cities_in_data) > 0:
      for i in cities_in_data:
        index = list_city.index(i)
        if list_country[index] in country_in_data:
          x = random.choice(list_country)
          text = text.replace(list_country[index], x)
          inde = list_country.index(x)
          text = text.replace(list_city[index], list_city[inde])
        else:
          text = text.replace(i, random.choice(list_city))
      for i in country_in_data:
        index = list_country.index(i)
        if list_city[index] in cities_in_data:
          x = random.choice(list_city)
          text = text.replace(list_city[index], x)
          inde = list_city.index(x)
          text = text.replace(list_country[index], list_country[inde])
        else:
          text = text.replace(i, random.choice(list_country))
  elif len(cities_in_data) > 0:
    for i in cities_in_data:
      text = text.replace(i, random.choice(list_city))
  return ' '.join(nltk.sent_tokenize(text))

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
  output_file = country_city(data)
  output_file = output_file.replace(' $$$$ . ', '\n\n')

  if args.output:
    file = open(args.output, 'w')
    file.write(output_file)


