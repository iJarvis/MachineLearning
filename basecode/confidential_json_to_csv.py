import json
import csv

def merge_keys(d):
  to_return = {}
  for key, value in d.items():
    if not isinstance(value, dict):
      to_return[key] = value
    else:
      for merged_key, merged_value in merge_keys(value).items():
        to_return["_".join((key, merged_key))] = merged_value
  return to_return

data = json.loads(open('talent_profiles_10k (UMD-Confidential).json').read())



