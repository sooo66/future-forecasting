# read data/url_pool/url_pool.jsonl
# calculate the total number of kinds of domain
import json

domains = []

with open('data/url_pool/url_pool.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        domain = data['domain']
        if domain not in domains:
            domains.append(domain)
print(len(domains))
print(domains)