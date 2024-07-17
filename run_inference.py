#run_inference.py

import requests

url = 'http://localhost:8000/v2/models/bloom-560m/infer'

sentences = [ "The happy cat is" ]
data = {
    'inputs': [
        {
            "name" : "input",
            "datatype" : "BYTES",
            "shape" : [ 1 ],
            "data" : sentences
        }
    ],
    'params.': [
        {
            'binary_data_size' : 0 ,
            'shared_memory_region' : 'bloom-560m',
        }
    ]
}
response = requests.post(url, json=data)
answer = response.json()
print(answer)
for i in range(len(answer['outputs'])) : print('\n 초기 문장 :', sentences, '\n 문장 완성 :', answer['outputs'][i]['data'])