import requests

headers = {
    'Content-Type': 'application/json',
}

data = '{"sentence":"this is a sentence"}'

response = requests.post('http://127.0.0.1:5000/api/pwg', headers=headers, data=data)

print(response.text)