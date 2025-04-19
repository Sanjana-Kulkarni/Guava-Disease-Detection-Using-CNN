import requests

url = 'http://localhost:5000/predict'
files = {'file': open('D:\guava\gPaper\test.jpeg', 'rb')}

response = requests.post(url, files=files)
print(response.json())