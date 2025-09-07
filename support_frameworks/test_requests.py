import requests

print(requests.get("http://127.0.0.1:8000/items/?name=Nails").json())

## adding an item
print(requests.post("http://127.0.0.1:8000/",json={"name":"Screwdriver","price":5.99,"stock":40,"id":4,"category":"tools"}).json())