import requests
import json
import base64

#rest api
url = "http://localhost:11434/api/generate"

img_path = "mine.png"

with open(img_path,"rb") as f:
    img_byte = f.read()
image_64 = base64.b64encode(img_byte).decode("utf-8")

payload = {
    "model":"llama3.2:1b",
    "prompt":"Explain what is machine leanring.",
    "image":[image_64]
}

resp = requests.post(url,json=payload,stream=True)

# for i in resp.iter_lines():
#     print(i)

output=""
for i in resp.iter_lines():
    if i:
        data = json.loads(i.decode("utf-8"))
        if "response" in data:
            output += data["response"]
        if data.get("done"):
            break

print(output)