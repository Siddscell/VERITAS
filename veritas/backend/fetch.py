import urllib.request, json
with open('00025_fake.mp4', 'rb') as f:
    b = f.read()
body = b'--b\r\nContent-Disposition: form-data; name="file"; filename="00025_fake.mp4"\r\nContent-Type: video/mp4\r\n\r\n' + b + b'\r\n--b--\r\n'
req = urllib.request.Request('http://127.0.0.1:8000/analyse/stream', data=body, headers={'Content-Type': 'multipart/form-data; boundary=b'})
r = urllib.request.urlopen(req)
for line in r:
    print(line.decode().strip())
