from curl_cffi import requests

try:
    r = requests.get("https://rumble.com/c/nicholasjfuentes", impersonate="chrome")
    print(f"Status: {r.status_code}")
    print(f"Content length: {len(r.text)}")
    if r.status_code == 200:
        print("Success!")
    else:
        print("Failed!")
except Exception as e:
    print(f"Error: {e}")
