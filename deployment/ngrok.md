
# Deploying Streamlit via Ngrok (Notebook/Colab Method)

This project can be shared publicly from a notebook or Colab session using **Pyngrok** to tunnel the Streamlit app.

## Steps

### 1) Install the agent: https://ngrok.com/download

### 2) Get the Authentication

### 3) Install dependencies
```python
!pip -q install streamlit pyngrok
```

### 4) Start Streamlit in a background thread and open the tunnel

```python
import threading, subprocess, shlex, time
from pyngrok import ngrok

def run_streamlit():
    subprocess.run(shlex.split("fuser -k 8501/tcp"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd = "streamlit run app2.py --server.port 8501 --server.address 0.0.0.0"
    subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

thread = threading.Thread(target=run_streamlit)
thread.start()
time.sleep(5)
ngrok.set_auth_token("31HWMKvlPRgYvEoVPZ3Di45ev4o_7Qr7TAHexXq8672Wbpmhc")  
tunnel = ngrok.connect(8501, "http")
print("Public URL:", tunnel.public_url)

You’ll see output like:
```
Public URL: https://<id>.ngrok-free.app
```
Share that link — keep the notebook running while others use the app.

### Stopping the tunnel and app
```python
from pyngrok import ngrok
ngrok.kill()  
```

---

## Tips

- **Address binding:** In notebook/Colab environments, `--server.address 0.0.0.0` is required so Ngrok can reach Streamlit.
- **Port matching:** If you change Streamlit’s port, update it in both the run command and `ngrok.connect()`.

## Troubleshooting

- **Upstream connection failed**: Check that Streamlit is running on the same port/address you’re tunneling.
- **Ngrok YAML errors**: Use the Python method above without adding legacy keys like `headers:` in `ngrok.yml`.
- **Port in use**: Free it with `fuser -k 8501/tcp` (Linux) or change the port in both places.
