import time
import requests

class HTTPClientError(Exception):
    pass

class HTTPClient:
    def __init__(self, host, port, timeout=60, max_retries=1, backoff_seconds=1, logger=None):
        self.base_url = f"http://{host}:{port}/api"
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.logger = logger
        
    def post_json(self, endpoint, payload, retries=0):
        url = self.base_url + endpoint
        for i in range(retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                return response.status_code, response.content
            except requests.RequestException as e:
                msg = f"HTTP request failed: {e}"
                if self.logger:
                    self.logger.error(msg)
                if i == retries:
                    raise HTTPClientError(msg)
                time.sleep(self.backoff_seconds)
                
def wait_for_server_ready(host, port, logger=None):
    url = f"http://{host}:{port}/"
    for i in range(10):
        try:
            res = requests.get(url, timeout=2)
            return True
        except:
            if logger:
                logger.info("Waiting for server to be ready...")
            time.sleep(2)
    return False
