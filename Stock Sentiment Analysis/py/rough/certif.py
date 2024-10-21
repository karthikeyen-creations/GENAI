import httpx
import certifi

client = httpx.Client(verify=certifi.where())
