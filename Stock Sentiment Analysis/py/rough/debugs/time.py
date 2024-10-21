import time
from datetime import datetime

current_timestamp = time.time()
print(current_timestamp)
gmt_time = datetime.fromtimestamp(current_timestamp)
print(gmt_time)