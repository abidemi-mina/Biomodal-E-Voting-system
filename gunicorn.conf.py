# gunicorn.conf.py
# Production WSGI server configuration for NigeriaVotes

import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Workers
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 60
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "nigerialvotes"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
