import logging
import os

from dotenv import load_dotenv
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Allow debug logs to propagate

def check_ip(version):
    url = "http://icanhazip.com"  # Works for both IPv4 and IPv6
    headers = {}

    if version == 4:
        headers["User-Agent"] = "curl/7.64.1"  # Mimic curl for IPv4
    elif version == 6:
        url = "http://[icanhazip.com]"  # Brackets force IPv6 resolution
        headers["User-Agent"] = "curl/7.64.1"

    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.text.strip()
    except requests.RequestException:
        return None

def supa_connx_str():
    """
    v: 4 or 6
    """
    ipv4, ipv6 = check_ip(4), check_ip(6)
    if ipv4: v = 4
    elif ipv6: v = 6
    else:
        logger.exception("No connection can be established")  # Captures stack trace
        raise ConnectionError("Could not determine a valid IP connection (IPv4 or IPv6)")

    for name, value in os.environ.items():
        if name.startswith(f"SUPABASE_IPv{v}"):
            del os.environ[name]
    load_dotenv()

    connection_string = (
        f"dbname={os.environ[f'SUPABASE_IPv{v}_database']}",
        f"user={os.environ[f'SUPABASE_IPv{v}_user']}",
        f"password={os.environ[f'SUPABASE_IPv{v}_password']}",
        f"host={os.environ[f'SUPABASE_IPv{v}_host']}",
        f"port={os.environ[f'SUPABASE_IPv{v}_port']}"
    )
    connection_string = " ".join(s for s in connection_string)

    return connection_string