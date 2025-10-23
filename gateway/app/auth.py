import time, hmac, hashlib, base64, json, os
SECRET = os.getenv("JWT_SECRET","dev-secret")

def sign(payload: dict, exp_sec: int = 3600):
    p = payload | {"exp": int(time.time()) + exp_sec}
    raw = base64.urlsafe_b64encode(json.dumps(p).encode()).rstrip(b"=")
    sig = hmac.new(SECRET.encode(), raw, hashlib.sha256).digest()
    tok = raw + b"." + base64.urlsafe_b64encode(sig).rstrip(b"=")
    return tok.decode()

def verify(token: str):
    try:
        raw, sig = token.split(".")
        expect = base64.urlsafe_b64encode(hmac.new(SECRET.encode(), raw.encode(), hashlib.sha256).digest()).rstrip(b"=").decode()
        if sig != expect: return None
        data = json.loads(base64.urlsafe_b64decode(raw + "=="))
        if data.get("exp",0) < int(time.time()): return None
        return data
    except Exception:
        return None