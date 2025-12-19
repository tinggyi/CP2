"""Simple diagnostics for CP2 project environment.
Run: python scripts/check_env.py
It will check: python deps, .env keys (Mistral + Google), and make a lightweight request to Mistral.
"""
import os
import sys
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def check_packages():
    print('Checking packages...')
    reqs = ['requests', 'sentence_transformers', 'google-cloud-vision']
    for r in reqs:
        try:
            __import__(r)
            print(f'  OK: {r}')
        except Exception as e:
            print(f'  MISSING: {r} -- {e}')


def check_keys():
    m = os.getenv('MISTRAL_API_KEY')
    g = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    print('MISTRAL_API_KEY:', 'SET' if m else 'NOT SET')
    print('GOOGLE_APPLICATION_CREDENTIALS:', g or 'NOT SET')
    if g and not os.path.exists(g):
        print('  -> GOOGLE_APPLICATION_CREDENTIALS file not found at path above')


def check_mistral():
    import requests
    k = os.getenv('MISTRAL_API_KEY')
    if not k:
        print('No MISTRAL_API_KEY set; skipping API test')
        return
    try:
        r = requests.post('https://openrouter.ai/api/v1/chat/completions',
                          headers={'Authorization': f'Bearer {k}'},
                          json={'model':'mistralai/devstral-2512:free','messages':[{'role':'user','content':'ping'}],'max_tokens':5},
                          timeout=10)
        print('Mistral status:', r.status_code)
        if r.status_code != 200:
            print('Response snippet:', r.text[:400])
    except Exception as e:
        print('Mistral request failed:', e)


if __name__ == '__main__':
    check_packages()
    check_keys()
    check_mistral()
