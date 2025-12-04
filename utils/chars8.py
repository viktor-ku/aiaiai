import secrets


def chars8() -> str:
    return secrets.token_hex(4)
