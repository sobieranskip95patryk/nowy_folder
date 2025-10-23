from fastapi import Header, HTTPException
from enum import Enum
from .auth import verify

class Role(str, Enum):
    METAGENIUSZ="MetaGeniusz"; GPT_PRESIDENT="GPT-President"; GPT_KING="GPT-King"; GPT_ORGANIZATION="GPT-Organization"; USER="User"
    
HIER = {Role.METAGENIUSZ:5, Role.GPT_PRESIDENT:4, Role.GPT_KING:3, Role.GPT_ORGANIZATION:2, Role.USER:1}

def require(role_min: Role):
    def _inner(authorization: str = Header(default="")):
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing token")
        tok = authorization.split(" ",1)[1]
        data = verify(tok)
        if not data: raise HTTPException(401, "invalid token")
        role = Role(data.get("role","User"))
        if HIER[role] < HIER[role_min]:
            raise HTTPException(403, "forbidden")
        return data
    return _inner