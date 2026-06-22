from dataclasses import dataclass
from enum import StrEnum


class UserType(StrEnum):
    PLUS = "plus"
    ENTERPRISE = "enterprise"


@dataclass(kw_only=True, frozen=True, slots=True)
class ChatContext:
    user_type: UserType = UserType.PLUS
