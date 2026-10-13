
import uuid

from auth.security import (
    AuthenticationError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def main() -> None:
    password = "Example-Strong-Password-123!"

    password_hash = hash_password(password)

    assert password_hash != password
    assert verify_password(password, password_hash)
    assert not verify_password("Wrong-Password", password_hash)

    user_id = uuid.uuid4()
    token = create_access_token(user_id)

    claims = decode_access_token(token)

    assert claims["sub"] == str(user_id)
    assert claims["type"] == "access"

    print("Password hashing: PASS")
    print("Password verification: PASS")
    print("JWT creation and verification: PASS")

    try:
        decode_access_token("invalid-token")
    except AuthenticationError:
        print("Invalid token rejection: PASS")
    else:
        raise AssertionError("An invalid token was accepted.")


if __name__ == "__main__":
    main()
