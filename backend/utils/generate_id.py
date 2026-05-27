import uuid

def unique_id() -> str:
    return str(uuid.uuid4())

if "__main__" == __name__:
    print(unique_id())