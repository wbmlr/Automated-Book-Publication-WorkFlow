def get_human_approval(text: str) -> bool:
    print("\n" + "="*20 + " FOR HUMAN REVIEW " + "="*20)
    print(text[:1000] + "...")  # Preview
    return input("Approve this version? [y/n]: ").lower() == 'y'

def get_human_feedback() -> str:
    return input("Provide feedback for the next iteration (or press Enter to skip): ")