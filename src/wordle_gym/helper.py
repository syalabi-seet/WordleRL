from string import ascii_uppercase as alphabets
import json

if __name__ == "__main__":
    position_dict = {"": 0}

    i = 1
    for letter in alphabets:
        for flag in range(1, 4):
            position_dict[f"{letter}|{flag}"] = i
            i += 1

    with open("state_dict.json", "w") as f:
        json.dump(position_dict, f, indent=4)