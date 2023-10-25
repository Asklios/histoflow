import os

# supported input formats
supported_formats = [".jpg", ".jpeg", ".png", ".tiff", ".tif"]


def filter_supported_files(directory):
    return [f for f in os.listdir(directory) if any(f.lower().endswith(ext) for ext in supported_formats)]


def request_yes_no():
    while True:
        user_input = input()
        if user_input.lower() == 'yes' or user_input.lower() == 'y':
            return True
        elif user_input.lower() == 'no' or user_input.lower() == 'n':
            return False
        else:
            print('Invalid input. Please enter "yes" or "no"')


# add your notification code here
def notify(message: str):
    pass
