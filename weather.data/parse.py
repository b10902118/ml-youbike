import json

# Assuming each line of your file contains a valid JSON object
json_file_path = "./rain/A0A010.json"

with open(json_file_path, "r") as file:
    for line in file:
        # Parse the JSON from the line
        json_data = json.loads(line)

        # Now you can work with the parsed data
        print(json_data)
