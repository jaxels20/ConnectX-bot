
import torch
import base64
import pickle

# Load your model's state dict
state_dict = torch.load('model.pth')

# Serialize and encode
state_dict_bytes = pickle.dumps(state_dict)
state_dict_str = base64.b64encode(state_dict_bytes).decode('utf-8')

# Now `state_dict_str` contains your model weights in a string format. 
# Embed this string into your `main.py` file.

# Write `state_dict_str` to a file
with open('state_dict.txt', 'w') as f:
    f.write(state_dict_str)