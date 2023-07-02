from train_glowtts import config
import json
import re

s = json.dumps(config, default=vars, indent=2)
s = re.sub(r'"test_sentences":\s*\[\],', '', s)
print(s)
