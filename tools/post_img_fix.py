import sys
import re
import os

# More on regex:  
# https://note.nkmk.me/en/python-str-replace-translate-re-sub/

file_ = sys.argv[1]

with open(file_) as f:
    ff = f.read()

xx = re.sub("(\!\[.*\]\()(.*)(/.*\))", r"\1/assets/img/post/{{ page.pid }}\3", ff)

with open(os.path.basename(file_), "w") as f:
    f.write(xx)
