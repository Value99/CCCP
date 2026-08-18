"""Brace/paren balance check for .cu sources (comments/strings stripped)."""
import re
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
src = re.sub(r"//[^\n]*", "", src)
src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
src = re.sub(r'"(\\.|[^"\\])*"', '""', src)
src = re.sub(r"'(\\.|[^'\\])*'", "'c'", src)

bal = par = 0
line = 1
for ch in src:
    if ch == "\n":
        line += 1
    elif ch == "{":
        bal += 1
    elif ch == "}":
        bal -= 1
    elif ch == "(":
        par += 1
    elif ch == ")":
        par -= 1
    if bal < 0 or par < 0:
        print(f"NEGATIVE at line {line}: brace={bal} paren={par}")
        sys.exit(1)
print(f"final: brace={bal} paren={par} lines={line}")
