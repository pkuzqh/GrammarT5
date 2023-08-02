import re

#def detokenize_code(tokenized_code: str) -> str:
#    tokenized_code = re.sub(r'\s*([\(\)\[\]\{\}\.,;=<>!+-/*%&|^?:])\s*', r'\1', tokenized_code)
#    tokenized_code = re.sub(r'\b(if|else|for|while|switch|return)\b\s*(?=\()', r'\1 ', tokenized_code)
#    return tokenized_code
def detokenize_code(tokenized_code: str) -> str:
    tokenized_code = re.sub(r'\s*([\(\)\[\]\{\}\.,;<>!+-/*%&|^?:])\s*', r'\1', tokenized_code)
    tokenized_code = re.sub(r'\b(if|else|for|while|switch|return)\b\s*(?=\()', r'\1 ', tokenized_code)
    tokenized_code = re.sub(r'(\w+)\s*=\s*(\w+)', r'\1 = \2', tokenized_code)  # 在等号两边添加空格
    tokenized_code = re.sub(r',(\w+>)', r', \1', tokenized_code)  # 修复泛型逗号问题
    return tokenized_code
with open('out.txt', 'r') as f:
    code = f.read()

formatted_code = ''
for line in code.split('\n'):
    formatted_line = detokenize_code(line)
    formatted_code += formatted_line + '\n'

with open('codetrans.txt', 'w') as f:
    f.write(formatted_code.strip())
