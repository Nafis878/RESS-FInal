"""Build the LaTeX manuscript: order the bibliography by first citation, compile with pdflatex (twice).
Usage: python build_tex.py"""
import os, re, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
main = open('manuscript.tex').read()
body = main
for m in re.finditer(r'\\input\{([^}]+)\}', main):
    f = m.group(1) + ('' if m.group(1).endswith('.tex') else '.tex')
    if f != 'refs.tex': body += open(f).read()
order = []
for m in re.finditer(r'\\cite[pt]?\{([^}]+)\}', body):
    for k in m.group(1).split(','):
        k = k.strip()
        if k not in order: order.append(k)
items = {}
for line in open('refs_all.tex'):
    k = re.match(r'\\bibitem\{([^}]+)\}', line).group(1); items[k] = line
missing = [k for k in order if k not in items]
assert not missing, f'missing references: {missing}'
with open('refs.tex', 'w') as f:
    f.write(f'\\begin{{thebibliography}}{{{len(order)}}}\n'); f.writelines(items[k] for k in order); f.write('\\end{thebibliography}\n')
print('references:', len(order), '| unused:', sorted(set(items) - set(order)))
for _ in range(2):
    r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'manuscript.tex'], capture_output=True, text=True, errors='replace')
log = open('manuscript.log', errors='ignore').read()
errs = [l for l in log.splitlines() if l.startswith('!')]
warn = [l for l in log.splitlines() if 'undefined' in l.lower() or 'multiply defined' in l.lower()]
print('errors:', errs[:5]); print('warnings:', warn[:10])
print(subprocess.run(['pdfinfo', 'manuscript.pdf'], capture_output=True, text=True, errors='replace').stdout.split('Pages:')[1].split('\n')[0].strip(), 'pages')
