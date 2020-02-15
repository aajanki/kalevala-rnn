import re
import os
import os.path
import shutil


def main():
    datadir = 'data'
    destdir = 'preprocessed'

    shutil.rmtree(destdir)
    os.makedirs(destdir)

    section_titles = [
        r"^\w+ runo$",
        r"^\* \* \*$",
        r"^[A-ZÅÄÖ]+ KIRJA.?$",
        r"^[IV]+\.(?: \w+)+\.?$",
        r"^\d.$",
        r"^\d+[.:](?: [-\w'().]+,?)+[.,?!]?$",
        r"^[a-d]\)(?: [-\w'().]+,?)+[.,?!]?$",
        r"Yhteisiä Lauluja",
        r"Erityisiä Lauluja.",
        r"Helsingissä \d+ päivä \w+ \d+.",
    ]
    section_title_re = re.compile('|'.join(section_titles), re.MULTILINE)
    parenthesis_re = re.compile(r'^\([\w .]+\)$', re.MULTILINE)
    multiemptyline_re = re.compile(r'\n{2,}', re.MULTILINE)

    for filename in os.listdir(datadir):
        with open(os.path.join(datadir, filename), encoding='utf-8-sig') as inf:
            text = inf.read()
            text = remove_preface(text)
            text = remove_introductions(text)
            text = section_title_re.sub('\n', text)
            text = parenthesis_re.sub('\n', text)
            text = multiemptyline_re.sub('\n\n', text)
            text = text.replace(' (vaikka loppumattomaan)', '')
            while text.startswith('\n'):
                text = text[1:]
            while text.endswith('\n'):
                text = text[:-1]

            # The replace some rare letters with more common letter
            # (somewhat arbitrary) to minimize the vocabulary
            text = (text
                    .replace('b', 'p')
                    .replace('c', 'k')
                    .replace('d', 't')
                    .replace('Ö', 'ö'))

            # remove very rare letters (likely errors)
            text = (text
                    .replace('x', '')
                    .replace('*', ''))

        with open(os.path.join(destdir, filename), 'w', encoding='utf-8-sig') as outf:
            outf.write(text)


def remove_preface(text):
    filtered = []
    
    alkulause = False
    for line in text.split('\n'):
        if line == 'ALKULAUSE':
            alkulause = True
        elif not alkulause:
            filtered.append(line)
        elif alkulause and line.startswith('1.'):
            alkulause = False
            filtered.append(line)

    return '\n'.join(filtered)


def remove_introductions(text):
    filtered = []
    intro = False
    for line in text.split('\n'):
        if len(line) > 60:
            intro = True
        elif intro and line == '':
            intro = False
            filtered.append(line)
        elif not intro:
            filtered.append(line)

    return '\n'.join(filtered)


if __name__ == '__main__':
    main()
