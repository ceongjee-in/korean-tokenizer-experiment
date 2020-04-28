from konlpy.tag import Mecab
import sentencepiece as spm
import sys

NO_JONGSUNG = 'ᴕ'

CHOSUNGS = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JOONGSUNGS = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNGS = [NO_JONGSUNG,  'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

N_CHOSUNGS = 19
N_JOONGSUNGS = 21
N_JONGSUNGS = 28

FIRST_HANGUL = 0xAC00 #'가'
LAST_HANGUL = 0xD7A3 #'힣'

def to_jaso(s):        
    result = []
    for c in s:
        if ord(c) < FIRST_HANGUL or ord(c) > LAST_HANGUL: # if a character is a hangul
            result.append(c)
        else:            
            code = ord(c) - FIRST_HANGUL
            jongsung_index = code % N_JONGSUNGS
            code //= N_JONGSUNGS
            joongsung_index = code % N_JOONGSUNGS
            code //= N_JOONGSUNGS
            chosung_index = code

            result.append(CHOSUNGS[chosung_index])
            result.append(JOONGSUNGS[joongsung_index])
            result.append(JONGSUNGS[jongsung_index])
    
    return ''.join(result)    

tagger = Mecab()

def tokenize_by_morpheme_char(s):
    return tagger.morphs(s)

def tokenize_by_morpheme_jaso(s):
    return [to_jaso(token) for token in tokenize_by_morpheme_char(s)]

sp_char = spm.SentencePieceProcessor()
sp_char.Load("model/spm.char.model")

def tokenize_by_subword_char(s):    
    return sp_char.EncodeAsPieces(s)

def tokenize_by_eojeol_char(s):
    return s.split(' ')    

def tokenize_by_eojeol_jaso(s):
    return [to_jaso(token) for token in tokenize_by_eojeol_char(s)]

# This is a hack which is easy to be broken later
def get_tokenizer(method, unit):
    return getattr(sys.modules[__name__], 'tokenize_by_{}_{}'.format(method, unit))

if __name__ == '__main__':
    import argparse
    import sys
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('-method', choices=['eojeol', 'subword', 'morpheme'])
    parser.add_argument('-unit', choices=['jaso', 'char'], required=True)
    parser.add_argument('input_path')
    parser.add_argument('output_path')    
    args = parser.parse_args()    
    
    tokenizer = get_tokenizer(args.method, args.unit)

    lines = open(args.input_path).read().splitlines()

    with open(args.output_path, 'w') as out:
        for line in tqdm(lines, unit=' line'):
            tokens = tokenizer(line)
            out.write(' '.join(tokens) + '\n')