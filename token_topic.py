from kiwipiepy import Kiwi
kiwi = Kiwi()
import re
import tomotopy as tp   # 토픽모델링 패키지
import pyLDAvis       # 시각화 패키지
import numpy as np    # 파이썬 계산기
import pandas as pd   # 파이썬 엑셀
from tqdm import tqdm

class Doc4Token:
    '''문서(문장)을 명사 형태소로 분리 등'''
    def __init__(self, doc=''):
        self.doc = doc

    # normalization
    def normalize(self, norm_dct={}):
        txt=self.doc
        chinese2korean={'與':'여당', '野':'야당', '政':'정부', '親':'친', '反':'반','尹':'윤석열', '文':'문재인', '黨':'당', '北':'북한', '美':'미국',
                    '日':'일본', '中':'중국', '佛':'프랑스', '俄':'러시아', '英':'영국', '獨':'독일','軍': '군대'}
        chinese2korean.update(chinese2korean)
        for k, v in chinese2korean.items():
            try: txt=txt.replace(k,v)
            except: txt=''
        return txt

    # 함수화: 명사 추출과 주요 품사 추출 수행
    def tokenize(self, nouns=True, remove1=False, stopwords=[]):
        '''문자열 txt를 받아 kiwi로 형태소 추출: nouns=명사만 추출 여부, remove1=1음절 토큰 제외 여부, stopwords=불용어 리스트 '''
        try:
            txt=self.doc
            # 정제(cleaning): 비문자숫자 등 노이즈 제거
            txt1=re.sub(r"[^\s가-힣a-zA-Z0-9]", " ", txt)   # re.sub: 문자열 부분 교체. r은 정규표현식 사용한다는 표시.
                                                            # "[^ 가-힣a-zA-Z1-9]"는 한글 영어 숫자 이외의 문자열 의미.
                                                            # txt1=txt1.replace("X", " "):  특정 단어만 삭제할 때에는 replace 함수로 간단히 실행
            # 토큰화(tokenization): 형태소 추출
            morphs=kiwi.tokenize(txt1)
            morphs_all=[m[0] for m in morphs]                # 모든 품사에 해당하는 형태소 모두 추출
            morphs_select=['NNG', 'NNP', 'NP', 'NR', 'VV', 'VX', 'VCP', 'VCN', 'VA','VA-I', 'MM', 'MAG']  # 일반명사, 고유명사, 용언(동사, 형용사 등), 관형사, 일반부사 # 품사 분류표 참조
            # 명사 추출(nou extraction) 여부 선택
            if nouns==True:
                token_lst=[m[0] for m in morphs if m[1] in morphs_select[:4]]
            else:
                token_lst=[m for m in morphs if m[1] in morphs_select]
                # stemming(어간 추출, 동사-형용사 등 용언의 원형 복구) 적용
                token_lst=[m[0]+'다' if m[1].startswith('V') else m[0] for m in token_lst]
            # 1음절 토큰 제외 여부 선택
            if remove1==True:
                token_lst=[t for t in token_lst if len(t)>1 ]
            else:
                pass
            # 불용어(stopwords) 적용: 제외해야 할 토큰들의 집합
            token_lst=[t for t in token_lst if t not in stopwords]
        except:
            token_lst=[]
        return ','.join(token_lst)

def add_kiwidct(words=[])
    '''kiwi 사전에 명사로 추가. 리스트 형태로 입력'''
    for word in words:
        kiwi.add_user_word(word, 'NNG')

def topic_model(docs=[""], k=20, min_cf=5, model_save=False, model_html=False):
    """토픽모델링 실시용 함수: docs=["단어,단어,단어","단어,단어,단어","단어,단어,단어"]로 입력, k=선정된 토픽 갯수 입력,
    model_save=모델 저장하려면 영어 절대경로 입력(C://path/---.bin). model_html=모델 시각화 저장하려면 영어 절대경로 입력(C://path/---.html)
    출력은 {'summary': summary, 'topic_term': topic_term, 'doc_topic': doc_topic} 형태. 모델 불러오기: tp.LDAModel.load()"""
    docs = [doc.split(",") if type(doc) == str else [""] for doc in docs]
    mdl = tp.LDAModel(k=k, alpha=0.1, eta=0.01, min_cf=min_cf, tw=tp.TermWeight.IDF) # min_cf: 코퍼스에서 최소 언급 횟수.. 보통 5.. #eta = beta에 해당
    for doc in docs:
        try:
            mdl.add_doc(doc)  # 한줄씩 입력해 mdl 만들기
        except:
            mdl.add_doc([""])
    mdl.burn_in = 100
    mdl.train(0)
    for i in tqdm(range(0, 1000, 10)):    # 100회 iteration
        mdl.train(10)
    summary=mdl.summary()
    if model_save:
        mdl.save(model_save)
    else:
        pass
    topic_term=pd.DataFrame([mdl.get_topic_words(k,top_n=30) for k in range(mdl.k)], \
                            index=["topic"+str(i) for i in range(k)], \
                            columns=["keyword"+str(i) for i in range(30)]).T
    docs_topic = []
    for i, doc in enumerate(mdl.docs):
        t=list(doc.get_topic_dist())
        t.append(t.index(max(t)))
        t.append(docs[i][0])
        docs_topic.append(t)
    doc_topic=pd.DataFrame(docs_topic, columns=["topic"+str(i) for i in range(k)]+["소속토픽","문서첫단어"])
    # 모델 시각화
    topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
    doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
    doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
    vocab = list(mdl.used_vocabs)
    term_frequency = mdl.used_vocab_freq
    prepared_data = pyLDAvis.prepare(
        topic_term_dists,
        doc_topic_dists,
        doc_lengths,
        vocab,
        term_frequency,
        start_index=0,  # tomotopy starts topic ids with 0, pyLDAvis with 1
        sort_topics=False)  # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
    if model_html:
        pyLDAvis.save_html(prepared_data, model_html)
    else:
        pass
    return {'summary': summary, 'topic_term': topic_term, 'doc_topic': doc_topic}


















