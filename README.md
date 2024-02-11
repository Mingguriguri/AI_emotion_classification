# AI_emotion_classification
[인공지능] 감정 분석기 with. Chat GPT

> [팀과제] ChatGPT의 코딩 요청하여 코딩 수준을 평가한 후 보고서를 작성하여 제출 (2023.5.24)
팀에서 프로그램 하나를 정해 챗GPT에게 요청한 후, 코딩 수준을 평가한 후, 보고서로 작성하여 제출
팀에서 선정한 프로그램을 chatGPT더러 만들어 달라고 하고, 그 결과를 가지고 여러가지 시험을 해보면 됩니다.
올바른 output을 받아도, "틀렸다. 다시 해달라"고 해보고, 미흡한 점이 있으면 가이드를 하여 제대로 된 것을 만들도록 하고, source code만 말고, flow graph, pseudo code,  test case도 달라고 해보고,학생이 작성한 code를 주면서, 무엇이 잘못되었던지 미흡한지 물어 보는 등입니다.
마지막으로, chatGPT가 가지고 올 긍정적/부정적 영향에 대해 생각해 보고,
이것이 교육, 인문계열, entertainment (웹툰, 영화, 공연 등), 문학, 예술 등에 미칠 영향에 대해 생각을 정리해서 제출하면 됩니다.

---
# 감정 분류 프로그램

> 역할 분담
- 김민수: chatGPT 수준 평가
- 김민정(ME): 감정 분석기 설명, 감정 분류 모델 구현 및 주석, 설명 작성
- 김홍주: chatGPT 수준 평가
- 김도연: 문서화, chatGPT 영향 분석

---
> **아래 내용은 내가 팀원들에게 '팀원들의 이해 및 레포트 작성 자료'로써 공유한 md 파일입니다. 제가 담당한 부분에 대해서만 있다는 점 참고해주세요**
- 자료조사 기간: 2023.5.6 ~ 2023.5.10
- 실제 모델 구현 기간: 2023.5.6 ~ 2023.5.21

# 자연어처리(Natural Languagel Processing; NLP)

- 컴퓨터가 인간의 언어를 이해, 생성, 조작할 수 있도록 해주는 인공지능(AI)의 한 분야
- 자연어 텍스트 또는 음성으로 데이터를 상호 연결

## Large Language Model

- Large Language Model(LLM) : 언어 데이터셋에 대해 학습된 딥러닝 모델
    - 대화형 챗봇, 번역, 문서 생성 등 다양한 NLP(Natural Language Processing)
- 특정 작업이나 도메인에 맞게 모델을 사용하려면 Fine-tuning이 필요함

## Transfer Learning(전이학습)


>📌 **학습 데이터가 부족한 분야의 모델 구축을 위해 데이터가 풍부한 분야에서 훈련된 모델을 재사용하는 학습 기법**

- 데이터 수가 많지 않거나 데이터를 확보하는데 많은 비용이 들 수 있음. 이를 해결하기 위해 사용함
- 특정 분야에서 학습된 신경망의 일부 능력을 유사하거나 전혀 새로운 분야에서 사용되는 신경망의 학습에 이용하는 방법
- 따라서 기존의 만들어진 모델을 사용하여 새로운 모델을 만들시 학습을 빠르게 하여, 예측을 더 높임
- Pretrained Model :전이학습에서 이용되는 학습된 신경망
    - **ex. ImageNet, ResNet, gooGleNet, VGGNet**
- 대규모의 데이터셋으로 잘 훈련된 Pretrained Model을 사용해서 사용자가 적용하려고 하는 문제 상황에 맞게 모델의 가중치를 약간씩 변화하여 사용함.
    - (실제로, CNN을 이용하는 경우에 처음부터 가중치를 초기화하는 것보단 pretrained model을 사용하여 어느저도 합당한 값을 가진 가중치를 사용한다.
- 사전 학습된 모델을 재정의했다면, pretrained model의 classfier는 삭제하고, 목적에 맞는 새로운 classifier를 추가함. 이렇게 새롭게 만들어진 모델을 Fine Tuning을 진행함

### Q. 왜 Transfer Learning을 사용하는가?

1. 이미 학습된 모델을 사용해 문제를 해결함
2. 이미 학습된 많은 모델은 적용하려는 데이터가 학습할 때의 데이터와 같은 분포를 가진다고 가정 했을 때 효율적. 
3. 새로운 문제를 해결할 때 데이터의 분포가 바뀌면 기전의 통계적 모델을 새로운 데이터로 다시 만들어야 할 때 좋음
4. 복잡한 모델일 때 학습 시간이 오래 걸릴 수 있고, 학습시키는데 어려움이 있음.
5. layer의 개수, Activation function, Hyper-parameters등 모델을 구성하는데 고려해야 할 사항이 많으며 직접 모델을 구성하여 학습시키는 것은 많은 시도가 필요함.

**⇒ 이러한 이유들로 인해 이미 잘 훈련된 모델이 있고, 만드려는 모델과 유사한 문제를 해결하는 모델일 경우 Transfer Learning(전이학습) 사용**


>📌 **이때 사용방법 : Fine-Tuning**

## Fine Tuning (파인 튜닝)

>📌 **기존에 학습되어져 있는 모델을 기반으로 아키텍쳐를 새로운 목적에 맞게 변형하고 
이미 학습된 모델 Weights로 부터 학습을 업데이트 하는 방법**


- 사전에 학습된 모델을 기반으로 특정 작업에 대한 추가 학습을 수행하여, 아키텍처를 새로운 목적에 맞게 변형하고 이미 학습된 모델의 가중치를 미세하게 조정하여 학습 시키는 방법
    - ex. 의료분야 챗봇을 만들기 위해 일반적인 언어 데이터셋뿐만 아니라 의료 관련 언어 데이터셋에 대해 추가 학습이 필요함
        
        ⇒ 따라서해당 도메인에 대한 충분한 데이터셋이 필요함.
        

## ⭐사용한 모델 : BERT, KoBERT

- **BERT**
    - **구글**에서 2018년에 공개됨
    - GPT 등 다른 모델과는 다르게 **양방향성**을 지향하기 때문에, 수많은 **NLP**의 한 획을 그은 모델로 평가받고 있음
    - **문맥 특성**을 활용하고 있고, **대용량 말뭉치**로 사전 학습이 이미 진행되어 언어에 대한 이해도가 높음
    - 하지만 BERT는 **한국어에 대해서 영어보다 정확도가 떨어짐**
    
- **KoBERT(Korean BERT)**
    - SKTBrain에서 공개함
    - **한국어 위키 5백만 문장과 한국어 뉴스 2천만 문장을 학습한 모델**
    - BERT모델 중에서 **KoBERT를 사용한 이유** : **“한국어”**에 대해 많은 사전 학습이 이루어져있고, 감정 분석할 때 긍정 부정으로 분류하는 것이 아닌 다중 분류가 가능하기 때문
    - 이는 자신의 목적에 따라서, **세부조정(Fine-tuning)이 가능하기 때문에 output lyaer만 추가로 달아주면 원하는 결과를 출력해낼 수 있음**

# 코드 설명

## 0. 데이터셋 설명 / 코딩 환경

- 네이버 리뷰 2중 분류 예제 코드 기반 ([https://github.com/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb](https://github.com/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb))
- AiHub의 감정 분류를 위한 대화 음성 데이터셋 이용
    - 7가지의 감정이 있음
        - 행복, 분노, 혐오, 공포, 중립, 슬픔, 놀람
- 코랩에서는 버전 충돌이 발생하여, **주피터 노트북**을 사용하여 코딩하고 테스트했음

(만든 파일은 따로 참고하는 게 좋을 듯!)

## 1. 환경 설정(🚨Error발생)
![1](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/5f6e4bee-e290-426f-9b20-15ce6f1ca0b0)

![2](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/768b0ca3-4886-4db4-8497-005dbb5b6c70)

KoBERT가 요구하는 최신 정보를 토대로 필요한 패키지를 설치한다. 

```python
!pip install mxnet
!pip install gluonnlp==0.8.0
!pip install pandas tqdm   
!pip install sentencepiece==0.1.91
!pip install transformers==4.8.2
!pip install torch
```

- `mxnet`이 우선 설치되어야 하며, `gluonnlp`는 0.8.0으로 설정한다.

## 2. github에서 KoBERT 파일을 로드하고 KoBERT모델을 불러오기

- 깃허브 파일의 kobert_tokenizer폴더를 다운받는다.
![3](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/7f58a279-ffc2-497d-bf30-b9394f59d5dd)

    ```python
    !pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
    ```

    
- 불러온 후, 우리가 필요한 tokenizer와 model, vocabulary를 불러온다.
    - tokenizer : `tokenizer`
    - model : `bertmodel`
    - vocabulary: `vocab`
      
    ![4](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/5bd59275-866b-4cec-b81e-9328cf103763)

    ```python
    from kobert_tokenizer import KoBERTTokenizer # 한국어 BERT 모델에 대한 tokenizer를 import
    from transformers import BertModel # transformers 라이브러리에서 BertModel을 import
    
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1') # pretrained된 'skt/kobert-base-v1' 모델로 KoBERTTokenizer를 초기화
    bertmodel = BertModel.from_pretrained('skt/kobert-base-v1') # pretrained된 'skt/kobert-base-v1' 모델로 BertModel을 초기화
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]') # tokenizer의 vocab_file을 사용하여 BERTVocab을 생성.padding 토큰은 '[PAD]'로 설정 
    
    #model, vocab = BertModel.from_pretrained('skt/kobert-base-v1', tokenizer.vocab_file)
    
    print(vocab)
    ```
    
    - 이 부분에서 굉장히 많은 오류가 발생한다.
    - 블로그에 나와있는 깃허브 방법이 아닌 **hugging face**를 통해 가져온다.
    

## 3. 필요한 라이브러리 불러오기 (🚨Error발생)

사전 학습된(pre-trained) 모델인 BERT를 사용할 때는 transformers라는 패키지를 자주 사용하기 때문에 호출하였다.

또한, 학습시간을 줄이기 위해 GPU를 사용한다.


```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel

#GPU 사용 시
device = torch.device("cuda:0")
```

## 4. 데이터셋 불러오기

이전에 AiHub에 기재되었던 데이터(현재는 없음)를 올려준 블로그를 통해서 다운 받았다.


![5](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/4c8aae58-eebb-446d-87b5-4b404342dfab)

```python
data = pd.read_excel('/home/jihwan/MinJeong Archive/감정분류데이터셋.xlsx')
data.sample(n=10)
```
![Untitled](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/6eb7d4c1-6400-4109-994b-d27ed87dc563)


해당 데이터 셋에 대해서 총 7개의 감정 class를 0~6개의 숫자(label)에 대응시켜 data_list에 담아준다.
![6](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/0fd1ccbb-d68d-4909-899c-0b2492dd9cdf)

```python
#loc를 사용하여 'Emotion' 열의 값이 특정 문자열과 일치하는 행을 찾고, 해당 행들의 'Emotion' 열 값을 새로운 값으로 변경하는 작업을 수행
data.loc[(data['Emotion'] == "공포"), 'Emotion'] = 0  #공포 => 0
data.loc[(data['Emotion'] == "놀람"), 'Emotion'] = 1  #놀람 => 1
data.loc[(data['Emotion'] == "분노"), 'Emotion'] = 2  #분노 => 2
data.loc[(data['Emotion'] == "슬픔"), 'Emotion'] = 3  #슬픔 => 3
data.loc[(data['Emotion'] == "중립"), 'Emotion'] = 4  #중립 => 4
data.loc[(data['Emotion'] == "행복"), 'Emotion'] = 5  #행복 => 5
data.loc[(data['Emotion'] == "혐오"), 'Emotion'] = 6  #혐오 => 6

data_list = []

# 데이터프레임의 'Sentence'과 'Emotion' column을 돌면서
for ques, label in zip(data['Sentence'], data['Emotion'])  :
    data = []   #  각 반복에서 새로운 데이터 리스트를 초기화한다.
    data.append(ques) # 리스트에 문장(ques)와 변환된 감정레이블을 추가
    data.append(str(label)) #감정 레이블을 문자열러 변환

    data_list.append(data) #처리한 데이터 리스트를 전체 데이터리스트에 추가
```

## 5. 입력 데이터셋을 토큰화하기 (🚨Error발생)

각 데이터가 BERT 모델의 입력으로 들어갈 수 있도록 **tokenization, int encoding, padding** 등을 해주는 코드이다.


```python
class BERTDataset(Dataset):

	# 초기화 함수, Dataset 객체를 생성할 때 실행
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len,
                 pad, pair):
				# BERT 모델에 적합한 형태로 문장을 변환하는 transform 함수를 생성
        # BERTSentenceTransform 함수는 주어진 문장을 BERT 모델이 처리할 수 있는 형태로 변환함
				# 해당 함수부분에서 에러 발생
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset] # 입력받은 dataset에서 각 문장(sent_idx에 해당하는 항목)을 BERT 모델이 이해할 수 있도록 변환하고 저장
        self.labels = [np.int32(i[label_idx]) for i in dataset] # 입력받은 dataset에서 각 문장의 레이블(label_idx에 해당하는 항목)을 저장

		# i에 해당해는 샘플을 반환하는 함수, 데이터를 불러올 때 사용
    def __getitem__(self, i): 
        return (self.sentences[i] + (self.labels[i], ))
         
		# 총 샘플의 수(데이터셋의 길이) 반환 함수
    def __len__(self):
				return len(self.sentences)
```

- **에러 발생 부분**
    - 주석으로 표시해놓은 부분, `nlp.data.BERTSentenceTransform`은 위에서 에러가 났던, `import gluonnlp as nlp`에서와 같은 패키지이다.
    - 이 부분에서는 파이썬이  업그레이드되면서 `gluonnlp`와는 호환이 되지 않으면서 문제가 발생한 것이며, 정확히는 `BERTSentenceTransform`에서 난 에러이다.
    - 이를 해결하기 위해  여러 자료를 찾아보았으며 깃허브도 찾아보았지만, 깃허브에도 업그레이드된 사항이 나와있지 않았다.
    - 이후, 다른 사람이 직접 `BERTSentenceTransform`을 직접 새로 구현하신 분의 자료를 발견하여 코드를 수정하였다. ([https://blog.naver.com/newyearchive/223097878715](https://blog.naver.com/newyearchive/223097878715))
        - py파일에서 class를 복붙하여 코드를 수정한다.
        - def __init__에서 input에 vocab를 받는 부분 추가,  self._vocab = vocab 을 추가하고,def __call__에서 vocab = self._vocab로 바꿔주셨다.

BERT 모델에 적합한 형태로 데이터를 변환하는 클래스인 **`BERTSentenceTransform`**를 재정의

![7(1)](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/204dac3d-3d5c-4b4a-a9db-934a8ae91f33)
![7(2)](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/9e1d28c4-045e-4b54-a1e5-0a15553ac6c4)


```python
class BERTSentenceTransform:
    """BERT style data transformation.

    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    """
		# 초기화 메서드, BERTSentenceTransform 객체가 생성될 때 실행
    def __init__(self, tokenizer, max_seq_length,vocab, pad=True, pair=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair
        self._vocab = vocab 

		# 객체가 함수처럼 호출될 때 실행되는 함수; bert모델에 적합한 형태로 문장을 변환함
    def __call__(self, line):
        # 입력된 문장을 unicode 형태로 변환
        text_a = line[0]
        if self._pair:
            assert len(line) == 2
            text_b = line[1]

				# 문장을 BERT 모델이 이해할 수 있는 토큰으로 분리
        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if self._pair:
            tokens_b = self._tokenizer(text_b)

				# 여기에서 tokens_a와 tokens_b는 BERT 모델의 입력으로 사용되는 토큰

        if tokens_b:
            # 이 토큰들의 길이가 max_seq_length를 초과하지 않도록 조정
		        # 이때 [CLS], [SEP], [SEP] 토큰의 자리를 확보하기 위해 "-3"
						# 각 시퀀스는 [CLS], [SEP] 토큰을 포함해야 하며, 두 시퀀스를 연결할 때는 두 번째 [SEP] 토큰이 필요하므로, 이 세 개의 토큰에 대한 공간을 확보하기 위해 최대 시퀀스 길이에서 3을 뺌
						# 즉, 전체 토큰의 개수가 [CLS], tokens_a, [SEP], tokens_b, [SEP]를 포함한 최대 길이를 초과하지 않도록 tokens_a와 tokens_b의 길이를 조정
            self._truncate_seq_pair(tokens_a, tokens_b,
                                    self._max_seq_length - 3)
        else:
            # 만약 tokens_b가 없다면, [CLS], [SEP] 토큰의 자리를 확보하기 위해 "-2"을 합니다.
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]

        # The embedding vectors for `type=0` and `type=1` were learned during
        # pre-training and are added to the wordpiece embedding vector
        # (and position vector). This is not *strictly* necessary since
        # the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned

        vocab = self._vocab #vocab = self._tokenizer.vocab
        tokens = []
        tokens.append(vocab.cls_token)
        tokens.extend(tokens_a)
        tokens.append(vocab.sep_token)
        segment_ids = [0] * len(tokens) 

        if tokens_b:
            tokens.extend(tokens_b)
            tokens.append(vocab.sep_token)
            segment_ids.extend([1] * (len(tokens) - len(segment_ids)))

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens) # 토큰을 해당하는 id로 변환

        valid_length = len(input_ids) # 실제 토큰의 길이를 저장

				# pad 옵션이 True인 경우, 문장의 길이를 max_seq_length로 맞춰주기 위해 padding을 추가
        if self._pad:
            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_length - valid_length
            # use padding tokens for the rest
            input_ids.extend([vocab[vocab.padding_token]] * padding_length)
            segment_ids.extend([0] * padding_length)
				

				# 결과 반환
        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
            np.array(segment_ids, dtype='int32')
```

이후 BERTDataset을 다시 정의한다. 해당 부분은 위와 

![8](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/03d2de05-915d-49cc-9ed4-b53370fa7065)

```python
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

class BERTDataset(Dataset):

	# 초기화 함수, Dataset 객체를 생성할 때 실행
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):

				# BERT 모델에 적합한 형태로 문장을 변환하는 transform 함수를 생성
        # BERTSentenceTransform 함수는 주어진 문장을 BERT 모델이 처리할 수 있는 형태로 변환함
				# 해당 함수부분에서 에러 발생
        transform = BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        #transform = nlp.data.BERTSentenceTransform(
        #    tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset] # 입력받은 dataset에서 각 문장(sent_idx에 해당하는 항목)을 BERT 모델이 이해할 수 있도록 변환하고 저장
        self.labels = [np.int32(i[label_idx]) for i in dataset]  # 입력받은 dataset에서 각 문장의 레이블(label_idx에 해당하는 항목)을 저장

		# i에 해당해는 샘플을 반환하는 함수, 데이터를 불러올 때 사용
    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
		
	# 총 샘플의 수(데이터셋의 길이) 반환 함수
    def __len__(self):
        return (len(self.labels))
```

- **`__init__`** 함수
    - BERTDataset 객체가 생성될 때 실행되며, BERT 모델에 필요한 데이터 전처리를 수행하ㅣ는 함수
    - **`transform`** 함수를 통해 각 문장은 BERT 모델이 처리할 수 있는 형태로 변환된다.
- **`__getitem__`** 함수
    - 인덱스 **`i`**에 해당하는 데이터를 반환하는 함수
    - 이 함수는 데이터 로더에서 사용되며, 특정 인덱스의 입력 데이터와 그에 해당하는 레이블을 반환한다.
- **`__len__`** 함수
    - 데이터셋의 총 샘플 수를 반환하는 함수
    - 이 함수는 데이터 로더가 배치를 생성할 때 필요한 정보를 제공한다.
    - 이를 통해 데이터 로더는 전체 데이터를 얼마나 많은 배치로 나눌 수 있는지를 결정할 수 있다.

이를 통해 data_train에서 `tok`으로 설정했던 부분을 `tokenizer`로 설정하여 수정하였다.


```python
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1') #미리 훈련된 'skt/kobert-base-v1' 모델을 기반으로 KoBERT 토크나이저를 로드합니다.
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False) # 마찬가지로 'skt/kobert-base-v1' 모델을 기반으로 BertModel을 로드합니다. 'return_dict=False'는 모델 출력을

# 토크나이저의 단어장 파일을 사용하여 BERTVocab 객체를 생성 
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

data_train = BERTDataset(dataset_train, 0, 1, tokenizer, vocab, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tokenizer, vocab, max_len, True, False)
# BERTDataset 클래스를 사용하여 훈련 데이터셋과 테스트 데이터셋을 처리
```

`parameter`의 경우, 예시 코드에 있는 값들을 동일하게 설정해주었다.

![9](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/8421db3b-ee12-479a-a486-0a45334803fb)

```python
# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5  
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5
```

사이킷런에서 제공해주는 `train_test_split` 메서드를 활용해 기존 `data_list`를 **train 데이터셋과 test 데이터셋**으로 나눈다. 5:1 비율로 나누었다.

![10](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/dd49f317-71fc-44cc-9a2d-86e313fab966)

```python
#train & test 데이터로 나누기
from sklearn.model_selection import train_test_split
dataset_train, dataset_test = train_test_split(data_list, test_size=0.2, shuffle=True, random_state=34)
```

위에서 구현한 `BERTDataset` 클래스를 활용해 **tokenization, int encoding, padding** 을 진행하였다.
![11](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/7af85489-48b1-4fd5-b32e-acb2a229c836)
<<!!!!!!!!!>>
```python
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)
```

torch 형식의 dataset을 만들어주면서, 입력 데이터셋의 처리가 모두 끝났다.

## 6. KoBERT 모델 구현하기
![12](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/e3969ffe-f240-4f42-a7dd-f5647fb4bb4c)

```python
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,  # BERT 모델의 hidden layer 크기
                 num_classes=7,  # 분류할 클래스의 개수
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__() 
        self.bert = bert
        self.dr_rate = dr_rate
        # 출력 레이어.
				#BERT의 hidden_size를 입력으로 받고, 출력은 클래스의 수로 설정   
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate) #
    
# BERT의 어텐션 메커니즘이 동작하기 위한 마스크를 생성하는 함수
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
		# 순전파 함수 정의
    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length) # 어텐션 마스크 생성
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
```

BERT모델을 불러온다. 
![13](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/a1ef00a1-c24e-481c-b2ed-86609a7c4a63)

```python
# BERTClassifier 클래스를 사용해 모델을 생성
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

#optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss() # 다중분류를 위한 대표적인 loss func

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
    
train_dataloader
```

해당 부분은 예제 코드와 동일하게 사용하였다.

## 7. 모델 학습시키기

- KoBERT 모델을 학습시키는 코드이다. **epoch**는 5로 지정하였다.
- 앞서 이 모델에서 학습시킬 수 있도록 입력 데이터셋을 처리하고, 파라미터를 모두 지정하였으므로 예시 코드와 동일하게 진행하였다.
- 이 부분에서 원본 예제 코드와 다른 점은 정확도가 초반에 17%로 나왔다는 것이었다.
- 따라서, 이 부분을 위하여 위에 `tokens_a = self._tokenizer(text_a)`로 되어 있던 부분을 `tokens_a = self._tokenizer.tokenize(text_a)`로 수정함으로써 정확도를 높였다.

![14(1)](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/a66d4a00-0f53-4eb9-85df-cd8591b7b447)

```python
train_history = []  # 훈련 정확도 기록 리스트
test_history = []  # 테스트 정확도 기록 리스트
loss_history = []  # 손실 기록 리스트

# 지정된 에폭 수만큼 반복
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()  # 모델을 훈련 모드로 설정

    # 훈련 데이터 로더에서 배치별로 데이터를 가져와 학습을 수행
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

        # 손실 계산 및 역전파 수행
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  
        train_acc += calc_accuracy(out, label)

        # 일정 주기마다 훈련 상태를 출력하고, 훈련 정확도와 손실을 기록
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(), train_acc / (batch_id + 1)))
            train_history.append(train_acc / (batch_id + 1))
            loss_history.append(loss.data.cpu().numpy())

    print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))

    model.eval()  # 모델을 평가 모드로 설정

    # 테스트 데이터 로더에서 배치별로 데이터를 가져와 평가
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)

    print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))
    test_history.append(test_acc / (batch_id + 1))
```
![Untitled 1](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/52d33242-4cc9-48de-a10a-a12ea059d24d)

- 수정하고 나니, **train dataset에 대해서는 0.979, test dataset에 대해서는 0.918**의 정확도를 기록했다.

## 8. 직접 입력한 새로운 문장으로 테스트

이제 직접 문장을 만들어 학습된 모델이 다중 분류를 잘 해내는지 알아보기 위해 학습된 모델을 활용하여 다중 분류된 클래스를 출력해주는 **`predict 함수`**를 구현한다.

- **`predict`** 함수는 예측할 문장을 입력으로 받아 해당 문장에 대한 감정을 예측하고 출력한다.
- 입력 문장을 데이터 형식에 맞게 리스트로 생성하고, 이를 데이터셋 형식으로 변환한다.
- 변환한 데이터셋을 사용하여 데이터 로더를 생성하고, 모델을 eval모드로 바꾼다.
- 데이터 로더에서 배치별로 데이터를 가져와 모델에 입력하여 예측 결과를 얻어 레이블을 생성하고 출력한다.

![15](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/abb740f7-cafc-450a-8c40-72bbb38354e0)

```python
def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data] 

    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model.eval() # 모델 평가모드

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids) # 모델에 문장을 입력하여 예측 결과를 얻는다.

        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

						# 예측 결과를 기반으로 감정 레이블을 생성
            if np.argmax(logits) == 0:
                test_eval.append("공포가")
            elif np.argmax(logits) == 1:
                test_eval.append("놀람이")
            elif np.argmax(logits) == 2:
                test_eval.append("분노가")
            elif np.argmax(logits) == 3:
                test_eval.append("슬픔이")
            elif np.argmax(logits) == 4:
                test_eval.append("중립이")
            elif np.argmax(logits) == 5:
                test_eval.append("행복이")
            elif np.argmax(logits) == 6:
                test_eval.append("혐오가")

        print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")
```

질문 및 입력을 무한 반복한다. 입력하면 입력한 내용에 따라 `predict()`에서 감정을 분류한 값을 출력한다. `0`을 입력하면 무한 반복이 종료된다.

![16](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/26673eed-794b-4dd8-8d62-10afd1d8e40e)

```python
#질문 무한반복하기! 0 입력시 종료
end = 1
while end == 1 :
    sentence = input("하고싶은 말을 입력해주세요 : ")
    if sentence == "0" :
        break
    predict(sentence)
    print("\n")
```
![result(2)](https://github.com/Mingguriguri/AI_emotion_classification/assets/101111603/33f8a06a-b7e2-4619-a77d-eca6bfd11b1d)



# 결론

수업시간에 배운 자연어처리를 더 깊게 나아가, 감정별로 분류를 해보았다. 데이터셋을 직접 fine tuning하는 과정을 거쳐 직접 질문에 분류하는 것까지 해보았다. 

후에 직접 데이터셋을 추가하고, 데이터를 더 정제하여 정확도도 높이면 더 도움이 될 것 같다.

---

참고자료

- [https://hoit1302.tistory.com/159](https://hoit1302.tistory.com/159)
- [https://github.com/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb](https://github.com/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb)
- 버그 해결 관련하여 참고한 곳 : SKTBrain/KoBERT깃허브 contact 페이지 https://github.com/SKTBrain/KoBERT/issues
- No module named ‘kobert’ 에러 해결, [https://blog.naver.com/newyearchive/223097878715](https://blog.naver.com/newyearchive/223097878715)
