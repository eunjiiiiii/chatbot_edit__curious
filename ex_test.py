'''
msg_decode = '저한테는 뭐든 이야기해도 괜찮아요.기분이 우울하시군요. 걱정돼요.그런 일이 있으셨군요.또 다른 문제는 없으세요?불안하실 땐'
msg = msg_decode.split('.')[0] + '. ' + msg_decode.split('.')[1]
print(msg)
'''

"""
#msg = 'B-음식'
msg = ['O', 'B-신체부위', 'O']
result_entity = []
for e in msg:
    if '-' in e:
        result = e.split('-')[1]
        if result == '신체부위':
            result = 'body'
        elif result == '증상':
            result = 'symptom'
        elif result == '장소':
            result = 'place'
        elif result == '음식':
            result = 'food'
        result_entity.append(result)
        print(result_entity)

print(type(result_entity))
"""
#print(msg.split('-')[1])

#print(['body'] + [])

'''
from chatbot.model.intent_entity.tokenization_kobert import KoBertTokenizer
from chatbot.model.intent_entity.utils import load_tokenizer

print(KoBertTokenizer._tokenize(self,'나 화장실 갈래'))
print()
load_tokenizer()
'''

"""
str = '_화장실'
print(str.strip('_'))
str2 = '래'
print(str2.strip('_'))
"""

'''
inputs = ['▁나', '▁화장실', '▁갈', '래']

tokens = []
for e in inputs:
    if e in ['[CLS]', '[SEP]']:
        continue
    print(e.strip('▁'))
    tokens.append(e.strip('▁'))

tokens
'''

'''
text = '나 화장실 가고 싶어'
print(text.split(' '))
'''

'''
import re

msg = '괜찮으세요? 제가 도움이 되고 싶어요.저만 듣기 아까운 이야기네요. 기쁨은 나누면 두 배가 된대요!내일도 오늘 같을 거예요.'
print(msg.split('?')[0] + '?' + msg.split('.')[0] + '.')
#print(re.split('\.$ | \?$', msg))
#p = re.compile(r'\W+')
#print(p.split(msg))
'''

'''
msg = '그런 일이 있으셨다니 충격이에요.그런 일이 있으셨다니 충격이에요.정말요?  너무 속상해요.'
print(msg.split('.'))
print(msg.split('.')[0])
print(msg.split('.')[1])
print(msg.split('.')[2])
'''

'''
if msg.split('.')[0] != msg.split('.')[1]:
    msg_res = msg.split('.')[0] + '. ' + msg.split('.')[1]
else:
    msg_res = msg.split('.')[0] + '. ' + msg.split('.')[2]
print(msg_res)
'''
'''
l = [1,2,3]
l_2 = list(l)
print(l)
print(l_2)
'''

'''
from emotionchat_engine import EmotionChat,final_emotion, most_freq
emotions = ['감정없음', '감정없음', '감정없음', '감정없음']
emotion_prob = [0.7374757528305054, 0.4204759895801544, 0.8598405718803406, 0.919521689414978]

max_emotion = most_freq(emotions)
print('max_emotion : ' + str(max_emotion))
pos = [i for i in range(len(emotions)) if emotions[i] == max_emotion]
print(pos)
#max_emotion_probs = emotion_prob[pos]

max_emotion_prob = emotion_prob[pos[0]]
for idx in pos:
    if emotion_prob[idx] > max_emotion_prob:
        max_emotion_prob = emotion_prob[idx]

print('max_emotion_prob: ' + str(max_emotion_prob))
'''

import re
t = '점심은'
print(re.sub('[은는이가을를]', '', t))