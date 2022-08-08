from emotionchat_engine import EmotionChat, final_emotion
import torch
import random
import os
import numpy as np

'''
import sys
sys.path.insert(0, 'C:/Users/eunji/PycharmProjects/pythonProject/chatbot')
'''

import sys
print('Python %s on %s' % (sys.version, sys.platform))

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



emotionchat = EmotionChat()

if __name__ == '__main__':
    # initialization
    seed_everything(1234)
    conversation_history = []
    turn_cnt, intent_turn_cnt = 0, 0
    turn_cnts = {'turn_cnt': turn_cnt,
                 'intent_turn_cnt': intent_turn_cnt
                 }
    result_dict = {
        'input': [],
        'intent': '',
        'entity': [],
        'state': '',
        'emotion': '',
        'emotions': [],
        'emotion_prob': [],
        'emotion_probs': [],
        'topics': [],
        'topic_prob': [],
        'answer': '',
        'previous_phase': [],
        'current_phase': '',
        'next_phase': [],
        'intent_turn_cnt': intent_turn_cnt
    }

    #previous_phase = ['/welcomemsg_chat', '/end_chat']
    #previous_phase = None

    while 1:
        sent = input('User: ')
        # wav_file = './exdata/' + sent + '.wav' # 수정
        wav_file = './exdata/test1.wav'
        result_dict = emotionchat.run(sent, wav_file, result_dict, turn_cnts)

        '''
        tokens = list(sent.split(' '))

        result_dict = {
            'input': list(tokens),
            'intent': '',
            'entity': [],
            'state': '',
            'emotion': '',
            'emotions': [],
            'emotion_prob': [],
            'topic': '',
            'topics': [],
            'topic_prob': [],
            'answer': None,
            'previous_phase': [],
            'current_phase': '',
            'next_phase': [],
            'intent_turn_cnt': intent_turn_cnt
        }
        '''

        conversation_history.append(result_dict)

        # turn_cnt 1 증가
        turn_cnt += 1

        print("**************\n\n")

        print("Bot : ")
        for i in range(len(result_dict['answer'])):
            print(result_dict['answer'][i])
        #print("Bot : " + str(result_dict['answer']))

        print("-" * 100 + "\n\n")


        '''
        print("\n\n*******시나리오 check*******")
        print("이전 인텐트 : " + str(conversation_history[turn_cnt-1]['intent']))
        print("현재 인텐트 : " + str(result_dict['intent']) + '\n') #
        print("이전 단계 : " + str(conversation_history[turn_cnt-1]['current_phase']))
        print("현재 단계 : " + str(result_dict['current_phase']))
        print("다음 예상 단계 : " + str(result_dict['next_phase']))
        print("pre_emotion: " + str(conversation_history[turn_cnt-1]['emotion']))
        print("현재_emotion: " + str(result_dict['emotion'])) #
        print("pre_emotions: " + str(result_dict['emotions'])) #
        print("pre_emotion_prob: " + str(result_dict['emotion_prob'])) #
        #print("현재_topic: " + str(result_dict['topic']))
        print("pre_topics: " + str(result_dict['topics']))
        print("pre_topic_prob: " + str(result_dict['topic_prob']))
        print('turn_cnt: ' + str(turn_cnt)) # run()함수로 넘어가는 turn_cnt
        '''



        '''
        pre_phase = result_dict['current_phase']
        pre_pred_phases = result_dict['next_phase']
        pre_intent = result_dict['intent']
        pre_emotion_prob = result_dict['emotion_prob']
        pre_topic_prob = result_dict['topic_prob']
        pre_emotion = result_dict['emotion']
        pre_emotions = result_dict['emotions']
        pre_topics = result_dict['topics']
        intent_turn_cnt = result_dict['intent_turn_cnt']

        pre_entity = result_dict['entity']
        pre_tokens = result_dict['input']
        '''

        # turn_cnt 종류 turn_cnts에 저장
        turn_cnts = {'turn_cnt': turn_cnt,
                     'intent_turn_cnt': intent_turn_cnt
                     }

        # 현재 단계명이 '/end_phase'이면 강제종료
        if result_dict['current_phase'] == '/end_phase':
            break


    # 최대 감정, 주제와 그에 대응하는 확률 출력
    final_emo_topic = final_emotion(result_dict)

    itemlist = final_emo_topic.items()

    print('\n' + '-'*100 + '-' * 100)
    print('<최종 딕셔너리에서 최대 감정, 주제와 그에 대응하는 확률 출력>')
    print(itemlist)
    print('-' * 100 + "-" * 100 + "\n\n")