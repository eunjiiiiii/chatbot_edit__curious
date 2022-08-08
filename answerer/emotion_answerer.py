from answerer.base_answerer import BaseAnswerer
import os
import numpy as np
import torch
from model.textgeneration import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer
import emotionchat_config as config
#from recommend_contents import recommendContents
import emotionchat_engine

class EmotionAnswerer(BaseAnswerer):
    """
    intent==감정호소 일 때
    답변 메세지 생성 클래스
    :return : 챗봇 답변 메세지
    """

    def __init__(self):
        #self.root_path = '/home/user/ittp'
        self.data_path = "./model/textgeneration/data/wellness_dialog_for_autoregressive_train.txt"
        self.checkpoint_path = "./model/textgeneration/model"
        self.save_ckpt_path = f"{self.checkpoint_path}/kogpt2-wellnesee-auto-regressive-0504_10.pth"

        ctx = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(ctx)

        # 저장한 Checkpoint 불러오기
        # self.checkpoint = torch.load(self.save_ckpt_path, map_location=self.device)
        # self.checkpoint = torch.load(self.save_ckpt_path, map_location='cpu')
        self.model = DialogKoGPT2()
        self.model.load_state_dict(torch.load(self.save_ckpt_path, map_location=self.device)['model_state_dict'])
        self.tokenizer = get_kogpt2_tokenizer()

        self.model.eval()

    def generate_answer_collection(self, emotion: str, pre_emotion: str, pre_emotions: list, max_emotion_prob: float, topic: str, max_topic_prob: float, text,
                                   turn_cnt: int, pre_emotion_prob: list) -> str:
        if max_emotion_prob < config.EMOTION['threshold'] and turn_cnt < 4:
            # 1. 감정이 명확히 분류되지 않은 경우 & turn 수 5회 미만

            # tokenizer = get_kogpt2_tokenizer()

            output_size = 200  # 출력하고자 하는 토큰 갯수
            # fill_slot = False

            # for i in range(5):
            sent = text  # ex) '요즘 기분이 우울한 느낌이에요'
            tokenized_indexs = self.tokenizer.encode(sent)

            input_ids = torch.tensor(
                [self.tokenizer.bos_token_id, ] + tokenized_indexs + [self.tokenizer.eos_token_id]).unsqueeze(0)
            # set top_k to 50
            # 답변 생성
            sample_output = self.model.generate(input_ids=input_ids)

            msg_decode = self.tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs) + 1:],
                                               skip_special_tokens=True)

            # 문장 자르기(문장 2개까지만 나오게)

            if '?' in msg_decode:
                # if msg_decode.index('?') < msg_decode.index('.'):
                # '?'가 제일 먼저 나올 경우
                msg = [str(msg_decode.split('?')[0] + '? ' + msg_decode.split('?')[1].split('.')[0] + '.')]
                # else:
                # msg = msg_decode.split('?')[0]
            elif '...!' in msg_decode:
                msg = [str(msg_decode.split('...!')[0] + msg_decode.split('...!')[1].split('.')[0] + '.')]

            else:
                if msg_decode.split('.')[0] != msg_decode.split('.')[1]:
                    msg = [str(msg_decode.split('.')[0] + '. ' + msg_decode.split('.')[1])]
                else:
                    msg = [str(msg_decode.split('.')[0] + '. ' + msg_decode.split('.')[2])]
            # self.turn_cnt += 1

        elif max_emotion_prob < config.EMOTION['threshold'] and turn_cnt >= 4:
            # 2. 감정이 명확히 분류되지 않은 경우 & turn 수 5회 이상
            # 감정의 종류 : 기쁨 분노 슬픔 놀람 불안 신뢰
            if emotion in ['기쁨', '신뢰']:
                # 긍정 메세지
                msg = config.ANSWER['default_error_end_p']
            elif emotion in ['분노', '슬픔', '불안']:
                # 부정 메세지
                msg = config.ANSWER['default_error_end_n']
            else:
                # 중립 메세지
                msg = config.ANSWER['default_error_end']

            # self.turn_cnt += 1

        elif max_emotion_prob > config.EMOTION['threshold'] and turn_cnt <= 1:
            # 3. 감정-주제가 명확히 분류되고 turn_cnt <= 1 인 경우
            # tokenizer = get_kogpt2_tokenizer()

            output_size = 200  # 출력하고자 하는 토큰 갯수
            # fill_slot = False

            # for i in range(5):
            sent = text  # ex) '요즘 기분이 우울한 느낌이에요'
            tokenized_indexs = self.tokenizer.encode(sent)

            input_ids = torch.tensor(
                [self.tokenizer.bos_token_id, ] + tokenized_indexs + [self.tokenizer.eos_token_id]).unsqueeze(0)
            # set top_k to 50
            # 답변 생성
            sample_output = self.model.generate(input_ids=input_ids)

            msg_decode = self.tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs) + 1:],
                                               skip_special_tokens=True)

            # 문장 자르기(문장 2개까지만 나오게)

            if '?' in msg_decode:
                # if msg_decode.index('?') < msg_decode.index('.'):
                # '?'가 제일 먼저 나올 경우
                msg = [str(msg_decode.split('?')[0] + '? ' + msg_decode.split('?')[1].split('.')[0] + '.')]
                # else:
                # msg = msg_decode.split('?')[0]
            elif '...!' in msg_decode:
                msg = [str(msg_decode.split('...!')[0] + msg_decode.split('...!')[1].split('.')[0] + '.')]

            else:
                if msg_decode.split('.')[0] != msg_decode.split('.')[1]:
                    msg = [str(msg_decode.split('.')[0] + '. ' + msg_decode.split('.')[1])]
                else:
                    msg = [str(msg_decode.split('.')[0] + '. ' + msg_decode.split('.')[2])]
        else:
            # 4. 감정이 명확히 분류되고 turn_cnt > 2 인 경우
            msg = config.ANSWER['default_contents']
            # 제가 기분 나아질 수 있게 컨텐츠 추천 해드려도 될까요?

            # 컨텐츠 추천 문구 추가
            # msg += recommendContents.recommend_contents(emotion, topic)

        return msg


    def generate_answer_under5(self, text: str) -> str:
        """
        DialogKoGPT2이용한 답변 생성 함수
        감정-주제도 명확히 안잡히면서 turn 수도 5회 전에
        :param text: human utterance
        :param emotion: 감정-주제 분류 모델의 return값(6가지 감정 + None(threshold 만족 X)
        :return: chatbot response
        """
        # tokenizer = get_kogpt2_tokenizer()

        output_size = 200  # 출력하고자 하는 토큰 갯수
        # fill_slot = False

        # for i in range(5):
        sent = text  # ex) '요즘 기분이 우울한 느낌이에요'
        tokenized_indexs = self.tokenizer.encode(sent)

        input_ids = torch.tensor(
            [self.tokenizer.bos_token_id, ] + tokenized_indexs + [self.tokenizer.eos_token_id]).unsqueeze(0)
        # set top_k to 50
        # 답변 생성
        sample_output = self.model.generate(input_ids=input_ids)

        msg_decode = self.tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs) + 1:],
                                           skip_special_tokens=True)
        print("(system msg) emotion_answerer > generate_answer_under5 함수 실행")

        # 문장 자르기(문장 2개까지만 나오게)

        if '?' in msg_decode:
            #if msg_decode.index('?') < msg_decode.index('.'):
            # '?'가 제일 먼저 나올 경우
            msg = [str(msg_decode.split('?')[0] + '? ' + msg_decode.split('?')[1].split('.')[0] + '.')]
            #else:
                #msg = msg_decode.split('?')[0]
        elif '...!' in msg_decode:
            msg = [str(msg_decode.split('...!')[0] + msg_decode.split('...!')[1].split('.')[0] + '.')]

        else:
            if msg_decode.split('.')[0] != msg_decode.split('.')[1]:
                msg = [str(msg_decode.split('.')[0] + '. ' + msg_decode.split('.')[1])]
            else:
                msg = [str(msg_decode.split('.')[0] + '. ' + msg_decode.split('.')[2])]

        return msg

    def generate_answer_over5(self, emotions: str) -> str:
        """
        default 감정 답변 출력 함수
        감정-주제도 명확히 안잡히면서 turn 수 5회 초과일 때
        :param text: human utterance
        :param emotion: 감정-주제 분류 모델의 return값(6가지 감정 + None(threshold 만족 X)
        :return: chatbot response
        """

        emotion = emotionchat_engine.most_freq(emotions)

        # 감정의 종류 : 기쁨 분노 슬픔 놀람 불안 신뢰
        if emotion in ['기쁨', '평온함']:
            # 긍정 메세지
            msg = config.ANSWER['default_error_end_p']
        elif emotion in ['분노', '슬픔', '불안']:
            # 부정 메세지
            msg = config.ANSWER['default_error_end_n']
        else:
            # 중립 메세지
            msg = config.ANSWER['goodbyemsg_chat']

        print("(system msg) emotion_answerer > generate_answer_over5 함수 실행")
        return msg

    def contents_answer(self, text: str, emotion: str, topic: str) -> str:
        """
        컨텐츠 추천 답변 함수
        감정-주제 명확 O turn 수 상관 X
        :param text: human utterance
        :param emotion: 감정-주제 분류 모델의 return값(6가지 감정 + None(threshold 만족 X)
        :return: chatbot response
        """
        msg = config.ANSWER['default_contents']
        # 제가 기분 나아질 수 있게 컨텐츠 추천 해드려도 될까요?

        # 컨텐츠 추천 문구 추가
        #msg += recommendContents.recommend_contents(emotion, topic)

        print("(system msg) emotion_answerer > contents_answer 함수 실행")
        return msg