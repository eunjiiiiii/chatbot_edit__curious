import os
import platform
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

root_dir = os.path.abspath(os.curdir)
# 만약 로딩이 안된다면 root_dir을 직접 적어주세요.
# 데모 기준에서 OS별 root path는 아래와 같이 적으면 됩니다.
# windows : C:Users/yourname/yourdirectory/emotionchat/demo
# linux : /home/yourname/yourdirectory/emotionchat/demo

_ = '\\' if platform.system() == 'Windows' else '/'
if root_dir[len(root_dir) - 1] != _: root_dir += _

human_name = 'DD'
bot_name = '마음결'

"""
    phase명 정리
    '/welcomemsg_chat': 인사,
    '/other_user': 넋두리,
    '/induce_ucs': 불궁감유도,
    '/recognize_uc_chat': 불궁대화인식,
    '/recognize_emotion_chat': 감정대화인식, #인텐트가 처음으로 '마음상태호소'일 때의 단계
    '/recognize_uc': (확실한) 불궁인식,
    '/generate_emotion_chat': 생성모델을 통한 챗봇 대화, # 인텐트가 '마음상태호소'인게 turn_cnt >=1(2번)인 경우
    '/recognize_emotion': (확실한) 감정인식,
     /recognize_topic: (확실한) 주제 인식,
    '/check_ucs': 확인용 재질의,
    '/check_ucs_positive': 확인용 재질의 긍정,
    '/check_ucs_negative':  확인용 재질의 부정,
    '/recommend_contents': (감정)컨텐츠제공,
    '/call_caregiver': (불편함)해결(간병인 호출),
    '/solve':  (궁금함)해결,
    '/end_chat':  작별인사
"""

BASE = {
    'root_dir': root_dir.format(_=_),  # 백엔드 루트경로
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'vector_size': 128,  # 단어 벡터 사이즈
    'batch_size': 512,  # 미니배치 사이즈
    'max_len': 8,  # 문장의 최대 길이 (패드 시퀀싱)
    'delimeter': _,  # OS에 따른 폴더 delimeter

    'PAD': 0,  # PAD 토큰 값 (전체가 0인 벡터)
    'OOV': 1  # OOV 토큰 값 (전체가 1인 벡터)
}

DATA = {
    'data_ratio': 0.8,  # 학습\\검증 데이터 비율
    'raw_data_dir': BASE['root_dir'] + "data{_}data{_}raw{_}".format(_=_),  # 원본 데이터 파일 경로
    'ood_data_dir': BASE['root_dir'] + "data{_}data{_}ood{_}".format(_=_),  # out of distribution 데이터셋
    'intent_data_dir': BASE['root_dir'] + "data{_}data{_}intent_data.csv".format(_=_),  # 생성된 인텐트 데이터 파일 경로
    'entity_data_dir': BASE['root_dir'] + "data{_}data{_}entity_data.csv".format(_=_),  # 생성된 엔티티 데이터 파일 경로

    'NER_categories': ['DATE', 'LOCATION', 'PLACE', 'RESTAURANT',
                       'BODY', 'SYMPTOM', 'FOOD',
                       'EMOTION', 'MAX_EMOTION_PROB', 'TOPIC', 'MAX_TOPIC_PROB', 'TEXT', 'TURN_CNT',
                       'PRE_EMOTION', 'PRE_EMOTIONS', 'PRE_EMOTION_PROB',
                       '신체부위', '증상', '장소', '음식'],  # 사용자 정의 태그
    'NER_tagging': ['B', 'E', 'I', 'S'],  # NER의 BEGIN, END, INSIDE, SINGLE 태그
    'NER_outside': 'O',  # NER의 O태그 (Outside를 의미)
}


API = {
    'request_chat_url_pattern': 'request_chat',  # request_chat 기능 url pattern
    'fill_slot_url_pattern': 'fill_slot',  # fill_slot 기능 url pattern
    'get_intent_url_pattern': 'get_intent',  # get_intent 기능 url pattern
    'get_entity_url_pattern': 'get_entity'  # get_entity 기능 url pattern
}

ANSWER = {
    # 고정된 만남 안내 메세지
    'welcomemsg_chat' : ['안녕하세요. 저는 {HUMAN_NAME}님의 \n심리 상담을 도와드릴 {BOT_NAME} 입니다.\n'.format(HUMAN_NAME=human_name, BOT_NAME=bot_name),
                        '상담 시작 전, 스피커 음량을 확인해 주세요.\n{BOT_NAME}이 문자와 소리 모두 제공합니다.\n',
                         '답변은 음성과 텍스트 모두 사용하실 수 있어요.\n',
                         '자 이제 심리상담을 시작할게요. \n오늘 기분이나 마음이 어떠세요?\n[말하기] 버튼을 터치하시면 말씀하실 수 있어요.\n'],
    # 간병인 호출
    'call_caregiver': ['간병인 불러드릴게요 \n'],
    'default_error_other_user': ['넋두리 사절\n'],
    'fallback': ["죄송해요. 제가 이해하지 못했어요.\n 다시 한 번 말씀해주실래요?\n"],
    'default_error': ['무슨 일 있으신가요?\n'],
    'default_error_uncomfort': ['다른 불편하신 점은 없으신가요?\n'],
    'default_error_curious': ['다른 궁금하신 점은 없으신가요?\n'],
    'default_error_ucs': ['그러시군요. 기분은 어떠세요? 아니면 다른 불편하거나 궁금하신 점이 있으신가요?\n'],
    'default_error_end_n': ['그러시군요. 또 기분이 안 좋아지면 언제든 저에게 이야기해주세요\n'],
    'default_error_end_p': ['그러시군요. 또 기분 좋은 일 생기시면 언제든 저에게 이야기 들려주세요\n'],
    # 'default_error_end': ['다음에 또 불러서 이야기 들려주세요.\n'],
    'default_contents': ['{HUMAN_NAME}님의 심리 상태를 이해했습니다\n마음을 다스릴 수 있는 좋은 글과 소리를\n제공해 드릴게요.\n'.format(HUMAN_NAME=human_name),
             '이거 보시고 마음이 좀 나아지셨으면 좋겠어요. \n이제 상담을 마무리 할 시간이네요.\n',
             '이만 작별인사 드릴게요. 다음에 또 이야기 들려주세요^^\n'],
    'default_check_emotion': ['그러시군요. 혹시 더 마음 쓰이는 일은 없으셨을까요?\n'],
    'goodbyemsg_uc': ['네. 다음에 또 불편한 점 있으시면 불러주세요. 좋은 하루 되세요^^\n'],
    'goodbyemsg_chat': ['네. 다음에 또 불러서 이야기 들려주세요. 좋은 하루 되세요^^\n'],
    'default_error_welcomemsg': ['반가워요. 오늘 기분은 어떠세요? 아니면 다른 불편하거나 궁금하신 점이 있으신가요?\n'],
    'default_error_emotion': ['그러시군요. 오늘 기분이 어떠신지 좀 더 구체적으로 말씀해주실 수 있으신가요?\n']
}

SORT_INTENT = {
    'QURIOUS': ['weather', 'dust', 'restaurant', 'travel'],
    'PHISICALDISCOMFORT' : ['기타활동요구', '욕구표출', '위생활동요구', '환경불편호소', '수면문제호소', '신체불편호소', '이동도움요구', '음식불편호소', '자세변경요구'],
    'PHSICALDISCOMFORTnQURIOUS': ['기타활동요구', '욕구표출', '위생활동요구', '환경불편호소', '수면문제호소', '신체불편호소', '이동도움요구', '음식불편호소', '자세변경요구',
                    'weather', 'dust', 'restaurant', 'travel', '궁금함'],
    'SENTIMENTDISCOMFORT': ['마음상태호소']
}

PHASE_INTENT = {
    '/welcomemsg_chat': [''],
    '/other_user': [''],
    '/induce_ucs': [''],
    '/recognize_uc_chat': [],
    '/recognize_emotion_chat': [],
    '/recognize_uc': [],
    '/generate_emotion_chat': [],
    '/recognize_emotion': [],
    '/check_ucs': [],
    '/check_ucs_positive': [],
    '/check_ucs_negative': [],
    '/recommend_contents': [],
    '/call_caregiver': [],
    '/solve': [],
    '/end_chat': []
}

# 해당 단계의 예상 단계를 config에 미리 저장해놓는게 나을지, 아님 이전 단계의 다음 예상 단계를 저장해놓는게 나을지
# 해당 단계의 예상 단계를 config에 미리 저장해놓자!

PRED_PHASE = {
    '/welcomemsg_chat': ['/other_user', '/recognize_uc_chat', '/recognize_emotion_chat', '/recognize_uc', '/recognize_emotion', '/recognize_topic', '/generate_emotion_chat', '/check_ucs',
                        '/fill_slot', '/end_phase'],  # ok
    '/other_user': ['/induce_ucs', '/recongnize_uc_chat', '/recongnize_emotion_chat',
                   '/recognize_uc', '/recognize_emotion', '/recognize_topic',
                    '/end_chat', '/generate_emotion_chat', '/recommend_contents', '/end_phase'],     # ok
    # '/induce_ucs': ['other_user', '/recognize_uc_chat', '/recognize_emotion_chat',
    #                          '/recognize_uc'],
    # '/recognize_uc_chat': ['/recognize_uc', '/fill_slot'],
    # '/recognize_emotion_chat': ['/generate_emotion_chat'],
    '/fill_slot': ['/fill_slot', '/recognize_uc', '/check_ucs','/check_ucs_positive', '/check_ucs_negative'],   # ok
    '/recognize_uc': ['/check_ucs', '/fill_slot', '/recognize_uc', '/check_ucs_positive', '/check_ucs_negative',
                      '/check_ucs', '/end_phase'],  # ok
    '/generate_emotion_chat': ['/generate_emotion_chat', '/end_chat', '/recognize_emotion_chat',
                            '/recommend_contents', '/end_phase'],   # ok
    # '/recognize_emotion': ['/check_ucs'],
    '/check_ucs': ['/check_ucs_positive', '/check_ucs_negative', '/check_ucs', '/end_phase'],   # ok
    '/check_ucs_positive': ['/end_chat'],   # ok
    '/check_ucs_negative': ['/end_chat'],   # ok
    # '/recommend_contents': ['/end_chat'],
    # '/call_caregiver': ['/end_chat'],
    # '/solve': ['/end_chat'],
    '/end_phase': ['/end_phase']   # ok
}

EMOTION = {
    'threshold': 0.75,
    '긍정': ['평온함',' 기쁨'],
    '부정': ['분노', '불안', '슬픔']
}

TOPIC = {
    'threshold': 0.25
}

INTENT = {
    'model_lr': 1e-4,  # 인텐트 학습시 사용되는 러닝레이트
    'loss_lr': 1e-2,  # 인텐트 학습시 사용되는 러닝레이트
    'weight_decay': 1e-4,  # 인텐트 학습시 사용되는 가중치 감쇠 정도
    'epochs': 300,  # 인텐트 학습 횟수
    'd_model': 512,  # 인텐트 모델의 차원
    'd_loss': 32,  # 인텐트 로스의 차원 (시각화차원, 높을수록 ood 디텍션이 정확해지지만 느려집니다.)
    'layers': 1,  # 인텐트 모델의 히든 레이어(층)의 수
    'grid_search': True,  # KNN과 Fallback Detector 학습시 그리드 서치 여부

    'lr_scheduler_factor': 0.75,  # 러닝레이트 스케줄러 감소율
    'lr_scheduler_patience': 10,  # 러닝레이트 스케줄러 감소 에폭
    'lr_scheduler_min_lr': 1e-12,  # 최소 러닝레이트
    'lr_scheduler_warm_up': 100,  # 러닝레이트 감소 시작시점

    # auto를 쓰려면 ood dataset을 함께 넣어줘야합니다.
    'distance_fallback_detection_criteria': 'auto',  # [auto, min, mean], auto는 OOD 데이터 있을때만 가능
    'distance_fallback_detection_threshold': -1,  # mean 혹은 min 선택시 임계값
    'softmax_fallback_detection_criteria': 'auto',  # [auto, other], auto는 OOD 데이터 있을때만 가능
    'softmax_fallback_detection_threshold': -1,  # other 선택시 fallback이 되지 않는 최소 값

    # 그리드 서치를 사용하지 않을때 KNN의 K값
    'num_neighbors': 10,

    # 그리드 서치를 사용할 때의 파라미터 목록
    'dist_param': {
        'n_neighbors': list(range(5, 15)),  # K값 범위 설정
        'weights': ["uniform"],  # ['uniform', 'distance']
        'p': [2],  # [1, 2] (맨하튼 vs 유클리디언)
        'algorithm': ['ball_tree']  # ['ball_tree', 'kd_tree']
    },

    # 폴백 디텍터 후보 (선형 모델을 추천합니다)
    'fallback_detectors': [
        LogisticRegression(max_iter=30000),
        LinearSVC(max_iter=30000)
    ]
}

PROC = {
    'logging_precision': 5,  # 결과 저장시 반올림 소수점 n번째에서 반올림
    'model_dir': BASE['root_dir'] + "saved{_}".format(_=_),  # 모델 파일, 시각화 자료 저장 경로
    'visualization_epoch': 50,  # 시각화 빈도 (애폭마다 시각화 수행)
    'save_epoch': 10  # 저장 빈도 (에폭마다 모델 저장)
}

GENSIM = {
    'window_size': 2,  # 임베딩 학습시 사용되는 윈도우 사이즈
    'workers': 8,  # 학습시 사용되는 쓰레드 워커 갯수
    'min_count': 2,  # 데이터에서 min count보다 많이 등장해야 단어로 인지
    'sg': 1,  # 0 : CBOW = 1 \\ SkipGram = 2
    'iter': 2000  # 임베딩 학습 횟수
}

LOSS = {
    'center_factor': 0.025,  # Center Loss의 weighting 비율
    'coco_alpha': 6.25,  # COCO loss의 alpha 값
    'cosface_s': 7.00,  # Cosface의 s값 (x^T dot W를 cos형식으로 바꿀 때 norm(||x||))
    'cosface_m': 0.25,  # Cosface의 m값 (Cosface의 마진)
    'gaussian_mixture_factor': 0.1,  # Gaussian Mixture Loss의 weighting 비율
    'gaussian_mixture_alpha': 0.00,  # Gaussian Mixture Loss의 alpha 값
}

ENTITY = {
    'model_lr': 1e-4,  # 엔티티 학습시 사용되는 모델 러닝레이트
    'loss_lr': 1e-4,  # 엔티티 학습시 사용되는 로스 러닝레이트 (아직 사용되지 않음)
    'weight_decay': 1e-4,  # 엔티티 학습시 사용되는 가중치 감쇠 정도
    'epochs': 1000,  # 엔티티 학습 횟수
    'd_model': 512,  # 엔티티 모델의 차원
    'layers': 1,  # 엔티티 모델의 히든 레이어(층)의 수
    'masking': True,  # loss 계산시 패딩 마스크 여부

    'lr_scheduler_factor': 0.75,  # 러닝레이트 스케줄러 감소율
    'lr_scheduler_patience': 10,  # 러닝레이트 스케줄러 감소 에폭
    'lr_scheduler_min_lr': 1e-12,  # 최소 러닝레이트
    'lr_scheduler_warm_up': 100  # 러닝레이트 감소 시작시점
}