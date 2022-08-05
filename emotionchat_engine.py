from scenarios.scenario_manager import ScenarioManager
import emotionchat_config as config
from model.emotion.predict import IAI_EMOTION
from model.topic.predict import IAI_TOPIC
from model.intent_entity.intent_entity import JointIntEnt
from model.textgeneration.predict import DialogKoGPT2
from scenarios.default_scenario import dust, weather, physicalDiscomfort, sleepProblem, moveHelp, changePosture, \
    higieneAct, otherAct, environmentalDiscomfort, expressDesire, foodDiscomfort, sentimentDiscomfort
from answerer.emotion_answerer import EmotionAnswerer
from data.dataset import Dataset
from model import curious_intent
from model.proc import DistanceClassifier
from model.loss import CenterLoss
from scenarios.scenario import Scenario

class EmotionChat:

    def __init__(self,
                 embed_processor,
                 intent_classifier,
                 entity_recognizer):

        """
        emotionChat 답변생성 클래스 입니다.

        :param embed_processor: 궁금함용 임베딩 프로세서 객체 or (, 학습여부)
        :param intent_classifier: 궁금함용 인텐트 분류기 객체 or (, 학습여부)
        :param entity_recognizer: 궁금함용 엔티티 인식기 객체 or (, 학습여부)
        """

        '''
        self.pre_result_dict =
        self.pre_tokens = []
        self.pre_phase = ''
        self.pre_pred_phases = ''
        self.pre_intent = ''
        self.pre_emotion_prob = 0.0
        self.pre_topic_prob = 0.0
        self.pre_emotion = ''
        self.pre_emotions = []
        self.pre_topics = []
        self.intent_turn_cnt = 0
        self.pre_entity = []
        '''

        self.dataset = Dataset(ood=True)
        self.scenario_manager = ScenarioManager()
        self.intent_entity_classifier = JointIntEnt("./model/intent_entity/jointbert_demo_model", no_cuda=True)
        self.emotion_recognizer = IAI_EMOTION
        self.topic_recognizer = IAI_TOPIC("./model/topic/model", no_cuda=True)
        self.response_generator = DialogKoGPT2()
        self.curious_intent_classifier = DistanceClassifier(model=curious_intent.CNN(self.dataset.intent_dict), loss=CenterLoss(self.dataset.intent_dict),)

        self.scenarios = [weather, dust,
                          physicalDiscomfort, sleepProblem, moveHelp,
                          changePosture, higieneAct, otherAct,
                          environmentalDiscomfort, expressDesire, foodDiscomfort, sentimentDiscomfort]

        self.embed_processor = embed_processor[0] \
            if isinstance(embed_processor, tuple) \
            else embed_processor

        self.intent_classifier = intent_classifier[0] \
            if isinstance(intent_classifier, tuple) \
            else intent_classifier

        self.entity_recognizer = entity_recognizer[0] \
            if isinstance(entity_recognizer, tuple) \
            else entity_recognizer

        if isinstance(embed_processor, tuple) \
                and len(embed_processor) == 2 and embed_processor[1] is True:
            self.__fit_embed()

        if isinstance(intent_classifier, tuple) \
                and len(intent_classifier) == 2 and intent_classifier[1] is True:
            self.__fit_intent()

        if isinstance(entity_recognizer, tuple) \
                and len(entity_recognizer) == 2 and entity_recognizer[1] is True:
            self.__fit_entity()


        for scenario in self.scenarios:
            self.scenario_manager.add_scenario(scenario)

    def run(self, text: str, wav_file, pre_result_dict: dict, turn_cnts: dict) -> dict:
        """
        인텐트 인식 후 단계 확인 후 시나리오에 적용해주는 함수
        모든 사용자 발화 text는 이 함수를 먼저 거쳐야 함.
        :param text: 사용자 input text
        :param wav_file: 사용자 음성 파일(.wav)
        :param pre_result_dict: 이전 단계 result_dict
        :param turn_cnts: 이전까지 turn_cnt, intent_turn_cnt 를 value로 가지는 딕셔너리
        :return: result_dict 딕셔너리
        """

        # 0. 이전 result_dict value들을 클래스 인스턴스 변수에 저장
        pre_tokens = pre_result_dict['input']  # 이전 input들의 누적된 token 리스트
        pre_phase = pre_result_dict['current_phase']   # 문자열 형태의 이전 단계
        pre_pred_phases = pre_result_dict['next_phase']    # 이전 단계에서 예상한 다음 단계 리스트
        pre_intent = pre_result_dict['intent'] # 이전 단계 인텐트
        pre_emotion_prob = pre_result_dict['emotion_prob'] # 이전 단계까지의 누적 감정 확률 리스트
        pre_topic_prob = pre_result_dict['topic_prob'] # 이전 단계까지의 누적 주제 확률 리스트
        pre_emotion = pre_result_dict['emotion']   # 문자열 형태의 이전 단계까지의 확실한 감정
        pre_emotions = pre_result_dict['emotions'] # 이전 단계까지의 누적 감정 리스트
        pre_topics = pre_result_dict['topics'] # 이전 단계까지의 누적 주제 리스트
        intent_turn_cnt = pre_result_dict['intent_turn_cnt']   # 이전 단계까지 이전 단계와 같은 인텐트 턴 횟수
        pre_entity = pre_result_dict['entity'] # 이전 entity 누적 리스트
        turn_cnt = turn_cnts['turn_cnt']



        # 1. 불편함/궁금함 인식 ,감정인식/주제인식 일 경우 intent 인식하지 않음
        c_ucs = True    # 이전 단계에서 불,궁,감 대화에 들어왔는가?
        if pre_phase == '' and pre_intent not in (
                config.SORT_INTENT['PHISICALDISCOMFORTnQURIOUS'] + config.SORT_INTENT['SENTIMENTDISCOMFORT']):
            # 이전 단계가 불편함, 마음상태호소, 궁금함 X -> 인텐트 인식
            intent, entity_ = self.intent_entity_classifier(text)
        elif '/check_ucs' in pre_pred_phases:
            # 이전 단계의 예상 단계에 /check_ucs (재질의) 가 있을 경우 = 현재 예상 단계가 재질의일 경우
            intent, entity_ = self.intent_entity_classifier(text)
            # c_ucs = False  # 이미 인식했기 때문. 재질의 필요 x => c_ucs == False
            c_ucs = False
        elif pre_phase == '':
            # 만남인사
            intent, _ = self.intent_entity_classifier(text)
            c_ucs = False
        else:
            # 이전 단계가 불편함, 마음상태호소, 궁금함 O -> 인텐트 인식 X
            _, entity_ = self.intent_entity_classifier(text)
            intent = pre_intent
            # 인텐트 인식 안해서 정확한 오류 확인할 수 없는 단점은 엔티티가 안채워진걸로 판단하면 됨
            # 감정은 감정확률이나 emotion의 유무


        # 2. intent_turn_cnt 기록

        # 이전 대화의 intent와 현재 대화의 intent가 같으면 intent_turn_cnt 기록
        if pre_intent == intent:

            intent_turn_cnt += 1

        elif pre_intent != intent:
            # 이전 대화의 intent와 현재 대화의 intent가 다르면 intent_turn_cnt 초기화
            intent_turn_cnt = 0

        print("*-" * 50 + "\n" + "(system msg) emotionchat_engine> intent_turn_cnt : " +
              str(intent_turn_cnt)  + "\n" + "*-" * 50)


        # 3. inputs -> tokens
        tokens = list(text.split(' '))


        # 4. 만남인사, 작별인사 처리

        if (("안녕" in text) or (intent == '만남인사')) and pre_phase == '':
            # 첫번째 turn이고, 만남인사일 경우
            return {
                'input': tokens + pre_tokens,
                'intent': '만남인사',
                'entity': [],
                'state': 'SUCCESS',
                'emotion': '',
                'emotions': pre_emotions,
                'emotion_prob': pre_emotion_prob,
                'emotion_probs': [1., 1., 1., 1., 1., 1.],
                #'topic': '',
                'topics': pre_topics,
                'topic_prob': pre_topic_prob,
                'answer': config.ANSWER['welcomemsg_chat'],
                'previous_phase': '',
                'current_phase': '/welcomemsg_chat',
                'next_phase': ['/other_user', '/recognize_uc_chat', '/recognize_emotion_chat', '/recognize_uc',
                               '/recognize_emotion', '/recognize_topic', '/generate_emotion_chat', '/check_ucs',
                               '/fill_slot', '/end_phase'],
                'intent_turn_cnt': intent_turn_cnt
            }

        elif intent == '작별인사' or "잘있어" in text or "다음에" in text or "잘가" in text:
            # 작별인사일 경우
            print("(system msg) 작별인사")
            return {
                'input': tokens + pre_tokens,
                'intent': '',
                'entity': [],
                'state': 'SUCCESS',
                'emotion': '',
                'emotions': pre_emotions,
                'emotion_prob': pre_emotion_prob,
                'emotion_probs': [1., 1., 1., 1., 1., 1.],
                #'topic': '',
                'topics': pre_topics,
                'topic_prob': pre_topic_prob,
                'answer': config.ANSWER['goodbyemsg_chat'],
                'previous_phase': ['check_ucs', '/recommend_contents'],
                'current_phase': '/end_phase',
                'next_phase': ['/end_phase'],
                'intent_turn_cnt': intent_turn_cnt
            }

        ## 현재 인텐트 출력
        print("(system msg) 현재 인텐트 :" + intent)


        '''
        시나리오에 적용
        '''


        # 5. 엔티티 앞에 I-, B- 제외

        entity = self._edit_entity(entity_)
        print("(system msg) 엔티티 : " + str(entity))

        # tokens(토큰화 된 입력 text)
        print("(system msg) tokens : " + str(tokens))
        # print("(system msg) tokens : " + str(tokens))
        print("(system msg) tokens + 이전 텍스트 : " + str(tokens + pre_tokens))


        # 6. 감정, 주제 라벨 & 확률값 받기
        if intent not in ['만남인사','작별인사']:
            emotion_label, emotion_probs_array, max_emotion_prob_array = self.emotion_recognizer().predict(text,
                                                                                                           wav_file)
            print('(system msg) emotion_probs_array: ' + str(emotion_probs_array[0]))
            print('(system msg) max_emotion: ' + str(max_emotion_prob_array))

            max_emotion_prob = float(max_emotion_prob_array)
            topic_label, topic_probs_array, max_topic_prob_array = self.topic_recognizer.predict(text)
            print('(system msg) topic_probs_array: ' + str(topic_probs_array[0]))
            print('(system msg) max_topic: ' + str(max_topic_prob_array))
            max_topic_prob = float(max_topic_prob_array)

            emotion, topic = self.__rename_emotion_topic(emotion_label, topic_label)
            print(
                "(system msg) 현재 감정 확률 : " + str(emotion) + str(max_emotion_prob) + " ,현재 주제 확률 : " + str(topic) + str(
                    max_topic_prob))

        ## scenario_manager() > apply_scenario()로 보내기 위해 result_dict로 변수 묶어놓음
        if intent in ['만남인사','작별인사']:
            result_dict = {
                'input': tokens,
                'intent': intent,
                'entity': entity,
                'state': 'EMOTIONCHAT_ENGINE',
                'emotion': '',
                'emotions': [],
                'emotion_prob': [],
                'emotion_probs': [1., 1., 1., 1., 1., 1.],
                'topics': [],
                'topic_prob': [],
                'answer': '',
                'previous_phase': pre_phase,
                'current_phase': '',
                'next_phase': [],
                'intent_turn_cnt': intent_turn_cnt
            }
        else:
            result_dict = {
                        'input': tokens,
                        'intent': intent,
                        'entity': entity,
                        'state': 'EMOTIONCHAT_ENGINE',
                        'emotion': '',
                        'emotions': [emotion],
                        'emotion_prob': [max_emotion_prob],
                        'emotion_probs': emotion_probs_array[0],
                        'topics': [topic],
                        'topic_prob': [max_topic_prob],
                        'answer': '',
                        'previous_phase': pre_phase,
                        'current_phase': '',
                        'next_phase': [],
                        'intent_turn_cnt': intent_turn_cnt
                    }

        # 7. scenario_manager의 apply_scenario() 실행
        result_dict = self.scenario_manager.apply_scenario(pre_result_dict, result_dict,
                                                               text, c_ucs, turn_cnt)

        # 8. 단계 check & 단계 오류 해결

        if self.__check_phase(pre_pred_phases, result_dict['current_phase']):
            # return self.scenario_manager.apply_scenario(intent, entity, tokens, emotion, topic, intent_turn_cnt)
            return result_dict

        else:
            # 단계 오류 (예상한 단계가 아닐 경우)
            return self.__handle_phase_error(turn_cnt, pre_result_dict, result_dict, text)

        '''
        # 예상한 단계가 아닐 경우에 error 메세지 출력
        else:
            print('*-' * 30)
            print('(system msg) 대화 단계 오류')
            print('(system msg) 이전 단계 : ' + str(pre_phase))
            print('(system msg) 현재 단계 : ' + result_dict['current_phase'])
            print("(system msg) intent_turn_cnt : " + str(intent_turn_cnt))
            print('*-' * 30)

            if turn_cnt > 5:
                # 전체 turn 횟수가 6회가 넘으면 종료
                return {
                    'input': tokens + pre_tokens,
                    'intent': '',
                    'entity': [],
                    'state': 'SUCCESS',
                    'emotion': '',
                    'emotions': [emotion] + pre_emotions,
                    'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                    #'topic': '',
                    'topics': [topic] + pre_topics,
                    'topic_prob': [max_topic_prob] + pre_topic_prob,
                    'answer': config.ANSWER['goodbyemsg_chat'],
                    'previous_phase': ['check_ucs', '/recommend_contents'],
                    'current_phase': '/end_phase',
                    'next_phase': ['/end_phase'],
                    'intent_turn_cnt': intent_turn_cnt
                }

            else:
                # 전체 turn 횟수가 6회가 넘지 않았을 경우
                if pre_intent == '만남인사' and intent_turn_cnt <= 1:
                    # 만남인사 하고 딴소리 하는 경우 -> 1번까지만 봐줌
                    return {
                        'input': tokens + pre_tokens,
                        'intent': intent,
                        'entity': entity,
                        'state': 'SUCCESS',
                        'emotion': '',
                        'emotions': pre_emotions,
                        'emotion_prob': pre_emotion_prob,
                        #'topic': topic,
                        'topics': pre_topics,
                        'topic_prob': pre_topic_prob,
                        'answer': config.ANSWER['default_error_welcomemsg'],
                        'previous_phase': pre_phase,
                        'current_phase': result_dict['current_phase'],
                        'next_phase': pre_pred_phases,
                        'intent_turn_cnt': intent_turn_cnt
                    }

                elif intent == '작별인사':
                    # 원래 작별인사 타이밍이 아닌데 작별인사 하는 경우
                    """
                    if emo_prob_chk == True:
                        if scenarios['EMOTION'] in ['분노', '슬픔', '불안']:
                            return{
    
                            }
                        return{
    
                        }
                    """

                    return {
                        'input': tokens + pre_tokens,
                        'intent': '',
                        'entity': [],
                        'state': 'SUCCESS',
                        'emotion': '',
                        'emotions': pre_emotions,
                        'emotion_prob': pre_emotion_prob,
                        #'topic': '',
                        'topics': pre_topics,
                        'topic_prob': pre_topic_prob,
                        'answer': config.ANSWER['goodbyemsg_chat'],
                        'previous_phase': ['check_ucs', '/recommend_contents'],
                        'current_phase': '/end_phase',
                        'next_phase': ['/end_phase'],
                        'intent_turn_cnt': intent_turn_cnt
                    }

                elif (pre_intent in ['마음상태호소', '부정', '긍정'] or intent in ['마음상태호소','부정','긍정']) \
                        and max_emotion_prob < config.EMOTION['threshold']:
                    # 대화 오류인데/ 이전까지의 대화가 감정대화이고, 현재 들어온 대화의 감정 확률이 threshold를 넘지 않았을 경우
                    if pre_emotion == '':
                        # 이전에 확실한 감정이 없었을 경우
                        return {
                            'input': tokens + pre_tokens,
                            'intent': '마음상태호소',
                            'entity': entity,
                            'state': 'REQUIRE_EMOTION',
                            'emotion': '',
                            'emotions': [emotion] + pre_emotions,
                            'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                            #'topic': '',
                            'topics': [topic] + pre_topics,
                            'topic_prob': [max_topic_prob] + pre_topic_prob,
                            'answer': EmotionAnswerer().generate_answer_under5(text),
                            'previous_phase': ['/recognize_emotion_chat', '/other_user'],
                            'current_phase': '/generate_emotion_chat',
                            'next_phase': ['/generate_emotion_chat', '/end_chat', '/recognize_emotion_chat',
                                           '/recommend_contents', '/end_phase'],
                            'intent_turn_cnt': intent_turn_cnt
                        }

                    else:
                        # 이전에 확실한 감정이 있었던 경우
                        return {
                            'input': tokens + pre_tokens,
                            'intent': '마음상태호소',
                            'entity': entity,
                            'state': 'REQUIRE_EMOTION',
                            'emotion': pre_emotion,
                            'emotions': [emotion] + pre_emotions,
                            'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                            #'topic': topic,
                            'topics': [topic] + pre_topics,
                            'topic_prob': [max_topic_prob] + pre_topic_prob,
                            'answer': config.ANSWER['default_error_emotion'],  # 수정 봐야 함!!!! (07/06)
                            'previous_phase': ['/recognize_emotion_chat', '/other_user'],
                            'current_phase': '/generate_emotion_chat',
                            'next_phase': ['/generate_emotion_chat', '/end_chat', '/recognize_emotion_chat',
                                           '/recommend_contents', '/end_phase'],
                            'intent_turn_cnt': intent_turn_cnt
                        }

                elif pre_phase != '/check_ucs' and intent in ['부정', '긍정', '만남인사', '욕구표출']:
                    # 이전 단계가 불궁을 인식한 단계가 아님에도 부정, 긍정, 만남인사, 욕구표출인텐트가 나오는 경우
                    if '/end_chat' in pre_pred_phases:
                        return {
                            'input': tokens + pre_tokens,
                            'intent': 'UNK',
                            'entity': entity,
                            'state': 'SUCCESS',
                            'emotion': '',
                            'emotions': [emotion] + pre_emotions,
                            'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                            #'topic': topic,
                            'topics': [topic] + pre_topics,
                            'topic_prob': [max_topic_prob] + pre_topic_prob,
                            'answer': config.ANSWER['default_error_end'],
                            'previous_phase': ['check_ucs', '/other_use', '/recommend_contents', '/end_phase'],
                            'current_phase': '/end_phase',
                            'next_phase': ['/end_phase'],
                            'intent_turn_cnt': intent_turn_cnt
                        }

                    else:
                        return {
                            'input': tokens + pre_tokens,
                            'intent': 'UNK',
                            'entity': entity,
                            'state': 'SUCCESS',
                            'emotion': '',
                            'emotions': [emotion] + pre_emotions,
                            'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                            #'topic': topic,
                            'topics': [topic] + pre_topics,
                            'topic_prob': [max_topic_prob] + pre_topic_prob,
                            'answer': config.ANSWER['default_error'],
                            'previous_phase': ['/welcomemsg_chat', '/other_user'],
                            'current_phase': '/other_user',
                            'next_phase': ['/induce_ucs', '/recongnize_uc_chat', '/recongnize_emotion_chat',
                                           '/recognize_uc', '/recognize_emotion', '/recognize_topic',
                                           '/end_chat', '/generate_emotion_chat', '/recommend_contents', '/end_phase'],
                            'intent_turn_cnt': intent_turn_cnt
                        }

                elif pre_intent == '마음상태호소' and max_emotion_prob > config.EMOTION['threshold']:
                    # 이전 대화가 감정대화 & 현재 감정 확률이 threshold를 넘었을 경우
                    return {
                        'input': tokens + pre_tokens,
                        'intent': intent,
                        'entity': [],
                        'state': 'SUCCESS',
                        'emotion': emotion,
                        'emotions': [emotion] + pre_emotions,
                        'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                        #'topic': topic,
                        'topics': [topic] + pre_topics,
                        'topic_prob': [max_topic_prob] + pre_topic_prob,
                        'answer': EmotionAnswerer().generate_answer_under5(text),
                        'previous_phase': ['/recognize_emotion_chat'],
                        'current_phase': '/generate_emotion_chat',
                        'next_phase': ['/generate_emotion_chat', '/recognize_emotion_chat', '/recommend_contents',
                                       '/end_phase'],
                        'intent_turn_cnt': intent_turn_cnt
                    }

                else:
                    # 그 외에 에러 처리(인텐트 인식 잘못 했을 경우)
                    return {
                        'input': tokens + pre_tokens,
                        'intent': intent,
                        'entity': entity,
                        'state': 'SUCCESS',
                        'emotion': '',
                        'emotions': [emotion] + pre_emotions,
                        'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                        #'topic': topic,
                        'topics': [topic] + pre_topics,
                        'topic_prob': [max_topic_prob] + pre_topic_prob,
                        'answer': config.ANSWER['default_error_ucs'],
                        'previous_phase': pre_phase,
                        'current_phase': result_dict['current_phase'],
                        'next_phase': pre_pred_phases,
                        'intent_turn_cnt': intent_turn_cnt
                    }

            """

            (07/18) 잘못 알아들었을 경우 fall back error 메세지 출력하는 코드 작성

            """
            '''

    def _edit_entity(self, entity: list) -> list:
        """
        한글로 된 엔티티를 영어로 바꿔주는 함수
        default_scenario의 entity key를 영어로 써놨기 때문에 변경 필수
        :param entity: 엔티티(한글)
        :return: 엔티티(영어)
        """
        result_entity = []
        for e in entity:
            if '-' in e:
                result = e.split('-')[1]
                if result == '신체부위':
                    e = 'body'
                elif result == '증상':
                    e = 'symptom'
                elif result == '장소':
                    e = 'place'
                elif result == '음식':
                    e = 'food'
            result_entity.append(e)

        return result_entity

    def __rename_emotion_topic(self, emotion_label: int, topic_label: int) -> str:
        """
        감정, 주제 라벨 값(int)을 문자열(한글) 바꿔주는 함수
        :param emotion_label: 감정 라벨 값(int)
        :param topic_label: 주제 라벨 값(int)
        :return: 감정, 주제 라벨 값(str)
        """
        if emotion_label == 0:
            emotion = '평온함'
        elif emotion_label == 1:
            emotion = '분노'
        elif emotion_label == 2:
            emotion = '놀람'
        elif emotion_label == 3:
            emotion = '기쁨'
        elif emotion_label == 4:
            emotion = '불안'
        elif emotion_label == 5:
            emotion = '슬픔'

        if topic_label == 0:
            topic = '건강'
        elif topic_label == 1:
            topic = '사회환경'
        elif topic_label == 2:
            topic = '정서인지'
        elif topic_label == 3:
            topic = '경제'

        return emotion, topic

    def __check_phase(self, pre_pred_phases: list, current_phase: str):
        """
        단계 체크 함수
        이전 단계에서 예상한 단계에 현재 단계가 포함되는지
        :param pre_pred_phases: 이전 단계에서 예상한 단계(list)
        :param current_phase: 현재 단계(str)
        :return: bool(True, False)
        """
        if current_phase in pre_pred_phases:
            return True
        return False

        # 다음 단계가 종료면 서버측에서 종료

    def __handle_phase_error(self, turn_cnt: int, pre_result_dict: dict, result_dict: dict, text: str) -> dict:
        """
        단계 오류 처리하는 함수
        :return: 다 채운 dictionary
        """

        intent_turn_cnt = result_dict['intent_turn_cnt'] + 1
        if len(result_dict['input']) != intent_turn_cnt:
            result_dict['input'] = result_dict['input'] + pre_result_dict['input']
        result_dict['emotions'] = result_dict['emotions'] + pre_result_dict['emotions']
        result_dict['emotion_prob'] = result_dict['emotion_prob'] + pre_result_dict['emotion_prob']
        result_dict['topics'] = result_dict['topics'] + pre_result_dict['topics']
        result_dict['topic_prob'] = result_dict['topic_prob'] + pre_result_dict['topic_prob']

        if turn_cnt > 5:
            # 전체 turn 횟수가 6회가 넘으면 종료

            result_dict['emotion'] = pre_result_dict['emotion']
            result_dict['state'] = 'OVER_TURN_5'
            result_dict['answer'] = config.ANSWER['goodbyemsg_chat']
            result_dict['previous_phase'] = pre_result_dict['current_phase']
            result_dict['current_phase'] = '/end_phase'
            result_dict['next_phase'] = ['/end_phase']

            '''
            return {
                'input': tokens + pre_tokens,
                'intent': '',
                'entity': [],
                'state': 'SUCCESS',
                'emotion': '',
                'emotions': [emotion] + pre_emotions,
                'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                # 'topic': '',
                'topics': [topic] + pre_topics,
                'topic_prob': [max_topic_prob] + pre_topic_prob,
                'answer': config.ANSWER['goodbyemsg_chat'],
                'previous_phase': ['check_ucs', '/recommend_contents'],
                'current_phase': '/end_phase',
                'next_phase': ['/end_phase'],
                'intent_turn_cnt': intent_turn_cnt
            }
            '''

        else:
            # 전체 turn 횟수가 6회가 넘지 않았을 경우

            if pre_result_dict['intent'] == '만남인사' and result_dict['intent_turn_cnt'] <= 1:
                # 만남인사 하고 딴소리 하는 경우 -> 1번까지만 봐줌

                print("만남인사 대화 오류 들어옴")

                #return result_dict

                '''
                return {
                    'input': tokens + pre_tokens,
                    'intent': intent,
                    'entity': entity,
                    'state': 'SUCCESS',
                    'emotion': '',
                    'emotions': pre_emotions,
                    'emotion_prob': pre_emotion_prob,
                    # 'topic': topic,
                    'topics': pre_topics,
                    'topic_prob': pre_topic_prob,
                    'answer': config.ANSWER['default_error_welcomemsg'],
                    'previous_phase': pre_phase,
                    'current_phase': result_dict['current_phase'],
                    'next_phase': pre_pred_phases,
                    'intent_turn_cnt': intent_turn_cnt
                }
                '''
            elif result_dict['intent'] == '작별인사':
                # 원래 작별인사 타이밍이 아닌데 작별인사 하는 경우
                """
                if emo_prob_chk == True:
                    if scenarios['EMOTION'] in ['분노', '슬픔', '불안']:
                        return{

                        }
                    return{

                    }
                """

                result_dict['emotion'] = pre_result_dict['emotion']
                result_dict['state'] = 'SUDDEN_GOODBYE'
                result_dict['answer'] = config.ANSWER['goodbyemsg_chat']
                result_dict['previous_phase'] = pre_result_dict['current_phase']
                result_dict['current_phase'] = '/end_phase'
                result_dict['next_phase'] = ['/end_phase']

                '''
                return {
                    'input': tokens + pre_tokens,
                    'intent': '',
                    'entity': [],
                    'state': 'SUCCESS',
                    'emotion': '',
                    'emotions': pre_emotions,
                    'emotion_prob': pre_emotion_prob,
                    # 'topic': '',
                    'topics': pre_topics,
                    'topic_prob': pre_topic_prob,
                    'answer': config.ANSWER['goodbyemsg_chat'],
                    'previous_phase': ['check_ucs', '/recommend_contents'],
                    'current_phase': '/end_phase',
                    'next_phase': ['/end_phase'],
                    'intent_turn_cnt': intent_turn_cnt
                }
                '''

            elif (pre_result_dict['intent'] in ['마음상태호소', '부정', '긍정'] or
                  result_dict['intent'] in ['마음상태호소', '부정', '긍정']) \
                    and result_dict['emotion_prob'] < config.EMOTION['threshold']:
                # 대화 오류인데/ 이전까지의 대화가 감정대화이고, 현재 들어온 대화의 감정 확률이 threshold를 넘지 않았을 경우
                if pre_result_dict['emotion'] == '':
                    # 이전에 확실한 감정이 없었을 경우

                    result_dict['intent'] = '마음상태호소'
                    result_dict['emotion'] = pre_result_dict['emotion']
                    result_dict['state'] = 'REQUIRE_EMOTION'
                    result_dict['answer'] = EmotionAnswerer().generate_answer_under5(text)
                    result_dict['previous_phase'] = pre_result_dict['current_phase']
                    result_dict['current_phase'] = '/generate_emotion_chat'
                    result_dict['next_phase'] = ['/generate_emotion_chat', '/end_chat', '/recognize_emotion_chat',
                                                 '/recommend_contents', '/end_phase']

                    '''
                    return {
                        'input': tokens + pre_tokens,
                        'intent': '마음상태호소',
                        'entity': entity,
                        'state': 'REQUIRE_EMOTION',
                        'emotion': '',
                        'emotions': [emotion] + pre_emotions,
                        'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                        # 'topic': '',
                        'topics': [topic] + pre_topics,
                        'topic_prob': [max_topic_prob] + pre_topic_prob,
                        'answer': EmotionAnswerer().generate_answer_under5(text,
                                                                            result_dict['emotion'],
                                                                            result_dict['topics'][0])
,
                        'previous_phase': ['/recognize_emotion_chat', '/other_user'],
                        'current_phase': '/generate_emotion_chat',
                        'next_phase': ['/generate_emotion_chat', '/end_chat', '/recognize_emotion_chat',
                                       '/recommend_contents', '/end_phase'],
                        'intent_turn_cnt': intent_turn_cnt
                    }
                    '''

                else:
                    # 이전에 확실한 감정이 있었던 경우

                    result_dict['emotion'] = pre_result_dict['emotion']
                    result_dict['state'] = 'REQUIRE_CERTAIN_EMOTION'
                    result_dict['answer'] = EmotionAnswerer().generate_answer_under5(text,
                                                                                     result_dict['emotion'],
                                                                                     result_dict['topics'][0])
                    result_dict['previous_phase'] = pre_result_dict['current_phase']
                    result_dict['current_phase'] = '/generate_emotion_chat'
                    result_dict['next_phase'] = ['/generate_emotion_chat', '/end_chat', '/recognize_emotion_chat',
                                                 '/recommend_contents', '/end_phase']

                    '''
                    return {
                        'input': tokens + pre_tokens,
                        'intent': '마음상태호소',
                        'entity': entity,
                        'state': 'REQUIRE_EMOTION',
                        'emotion': pre_emotion,
                        'emotions': [emotion] + pre_emotions,
                        'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                        # 'topic': topic,
                        'topics': [topic] + pre_topics,
                        'topic_prob': [max_topic_prob] + pre_topic_prob,
                        'answer': config.ANSWER['default_error_emotion'],  # 수정 봐야 함!!!! (07/06)
                        'previous_phase': ['/recognize_emotion_chat', '/other_user'],
                        'current_phase': '/generate_emotion_chat',
                        'next_phase': ['/generate_emotion_chat', '/end_chat', '/recognize_emotion_chat',
                                       '/recommend_contents', '/end_phase'],
                        'intent_turn_cnt': intent_turn_cnt
                    }
                    '''

            elif pre_result_dict['current_phase'] != '/check_ucs' and \
                    result_dict['intent'] in ['부정', '긍정', '만남인사', '욕구표출']:
                # 이전 단계가 불궁을 인식한 단계가 아님에도 부정, 긍정, 만남인사, 욕구표출인텐트가 나오는 경우
                if '/end_chat' in pre_result_dict['next_phase']:

                    result_dict['emotion'] = pre_result_dict['emotion']
                    result_dict['state'] = 'NOT_RECOGNIZE_UC'
                    result_dict['answer'] = config.ANSWER['default_error_end']
                    result_dict['previous_phase'] = pre_result_dict['current_phase']
                    result_dict['current_phase'] = '/end_phase'
                    result_dict['next_phase'] = ['/end_phase']

                    '''
                    return {
                        'input': tokens + pre_tokens,
                        'intent': 'UNK',
                        'entity': entity,
                        'state': 'SUCCESS',
                        'emotion': '',
                        'emotions': [emotion] + pre_emotions,
                        'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                        # 'topic': topic,
                        'topics': [topic] + pre_topics,
                        'topic_prob': [max_topic_prob] + pre_topic_prob,
                        'answer': config.ANSWER['default_error_end'],
                        'previous_phase': ['check_ucs', '/other_use', '/recommend_contents', '/end_phase'],
                        'current_phase': '/end_phase',
                        'next_phase': ['/end_phase'],
                        'intent_turn_cnt': intent_turn_cnt
                    }
                    '''

                else:

                    result_dict['emotion'] = pre_result_dict['emotion']
                    result_dict['state'] = 'NOT_RECOGNIZE_UC'
                    result_dict['answer'] = config.ANSWER['default_error']
                    result_dict['previous_phase'] = pre_result_dict['current_phase']
                    result_dict['current_phase'] = '/other_user'
                    result_dict['next_phase'] = ['/induce_ucs', '/recongnize_uc_chat', '/recongnize_emotion_chat',
                                       '/recognize_uc', '/recognize_emotion', '/recognize_topic',
                                       '/end_chat', '/generate_emotion_chat', '/recommend_contents', '/end_phase'],

                    '''
                    return {
                        'input': tokens + pre_tokens,
                        'intent': 'UNK',
                        'entity': entity,
                        'state': 'SUCCESS',
                        'emotion': '',
                        'emotions': [emotion] + pre_emotions,
                        'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                        # 'topic': topic,
                        'topics': [topic] + pre_topics,
                        'topic_prob': [max_topic_prob] + pre_topic_prob,
                        'answer': config.ANSWER['default_error'],
                        'previous_phase': ['/welcomemsg_chat', '/other_user'],
                        'current_phase': '/other_user',
                        'next_phase': ['/induce_ucs', '/recongnize_uc_chat', '/recongnize_emotion_chat',
                                       '/recognize_uc', '/recognize_emotion', '/recognize_topic',
                                       '/end_chat', '/generate_emotion_chat', '/recommend_contents', '/end_phase'],
                        'intent_turn_cnt': intent_turn_cnt
                    }
                    '''

            elif pre_result_dict['intent'] == '마음상태호소' and \
                    result_dict['emotion_prob'] > config.EMOTION['threshold']:
                # 이전 대화가 감정대화 & 현재 감정 확률이 threshold를 넘었을 경우

                result_dict['emotion'] = result_dict['emotions'][0]
                result_dict['state'] = 'REQUIRE_CERTAIN_EMOTION'
                result_dict['answer'] = EmotionAnswerer().generate_answer_under5(text,
                                                                                 result_dict['emotion'],
                                                                                 result_dict['topics'][0])
                result_dict['previous_phase'] = pre_result_dict['current_phase']
                result_dict['current_phase'] = '/generate_emotion_chat'
                result_dict['next_phase'] = ['/generate_emotion_chat', '/end_chat', '/recognize_emotion_chat',
                                             '/recommend_contents', '/end_phase']

                '''
                return {
                    'input': tokens + pre_tokens,
                    'intent': intent,
                    'entity': [],
                    'state': 'SUCCESS',
                    'emotion': emotion,
                    'emotions': [emotion] + pre_emotions,
                    'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                    # 'topic': topic,
                    'topics': [topic] + pre_topics,
                    'topic_prob': [max_topic_prob] + pre_topic_prob,
                    'answer': EmotionAnswerer().generate_answer_under5(text),
                    'previous_phase': ['/recognize_emotion_chat'],
                    'current_phase': '/generate_emotion_chat',
                    'next_phase': ['/generate_emotion_chat', '/recognize_emotion_chat', '/recommend_contents',
                                   '/end_phase'],
                    'intent_turn_cnt': intent_turn_cnt
                }
                '''

            else:
                # 그 외에 에러 처리(인텐트 인식 잘못 했을 경우)

                result_dict['emotion'] = pre_result_dict['emotion']
                result_dict['state'] = 'ERROR_INTENT'
                result_dict['answer'] = config.ANSWER['default_error_ucs']
                result_dict['previous_phase'] = pre_result_dict['current_phase']
                result_dict['current_phase'] = pre_result_dict['current_phase']
                result_dict['next_phase'] = pre_result_dict['next_phase']

                '''
                return {
                    'input': tokens + pre_tokens,
                    'intent': intent,
                    'entity': entity,
                    'state': 'SUCCESS',
                    'emotion': '',
                    'emotions': [emotion] + pre_emotions,
                    'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                    # 'topic': topic,
                    'topics': [topic] + pre_topics,
                    'topic_prob': [max_topic_prob] + pre_topic_prob,
                    'answer': config.ANSWER['default_error_ucs'],
                    'previous_phase': pre_phase,
                    'current_phase': result_dict['current_phase'],
                    'next_phase': pre_pred_phases,
                    'intent_turn_cnt': intent_turn_cnt
                }
                '''
        return result_dict

    # def check_turn_cnt(self, turn_cnt):

    def _input2token(self, inputs: list) -> list:
        """
        inputs(CLS, SEP 토큰 포함)을 tokens로 바꾸기
        :param inputs: JointEnt 모델의 _tokenize() return 값
        :return: inputs에서 CLS, SEP 토큰 제외하고 언더바('_') 제외한 list
        """
        tokens = []
        for e in inputs:
            if e in ['[CLS]', '[SEP]']:
                continue
            tokens.append(e.strip('▁'))

        return tokens

    def __fit_intent(self):
        """
        Intent Classifier를 학습합니다.
        """

        self.intent_classifier.fit(self.dataset.load_intent(self.embed_processor))

    def __fit_entity(self):
        """
        Entity Recognizr를 학습합니다.
        """

        self.entity_recognizer.fit(self.dataset.load_entity(self.embed_processor))

    def __fit_embed(self):
        """
        Embedding을 학습합니다.
        """

        self.embed_processor.fit(self.dataset.load_embed())


def final_emotion(dict_: dict) -> dict:
    """
    컨텐츠 추천을 위한 최종 감정-주제와, 그 확률들을 리턴하는 함수
    :param dict_: 마지막 turn의 return dictionary
    :return: 감정과 주제가 key, 그에 대응하는 확률이 value인 딕셔너리
    """
    emotions = dict_['emotions']
    emotion_prob = dict_['emotion_prob']
    topics = dict_['topics']
    topic_prob = dict_['topic_prob']

    # emotion
    max_emotion = most_freq(emotions)
    pos = [i for i in range(len(emotions)) if emotions[i] == max_emotion]
    max_emotion_prob = emotion_prob[pos[0]]
    for idx in pos:
        if emotion_prob[idx] > max_emotion_prob:
            max_emotion_prob = emotion_prob[idx]

    # topic
    max_topic = most_freq(topics)
    pos = [i for i in range(len(topics)) if topics[i] == max_topic]
    max_topic_prob = topic_prob[pos[0]]
    for idx in pos:
        if topic_prob[idx] > max_topic_prob:
            max_topic_prob = topic_prob[idx]

    return {max_emotion: max_emotion_prob, max_topic: max_topic_prob}


def most_freq(list_: list):
    """
    리스트에서 최빈 값 리턴하는 함수
    :param list_: 감정(주제) 리스트
    :return: 리스트 최빈값
    """

    length = len(list_)
    arr = [0 for i in range(length)]
    for i in range(length):
        for j in range(i + 1, length):
            if list_[i] == list_[j]:
                arr[i] += 1
    max_idx = 0
    for i in range(length):
        for j in range(i + 1, length):
            if arr[i] < arr[j]:
                max_idx = j

    return list_[max_idx]
