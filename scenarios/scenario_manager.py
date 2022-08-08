from scenarios.scenario import Scenario
import emotionchat_config as config
#from data.organizer import Organizer
#from data.preprocessor import Preprocessor
#from decorators import data
#from model.curious_intent.proc.base_processor import BaseProcessor
from model.proc import DistanceClassifier, GensimEmbedder, IntentClassifier, EntityRecognizer
from model import curious_intent, embed, curious_entity
from model.loss import CenterLoss, CRFLoss
from data.dataset import Dataset

class ScenarioManager:
    """
    시나리오 객체 관리하는 클래스
    불,궁,감,모름을 구분해 각각 다른 시나리오를 적용함.
    """

    def __init__(self,
                 embed_processor,
                 intent_classifier,
                 entity_recognizer):

        self.scenarios = []
        self.dataset = Dataset(ood=True)
        self.intent_dict = {'날씨': 1, '미세먼지': 3}

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


    def add_scenario(self, scen: Scenario):
        if isinstance(scen, Scenario):
            self.scenarios.append(scen)
        else:
            raise Exception('시나리오 객체만 입력 가능합니다.')

    def apply_scenario(self, pre_result_dict, result_dict, text, c_ucs, turn_cnt):
        '''
        if result_dict['intent'] == '궁금함':
            # 현재 대화가 궁금함 대화일 경우
            print('(system msg) intent 궁금함 들어옴')
            prep = self.dataset.load_predict(text, self.embed_processor)
            intent = self.intent_classifier.predict(prep, calibrate=False)
            entity = self.entity_recognizer.predict(prep)
            print('(system msg) intent : ' + str(intent))
            result_dict['intent'] = result_dict['intent'] + '_' + intent  # 궁금함_dust
            result_dict['entity'] = entity
        '''

        for scenario in self.scenarios:
            # default_scenario 에 있는 경우 default_scenario를 기본적으로 따르도록
            if c_ucs:
                # 이전 단계에서 불,궁,감 대화에 들어왔으면
                if pre_result_dict['current_phase'] == '/check_ucs':
                    # 이전 단계가 check_ucs 였을 경우
                    #if result_dict['emotions'][0] in config.EMOTION['긍정']:
                       ## 현재 감정이 긍정에 속하는 감정일 경우

                    result_dict['input'] = result_dict['input']+ pre_result_dict['input']
                    result_dict['emotion'] = pre_result_dict['emotion']
                    result_dict['emotions'] = result_dict['emotions'] + pre_result_dict['emotions']
                    result_dict['emotion_prob'] = result_dict['emotion_prob'] + pre_result_dict['emotion_prob']
                    result_dict['topics'] = result_dict['topics'] + pre_result_dict['topics']
                    result_dict['topic_prob'] = result_dict['topic_prob'] + pre_result_dict['topic_prob']
                    result_dict['state'] = 'END'
                    result_dict['answer'] = config.ANSWER['goodbyemsg_uc']
                    result_dict['previous_phase'] = ['/welcomemsg_chat', '/other_user']
                    result_dict['previous_phase'] = '/end_phase'
                    result_dict['previous_phase'] = ['/end_phase']

                    '''
                    return {
                        # 수정
                        'input': result_dict['input']+ pre_result_dict['input'],
                        'intent': result_dict['intent'],
                        'entity': result_dict['entity'],
                        'emotion': pre_result_dict['emotion'],
                        'emotions': result_dict['emotions'] + pre_result_dict['emotions'],  # 'emotions': [pre_emotion] + pre_emotions,
                        'emotion_prob': result_dict['emotion_prob'] + pre_result_dict['emotion_prob'],
                        'emotion_probs': result_dict['emotion_probs'],
                        'topics': result_dict['topics'] + pre_result_dict['topics'],  # 'topics': [topic] + pre_topics,
                        'topic_prob': result_dict['topic_prob'] + pre_result_dict['topic_prob'],  # 'topic_prob': [0] + pre_topic_prob,
                        'state': 'END',
                        'answer': config.ANSWER['goodbyemsg_uc'],
                        'previous_phase': ['/welcomemsg_chat', '/other_user'],
                        'current_phase': '/end_phase',
                        'next_phase': ['/end_phase'],
                        'intent_turn_cnt': result_dict['intent_turn_cnt']
                    }
                    '''

                    '''
                    else:
                        return {
                            'input': result_dict['input']+ pre_result_dict['input'],
                            'intent': intent,
                            'entity': entity,
                            'emotion': pre_emotion,
                            'emotions': [emotion] + pre_emotions,
                            'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                            # 'topic': topic,
                            'topics': [topic] + pre_topics,
                            'topic_prob': [max_topic_prob] + pre_topic_prob,
                            'state': 'FALLBACK',
                            'answer': config.ANSWER['goodbyemsg_uc'],
                            'previous_phase': ['/welcomemsg_chat', '/other_user'],
                            'current_phase': '/end_phase',
                            'next_phase': ['/end_phase'],
                            'intent_turn_cnt': intent_turn_cnt
                        }
                    '''

                elif (scenario.intent == pre_result_dict['intent']) and (pre_result_dict['intent'] in config.SORT_INTENT['PHISICALDISCOMFORT']):
                    # 이전 단계에서 불편함 대화였으면
                    return scenario.apply(pre_result_dict, result_dict)

                elif (scenario.intent == pre_result_dict['intent']) and (pre_result_dict['intent'] in config.SORT_INTENT['QURIOUS']):
                    # 이전 단계에서 궁금함 대화였으면
                    prep = self.dataset.load_predict(text, self.embed_processor)
                    #intent = self.intent_classifier.predict(prep, calibrate=False)
                    entity = self.entity_recognizer.predict(prep)
                    #print('(system msg) intent : ' + str(intent))
                    #result_dict['intent'] = intent    # 궁금함_dust
                    result_dict['entity'] = entity
                    return scenario.apply(pre_result_dict, result_dict)

                elif (scenario.intent == pre_result_dict['intent']) and pre_result_dict['intent'] == '마음상태호소':
                    # 이전 단계에서 감정 대화였으면
                    return scenario.apply_emotion(pre_result_dict, result_dict, text, turn_cnt)

                else:
                    continue

            ############################# #############################
            else:
                # 이전 대화에서 불,궁,감 대화에 안들어왔으면
                # 다른 인텐트 존재 가능

                if result_dict['intent'] == '궁금함':
                    # 현재 대화가 궁금함 대화일 경우
                    prep = self.dataset.load_predict(text, self.embed_processor)
                    intent = self.intent_classifier.predict(prep, calibrate=False)
                    entity = self.entity_recognizer.predict(prep)
                    print('(system msg) intent : ' + str(intent))
                    result_dict['intent'] = intent    # 궁금함_dust
                    result_dict['entity'] = entity
                    return scenario.apply(pre_result_dict, result_dict)


                # (불궁일 때)현재 대화의 scenario의 intent랑 들어온 인텐트가 같으면 default_scenario대로 수행하게
                elif (scenario.intent == result_dict['intent']) and (result_dict['intent'] in config.SORT_INTENT['PHISICALDISCOMFORT']):
                # 각 intent 별 시나리오를 demo.scenarios.py에 저장해놨기 때문에 그 시나리오에 기록하면서 사용
                    print('(system msg) 엔티티 : ' + str(result_dict['entity']))
                    return scenario.apply(pre_result_dict, result_dict)

                # (감정일 때)현재 대화의 scenario의 intent랑 들어온 인텐트가 같으면 감정, 주제 필링 수행
                elif (scenario.intent == result_dict['intent']) and (result_dict['intent'] in config.SORT_INTENT['SENTIMENTDISCOMFORT']):
                    # 각 intent 별 시나리오를 demo.scenarios.py에 저장해놨기 때문에 그 시나리오에 기록하면서 사용
                    return scenario.apply_emotion(pre_result_dict, result_dict, text, turn_cnt)

                else:
                    continue

            #############################  #############################

        else:
            print("(system msg) scenario 반복문 빠져나옴")
        # default_scenario에 없는 시나리오 즉, 넋두리(긍정, 부정일 경우에도 여기에 속함)
            if result_dict['intent'] in ['부정', '긍정']:
                return scenario.apply_np(pre_result_dict, result_dict)
            # (인사일 때)
            elif result_dict['intent'] == '만남인사':
                # 각 intent 별 시나리오를 demo.scenarios.py에 저장해놨기 때문에 그 시나리오에 기록하면서 사용
                print("(system msg) scenario 반복문 빠져나옴")
                return scenario.apply_greet(pre_result_dict, result_dict)
            # (UNK일 때)
            else:
                return scenario.apply_unk(pre_result_dict, result_dict)  # apply_unk() 생성 예정

        '''
        # 넋두리(모름)일 때
        # 인텐트 == 작별인사, 만남인사, 부정, 긍정, 궁금함, UNK
        if intent_turn_cnt < 4:
            print('(system msg) scenario_manager for문 끝나고의 단계/ 넋두리 turn_cnt:', intent_turn_cnt)
            return {
                    'input': list(tokens),
                    'intent': intent,
                    'entity': entity,
                    'emotion': pre_emotion,
                    'emotions': [emotion] + pre_emotions,
                    'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                    #'topic': topic,
                    'topics': [topic] + pre_topics,
                    'topic_prob': [max_topic_prob] + pre_topic_prob,
                    'state': 'FALLBACK',
                    'answer': '(현재 intent :' + intent  + ')' + config.ANSWER['default_error_ucs'],
                    'previous_phase': ['/welcomemsg_chat', '/other_user'],
                    'current_phase': '/other_user',
                    'next_phase': ['/induce_ucs', '/recongnize_uc_chat', '/recongnize_emotion_chat',
                                   '/recognize_uc', '/recognize_emotion', '/recognize_topic',
                                   '/check_ucs', '/end_chat','/generate_emotion_chat'],
                    'intent_turn_cnt': intent_turn_cnt
                }
        else:
            print('(system msg) scenario_manager for문 끝나고의 단계/ 넋두리 turn_cnt 5회 이상')
            return {
                    'input': list(tokens),
                    'intent': intent,
                    'entity': entity,
                    'emotion': pre_emotion,
                    'emotions': [emotion] + pre_emotions,
                    'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                    #'topic': topic,
                    'topics': [topic] + pre_topics,
                    'topic_prob': [max_topic_prob] + pre_topic_prob,
                    'state': 'FALLBACK',
                    'answer': config.ANSWER['default_error_other_user'],
                    'previous_phase': ['/welcomemsg_chat', '/other_user'],
                    'current_phase': '/end_phase',
                    'next_phase': ['/end_phase'],
                    'intent_turn_cnt': intent_turn_cnt
                }

        '''

        '''
    def load_predict(self, text: str, emb_processor: BaseProcessor) -> Tensor:
        """
        실제 애플리케이션 등에서 유저 입력에 대한 인퍼런스를 수행할 때
        사용자가 입력한 Raw 텍스트(str)를 텐서로 변환합니다.

        :param text: 사용자의 텍스트 입력입니다.
        :param emb_processor: 임베딩 과정이 들어가므로 임베딩 프로세서를 입력해야합니다.
        :return: 유저 입력 추론용 텐서를 리턴합니다.
        """

        text = self.prep.tokenize(text, train=False)  # 토크나이징

        if len(text) == 0:
            raise Exception("문장 길이가 0입니다.")

        text = emb_processor.predict(text)  # 임베딩
        text, _ = self.prep.pad_sequencing(text)  # 패드 시퀀싱
        return text.unsqueeze(0).to(self.device)  # 차원 증가 (batch_size = 1)
        '''