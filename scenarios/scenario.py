# Copyright 2020 emotionchat. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from collections import Callable
from copy import deepcopy
from random import randint
import decorators.data as data
import emotionchat_config as config
from answerer.emotion_answerer import EmotionAnswerer
from answerer.discomfort_answerer import DiscomfortAnswerer
import re


# @data
class Scenario:
    """
    시나리오 객체
    불,궁,감의 여러 상황(slot filling, slot 다 채워진 경우, turn_cnt < 5 등..)에서의
    시나리오를 구분하고, 알맞은 답변을 answer에 저장한다.
    """

    def __init__(self, intent, api, scenario=None):
        self.intent = intent
        self.scenario, self.default = \
            self.__make_empty_dict(scenario)

        self.api, self.dict_keys, self.params = \
            self.__check_api(api)


    def __check_api(self, api):

        """
        api 체크 private 함수

        :param api: api 종류 ex) WeatherCrawler().request, DustCrawler().request, MapCrawler().request
        :return: api, dict_keys, parameters
        """

        if not isinstance(api, Callable):
            raise Exception('\n\n'
                            '0반드시 api는 callable 해야합니다.\n'
                            '입력하신 api의 타입은 {}입니다.\n'
                            '가급적이면 함수 이름 자체를 입력해주세요.'.format(type(api)))

        dict_keys = list(self.scenario.keys())
        pre_defined_entity = [entity.lower() for entity in config.DATA['NER_categories']]
        parameters = inspect.getfullargspec(api).args
        if 'self' in parameters: del parameters[0]
        # 만약 클래스의 멤버라면 self 인자를 지웁니다.

        if len(parameters) != len(dict_keys):
            raise Exception('\n\n'
                            '엔티티의 종류와 입력하신 API의 파라미터 수가 맞지 않습니다.\n'
                            '시나리오에 정의된 엔티티의 종류와 API의 파라미터 수는 일치해야합니다.\n'
                            '- 시나리오 엔티티 : {0}, {1}개\n'
                            '- API의 파라미터 : {2}, {3}개'.format(dict_keys, len(dict_keys),
                                                             parameters, len(parameters)))

        for entity in zip(parameters, dict_keys):
            api_param = entity[0]
            dict_key = entity[1]

            if dict_key.lower() not in pre_defined_entity:
                raise Exception('\n\n'
                                'emotionchat은 최대한 정확한 기능 수행을 위해 Config값에 정의된 Entity만 허용합니다. \n'
                                '- config에 정의된 엔티티 : {0}\n'
                                '- 시나리오 엔티티 : {1}\n'
                                '- 일치하지 않은 부분 : {2} not in {0}'.format(pre_defined_entity, dict_keys, dict_key))

            if api_param.lower() != dict_key.lower():
                raise Exception('\n\n'
                                'emotionchat은 최대한 정확한 기능 수행을 위해 API의 파라미터의 이름과 순서를 고려하여 엔티티와 맵핑합니다.\n'
                                'API 파라미터 이름과 시나리오의 엔티티의 \'순서\'와 \'이름\'을 가급적이면 동일하게 맞춰주시길 바랍니다.\n'
                                'API 파라미터 이름과 시나리오의 엔티티는 철자만 동일하면 됩니다, 대/소문자는 일치시킬 필요 없습니다.\n'
                                '- 시나리오 엔티티 : {0}\n'
                                '- API의 파라미터 : {1}\n'
                                '- 일치하지 않은 부분 : {2} != {3}'.format(dict_keys, parameters,
                                                                   api_param, dict_key))
        return api, dict_keys, parameters

    def __make_empty_dict(self, scenario):
        """

        entity 리스트 초기화 함수
        :param scenario:
        :return: scenarios, default
        """

        default = {}

        for k, v in scenario.items():
            if len(scenario[k]) > 0:
                default[k] = v
                # 디폴트 딕셔너리로 일단 빼놓고

                scenario[k] = []
                # 해당 엔티티의 리스트를 비워둠

            else:
                default[k] = []
                # 디폴드 없으면 리스트로 초기화

        return scenario, default

    def __check_entity(self, entity: list, tokens: list, dict_: dict) -> dict:
        """
        문자열과 엔티티를 함께 체크해서
        딕셔너리에 정의된 엔티티에 해당하는 단어 토큰만 추가합니다.

        :param tokens: 입력문자열 토큰(단어) 리스트
        :param entity: 엔티티 리스트(들어온 tokens을 entity recognizer로 인식한 토큰별 엔티티 리스트)
        :param dict_: 필요한 엔티티가 무엇인지 정의된 딕셔너리
        :return: 필요한 토큰들이 채워진 딕셔너리
        """

        for t, e in zip(tokens, entity):
            for k, v in dict_.items():
                if k.lower() in e.lower():
                    v.append(re.sub('[은는이가을를]', '', t))

        return dict_

    """
    def __check_emotion_topic(self, emotion: str, topic: str, tokens: list, dict_: dict) -> dict:
        for t, e in zip(tokens, emotion_topic):
            for k, v in dict_.items():
                if k.lower() in e.lower():
                    v.append(t)

        return dict_
    """


    def __set_default(self, result):
        """
        Scenario객체의 scenarios key의 엔티티들에게 default값이 여러 개 부여되었을 때,
        이 중 하나만 선택해서 하기.
        :param result:
        :return:
        """
        for k, v in result.items():
            if len(result[k]) == 0 and len(self.default[k]) != 0:
                # 디폴트 값 중에서 랜덤으로 하나 골라서 넣음
                result[k] = \
                    [self.default[k][randint(0, len(self.default[k]) - 1)]]

            result[k] = ' '.join(result[k])
        return result


    def apply(self, entity, tokens, intent_turn_cnt,
              emotion, pre_emotion, pre_emotions, max_emotion_prob, pre_emotion_prob,
              topic, pre_topics, max_topic_prob, pre_topic_prob, pre_entity) -> dict:
        """
        불궁 대화 시나리오 채우기
        :param entity: 엔티티
        :param tokens: 사용자 발화
        :return: 다 채워진 시나리오
        """

        scenario = deepcopy(self.scenario)
        result = self.__check_entity(entity, tokens, scenario)
        result = self.__set_default(result)
        required_entity = [k for k, v in result.items() if len(v) == 0]  # 필요한 엔티티 종류

        print("(system msg) pre_entity : " + str(pre_entity))
        print("(system msg) required_entity : " + str(required_entity))

        if len(required_entity) == 0:
            # entity가 채워진 경우
            return {
                'input': tokens,
                'intent': self.intent,
                'entity': entity,
                'state': 'SUCCESS',
                'emotion': pre_emotion,
                'emotions': [emotion] + pre_emotions,
                'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                #'topic': topic,
                'topics': [topic] + pre_topics,
                'topic_prob': [max_topic_prob] + pre_topic_prob,
                'answer': self.api(*result.values()),
                'previous_phase': ['/recognize_uc_chat', '/recognize_uc'],
                'current_phase': '/check_ucs',
                'next_phase': ['/check_ucs_positive', '/check_ucs_negative', '/check_ucs', '/end_phase'],
                'intent_turn_cnt': intent_turn_cnt
            }

        else:
            # entity가 채워지지 않은 경우
            if intent_turn_cnt < 4:
                # entity가 채워지지 않은 경우 & turn_cnt < 4
                if len(pre_entity) != 0:
                    # 이전에 채워진 엔티티가 있으면
                    return {
                        'input': tokens,
                        'intent': self.intent,
                        'entity': entity,
                        'state': 'SUCCESS',
                        'emotion': pre_emotion,
                        'emotions': [emotion] + pre_emotions,
                        'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                        #'topic': topic,
                        'topics': [topic] + pre_topics,
                        'topic_prob': [max_topic_prob] + pre_topic_prob,
                        'answer': self.api(*result.values()),   # 수정!!
                        'previous_phase': ['/recognize_uc_chat', '/recognize_uc'],
                        'current_phase': '/check_ucs',
                        'next_phase': ['/check_ucs_positive', '/check_ucs_negative', '/check_ucs', '/end_phase'],
                        'intent_turn_cnt': intent_turn_cnt
                    }

                else:
                    # 이전에 채워진 엔티티가 없으면
                    return {
                        'input': tokens,
                        'intent': self.intent,
                        'entity': entity + pre_entity,
                        'state': 'REQUIRE_' + '_'.join(required_entity),
                        'emotion': pre_emotion,
                        'emotions': [emotion] + pre_emotions,
                        'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                        #'topic': topic,
                        'topics': [topic] + pre_topics,
                        'topic_prob': [max_topic_prob] + pre_topic_prob,
                        'answer': DiscomfortAnswerer().fill_slot(required_entity),  # 수정
                        'previous_phase': ['/recognize_uc_chat', '/fill_slot'],
                        'current_phase': '/fill_slot',
                        'next_phase': ['/fill_slot', '/recognize_uc', '/check_ucs','/check_ucs_positive', '/check_ucs_negative'],
                        'intent_turn_cnt': intent_turn_cnt
                    }

            else:
                # entity가 채워지지 않은 경우 & turn_cnt >= 4
                if len(pre_entity) == 0:
                    return {
                        'input': tokens,
                        'intent': self.intent,
                        'entity': entity,
                        'state': 'FAIL_FILLING_SLOT',
                        'emotion': None,
                        'emotions': pre_emotions,
                        'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                        #'topic': topic,
                        'topics': [topic] + pre_topics,
                        'topic_prob': [max_topic_prob] + pre_topic_prob,
                        'answer': config.ANSWER['default_error_ucs'],
                        'previous_phase': ['/fill_slot', '/welcomemsg_chat', '/other_user',
                                           '/induce_ucs', '/recognize_uc_chat'],
                        'current_phase': '/recognize_uc',
                        'next_phase': ['/check_ucs', '/fill_slot', '/recognize_uc', '/check_ucs_positive', '/check_ucs_negative','/check_ucs', '/end_phase'],
                        'intent_turn_cnt': intent_turn_cnt
                    }

                return {
                    'input': tokens,
                    'intent': self.intent,
                    'entity': entity + pre_entity,
                    'state': 'FAIL_FILLING_SLOT',
                    'emotion': None,
                    'emotions': pre_emotions,
                    'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                    #'topic': topic,
                    'topics': [topic] + pre_topics,
                    'topic_prob': [max_topic_prob] + pre_topic_prob,
                    'answer': config.ANSWER['default_error_ucs'],
                    'previous_phase': ['/fill_slot', '/welcomemsg_chat', '/other_user',
                                       '/induce_ucs', '/recognize_uc_chat'],
                    'current_phase': '/recognize_uc',
                    'next_phase': ['/check_ucs'],
                    'intent_turn_cnt': intent_turn_cnt
                }


    def apply_greet(self, text: str, intent: str, pre_emotions: list, pre_emotion_prob: list, pre_topics: list,
                    pre_topic_prob: list, intent_turn_cnt: int) -> dict:
        """
        만남인사, 작별인사일 때 시나리오
        :param intent: 현재 인텐트
        :return: dict
        """
        return {
            'input': text,
            'intent': intent,
            'entity': [],
            'state': 'SUCCESS',
            'emotion': None,
            'emotions': pre_emotions,
            'emotion_prob': pre_emotion_prob,
            #'topic': None,
            'topics': pre_topics,
            'topic_prob': pre_topic_prob,
            'answer': config.ANSWER['welcomemsg_chat'],
            'previous_phase': None,
            'current_phase': '/welcomemsg_chat',
            'next_phase': ['/other_user', '/recognize_uc_chat', '/recognize_emotion_chat', '/recognize_uc', '/generate_emotion_chat', '/check_ucs'],
            'intent_turn_cnt': intent_turn_cnt
        }

    def apply_np(self, tokens: list, pre_tokens: list, intent, pre_intent, pre_emotions: list, pre_emotion_prob: list, pre_topics: list,
                 pre_topic_prob: list, intent_turn_cnt: int):
        """
        재질의에 대한 답변 구분 함수
        :param text: user utterance
        :param intent: 현재 인텐트
        :param pre_intent: 이전 인텐트
        :return: dict
        """
        if intent == '부정':
            return {
                'input': tokens + pre_tokens,
                'intent': intent,
                'entity': [],
                'state': 'SUCCESS',
                'emotion': None,
                'emotions': pre_emotions,
                'emotion_prob': pre_emotion_prob,
                #'topic': None,
                'topics': pre_topics,
                'topic_prob': pre_topic_prob,
                'answer': config.ANSWER['default_error_end_n'],
                'previous_phase': None,
                'current_phase': '/end_phase',
                'next_phase': ['/end_phase'],
                'intent_turn_cnt': intent_turn_cnt
            }
        else:
            # 긍정
            if pre_intent == '마음상태호소':
                return {
                    'input': tokens + pre_tokens,
                    'intent': intent,
                    'entity': [],
                    'state': 'SUCCESS',
                    'emotion': None,
                    'emotions': pre_emotions,
                    'emotion_prob': pre_emotion_prob,
                    #'topic': None,
                    'topics': pre_topics,
                    'topic_prob': pre_topic_prob,
                    'answer': config.ANSWER['default_contents'],
                    'previous_phase': None,
                    'current_phase': '/check_ucs_positive',
                    'next_phase': ['/end_chat'],
                    'intent_turn_cnt': intent_turn_cnt
                }

            elif pre_intent in config.SORT_INTENT['PHISICALDISCOMFORT']:
                return {
                    'input': tokens + pre_tokens,
                    'intent': intent,
                    'entity': [],
                    'state': 'SUCCESS',
                    'emotion': None,
                    'emotions': pre_emotions,
                    'emotion_prob': pre_emotion_prob,
                    #'topic': None,
                    'topics': pre_topics,
                    'topic_prob': pre_topic_prob,
                    'answer': config.ANSWER['call_caregiver'],
                    'previous_phase': None,
                    'current_phase': '/check_ucs_positive',
                    'next_phase': ['/end_chat'],
                    'intent_turn_cnt': intent_turn_cnt
                }

            else:
                return {
                    'input': tokens + pre_tokens,
                    'intent': intent,
                    'entity': [],
                    'state': 'SUCCESS',
                    'emotion': None,
                    'emotions': pre_emotions,
                    'emotion_prob': pre_emotion_prob,
                    #'topic': None,
                    'topics': pre_topics,
                    'topic_prob': pre_topic_prob,
                    'answer': '그러시군요. ' + config.ANSWER['default_error_end'],
                    'previous_phase': None,
                    'current_phase': '/end_phase',
                    'next_phase': ['/end_phase'],
                    'intent_turn_cnt': intent_turn_cnt
                }

    def apply_emotion(self, emotion: str, pre_emotion: str, pre_emotions: list,
                      max_emotion_prob: float, pre_emotion_prob: list,
                      topic: str, pre_topics: list, max_topic_prob: float, pre_topic_prob: list,
                      text: str, tokens: list, intent_turn_cnt: int, turn_cnt: int) -> dict:

        if turn_cnt < 5:
            if intent_turn_cnt <= 1:
                # 1-1. turn 수가 2회 이하일 경우
                if max_emotion_prob < config.EMOTION['threshold']:
                    # 2-1. 감정확률 < threshold일 경우
                    if pre_emotion == None:
                        # 3-1. 이전에 확실한 감정이 없었을 경우
                        return {
                            'input': tokens,
                            'intent': self.intent,
                            'entity': [],
                            'state': 'SUCCESS',
                            'emotion': None,
                            'emotions': [emotion] + pre_emotions,
                            'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                            #'topic': topic,
                            'topics': [topic] + pre_topics,
                            'topic_prob': [max_topic_prob] + pre_topic_prob,
                            'answer': EmotionAnswerer().generate_answer_under5(text, emotion, topic),
                            'previous_phase': ['/generate_emotion_chat'],
                            'current_phase': '/generate_emotion_chat',
                            'next_phase': ['/generate_emotion_chat', '/end_chat', '/recognize_emotion_chat', '/recommend_contents', '/end_phase'],
                            'intent_turn_cnt': intent_turn_cnt
                        }
                    else:
                        # 3-2. 이전에 확실한 감정이 있었을 경우
                        return{
                            'input': tokens,
                            'intent': self.intent,
                            'entity': [],
                            'state': 'SUCCESS',
                            'emotion': pre_emotion,
                            'emotions': [emotion] + pre_emotions,
                            'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                            #'topic': topic,
                            'topics': [topic] + pre_topics,
                            'topic_prob': [max_topic_prob] + pre_topic_prob,
                            'answer': EmotionAnswerer().generate_answer_under5(text, emotion, topic),
                            'previous_phase': ['/generate_emotion_chat'],
                            'current_phase': '/generate_emotion_chat',
                            'next_phase': ['/generate_emotion_chat', '/end_chat', '/recognize_emotion_chat', '/recommend_contents', '/end_phase'],
                            'intent_turn_cnt': intent_turn_cnt
                        }
                else:
                    # 2-2. 감정확률 >= threshold 일 경우
                    return {
                        'input': tokens,
                        'intent': self.intent,
                        'entity': [],
                        'state': 'SUCCESS',
                        'emotion': emotion,
                        'emotions': [emotion] + pre_emotions,
                        'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                        #'topic': topic,
                        'topics': [topic] + pre_topics,
                        'topic_prob': [max_topic_prob] + pre_topic_prob,
                        'answer': EmotionAnswerer().generate_answer_under5(text, emotion, topic),
                        'previous_phase': ['/generate_emotion_chat'],
                        'current_phase': '/generate_emotion_chat',
                        'next_phase': ['/generate_emotion_chat', '/end_chat', '/recognize_emotion_chat', '/recommend_contents', '/end_phase'],
                        'intent_turn_cnt': intent_turn_cnt
                    }

            elif 2 <= intent_turn_cnt <= 4:
                # 1-2. turn 수가 3회 이상 5회 이하일 경우
                if max_emotion_prob < config.EMOTION['threshold']:
                    # 2-1. 감정확률 < threshold일 경우
                    if pre_emotion == None:
                        # 3-1. 이전에 확실한 감정이 없었을 경우
                        return {
                            'input': tokens,
                            'intent': self.intent,
                            'entity': [],
                            'state': 'SUCCESS',
                            'emotion': None,
                            'emotions': [emotion] + pre_emotions,
                            'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                            #'topic': topic,
                            'topics': [topic] + pre_topics,
                            'topic_prob': [max_topic_prob] + pre_topic_prob,
                            'answer': EmotionAnswerer().generate_answer_under5(text, emotion, topic),
                            'previous_phase': ['/generate_emotion_chat'],
                            'current_phase': '/generate_emotion_chat',
                            'next_phase': ['/generate_emotion_chat', '/end_chat', '/recognize_emotion_chat',
                                           '/recommend_contents', '/end_phase'],
                            'intent_turn_cnt': intent_turn_cnt
                        }
                    else:
                        # 3-2. 이전에 확실한 감정이 있었을 경우
                        return {
                            'input': tokens,
                            'intent': self.intent,
                            'entity': [],
                            'state': 'SUCCESS',
                            'emotion': pre_emotion,
                            'emotions': [emotion] + pre_emotions,
                            'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                            #'topic': topic,
                            'topics': [topic] + pre_topics,
                            'topic_prob': [max_topic_prob] + pre_topic_prob,
                            'answer': EmotionAnswerer().contents_answer(text, emotion, topic),
                            'previous_phase': ['/generate_emotion_chat'],
                            'current_phase': '/end_phase',
                            'next_phase': ['/end_phase'],
                            'intent_turn_cnt': intent_turn_cnt
                        }
                else:
                    # 2-2. 감정확률 >= threshold 일 경우
                    if pre_emotion == None:
                        # 3-1. 이전에 확실한 감정이 없었을 경우
                        return {
                            'input': tokens,
                            'intent': self.intent,
                            'entity': [],
                            'state': 'SUCCESS',
                            'emotion': emotion,
                            'emotions': [emotion] + pre_emotions,
                            'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                            #'topic': topic,
                            'topics': [topic] + pre_topics,
                            'topic_prob': [max_topic_prob] + pre_topic_prob,
                            'answer': EmotionAnswerer().contents_answer(text, emotion, topic),
                            'previous_phase': ['/generate_emotion_chat'],
                            'current_phase': '/end_phase',
                            'next_phase': ['/end_phase'],
                            'intent_turn_cnt': intent_turn_cnt
                        }
                    else:
                        # 3-2. 이전에 확실한 감정이 있었을 경우
                        return {
                            'input': tokens,
                            'intent': self.intent,
                            'entity': [],
                            'state': 'SUCCESS',
                            'emotion': emotion,
                            'emotions': [emotion] + pre_emotions,
                            'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                            #'topic': topic,
                            'topics': [topic] + pre_topics,
                            'topic_prob': [max_topic_prob] + pre_topic_prob,
                            'answer': EmotionAnswerer().contents_answer(text, emotion, topic),
                            'previous_phase': ['/generate_emotion_chat'],
                            'current_phase': '/end_phase',
                            'next_phase': ['/end_phase'],
                            'intent_turn_cnt': intent_turn_cnt
                        }

            else:
                # 1-3. turn 수가 5회 이상일 경우
                if pre_emotion == None:
                    # 3-1. 이전에 확실한 감정이 없었을 경우
                    return {
                        'input': tokens,
                        'intent': self.intent,
                        'entity': [],
                        'state': 'FAIL_EMOTION',
                        'emotion': pre_emotion,
                        'emotions': [emotion] + pre_emotions,
                        'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                        #'topic': topic,
                        'topics': [topic] + pre_topics,
                        'topic_prob': [max_topic_prob] + pre_topic_prob,
                        'answer': EmotionAnswerer().generate_answer_over5(pre_emotions),
                        'previous_phase': ['/recognize_emotion_chat'],
                        'current_phase': '/end_phase',
                        'next_phase': ['/end_phase'],
                        'intent_turn_cnt': intent_turn_cnt
                    }
                else:
                    # 3-2. 이전에 확실한 감정이 있었을 경우
                    return {
                        'input': tokens,
                        'intent': self.intent,
                        'entity': [],
                        'state': 'SUCCESS',
                        'emotion': emotion,
                        'emotions': [emotion] + pre_emotions,
                        'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                        #'topic': topic,
                        'topics': [topic] + pre_topics,
                        'topic_prob': [max_topic_prob] + pre_topic_prob,
                        'answer': EmotionAnswerer().contents_answer(text, pre_emotion, topic),
                        'previous_phase': ['/generate_emotion_chat'],
                        'current_phase': '/end_phase',
                        'next_phase': ['/end_phase'],
                        'intent_turn_cnt': intent_turn_cnt
                    }

        else:
            # turn_cnt >=5 (총 6회 안녕 제외하고 5회)
            if pre_emotion == None:
                # 이전에 확실한 감정이 없었을 경우
                if max_emotion_prob > config.EMOTION['threshold']:
                    # 현재 감정이 threshold를 넘었을 경우
                    return {
                        'input': tokens,
                        'intent': self.intent,
                        'entity': [],
                        'state': 'SUCCESS',
                        'emotion': emotion,
                        'emotions': [emotion] + pre_emotions,
                        'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                        # 'topic': topic,
                        'topics': [topic] + pre_topics,
                        'topic_prob': [max_topic_prob] + pre_topic_prob,
                        'answer': EmotionAnswerer().contents_answer(text, pre_emotion, topic),
                        'previous_phase': ['/generate_emotion_chat'],
                        'current_phase': '/end_phase',
                        'next_phase': ['/end_phase'],
                        'intent_turn_cnt': intent_turn_cnt
                    }
                else:
                    # 현재 감정이 threshold를 넘지 않았을 경우
                    return {
                        'input': tokens,
                        'intent': self.intent,
                        'entity': [],
                        'state': 'FAIL_EMOTION',
                        'emotion': pre_emotion,
                        'emotions': [emotion] + pre_emotions,
                        'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                        # 'topic': topic,
                        'topics': [topic] + pre_topics,
                        'topic_prob': [max_topic_prob] + pre_topic_prob,
                        'answer': EmotionAnswerer().generate_answer_over5(pre_emotions),
                        'previous_phase': ['/recognize_emotion_chat'],
                        'current_phase': '/end_phase',
                        'next_phase': ['/end_phase'],
                        'intent_turn_cnt': intent_turn_cnt
                    }
            else:
                # 이전에 확실한 감정이 있었을 경우
                return {
                    'input': tokens,
                    'intent': self.intent,
                    'entity': [],
                    'state': 'SUCCESS',
                    'emotion': emotion,
                    'emotions': [emotion] + pre_emotions,
                    'emotion_prob': [max_emotion_prob] + pre_emotion_prob,
                    # 'topic': topic,
                    'topics': [topic] + pre_topics,
                    'topic_prob': [max_topic_prob] + pre_topic_prob,
                    'answer': EmotionAnswerer().contents_answer(text, pre_emotion, topic),
                    'previous_phase': ['/generate_emotion_chat'],
                    'current_phase': '/end_phase',
                    'next_phase': ['/end_phase'],
                    'intent_turn_cnt': intent_turn_cnt
                }
