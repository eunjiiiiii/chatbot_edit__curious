"""

시나리오를 만든 이유
각 인텐트마다 필요한 엔티티가 다르게 정해져 있기 때문.
ex) 신체불편호소 : 신체부위(BODY), 증상(SYMPTOM)

@intent 종류
1) 궁금함
-weather
-dust
-restaurant
-travel

2) 불편함
-physicalDiscomfort
-sleepProblem
-moveHelp
-changePosture
-higieneAct
-otherAct
-environmentalDiscomfort
-expressDesire
-foodDiscomfort

3) 감정 불편 호소
sentimentDiscomfort

@entity 종류
-BODY
-SYMPTOM
-PLACE
-FOOD
-DAILYSUPPLIES
-ACT(IVITY)
-HOMEAPP
-ASSISTANTDEV
-RELIGIOUSACT
-TIME
-WEATHER
-MONEY
"""

from kocrawl.dust import DustCrawler
from kocrawl.weather import WeatherCrawler
from scenarios.scenario import Scenario
from kocrawl.map import MapCrawler
from answerer.discomfort_answerer import DiscomfortAnswerer
from answerer.emotion_answerer import EmotionAnswerer


weather = Scenario(
    intent='weather',
    api=WeatherCrawler().request,
    scenario={
        'LOCATION': [],
        'DATE': ['오늘']
    }
)

dust = Scenario(
    intent='dust',
    api=DustCrawler().request,
    scenario={
        'LOCATION': [],
        'DATE': ['오늘']
    }
)

### 추가
# 불편함

physicalDiscomfort = Scenario(
    intent='신체불편호소',
    api=DiscomfortAnswerer().physicalDiscomfort_check_form,
    scenario={
        'BODY': [],
        'SYMPTOM': ['']
    }
)

sleepProblem = Scenario(
    intent='수면문제호소',
    api=DiscomfortAnswerer().sleepProblem_check_form,
    scenario={
    }
)

moveHelp = Scenario(
    intent='이동도움요구',
    api=DiscomfortAnswerer().moveHelp_check_form,
    scenario={
        'PLACE': []
    }
)

changePosture = Scenario(
    intent='자세변경요구',
    api=DiscomfortAnswerer().changePosture_check_form,
    scenario={
    }
)

higieneAct = Scenario(
    intent='위생활동요구',
    api=DiscomfortAnswerer().higieneAct_check_form,
    scenario={
    }
)

otherAct = Scenario(
    intent='기타활동요구',
    api=DiscomfortAnswerer().otherAct_check_form,
    scenario={
    }
)

environmentalDiscomfort = Scenario(
    intent='환경불편호소',
    api=DiscomfortAnswerer().environmentalDiscomfort_check_form,
    scenario={
        'PLACE': []
    }
)

expressDesire = Scenario(
    intent='욕구표출',
    api=DiscomfortAnswerer().expressDesire_check_form,
    scenario={
    }
)

foodDiscomfort = Scenario(
    intent='음식불편호소',
    api=DiscomfortAnswerer().foodDiscomfort_check_form,
    scenario={
        'FOOD': []
    }
)

sentimentDiscomfort = Scenario(
    intent='마음상태호소',
    api=EmotionAnswerer().generate_answer_collection,
    scenario={
        'EMOTION': [],
        'PRE_EMOTION': [None],
        'PRE_EMOTIONS': [None],
        'MAX_EMOTION_PROB': [0],
        'TOPIC': [],
        'MAX_TOPIC_PROB': [0],
        'TEXT': ['나 우울해'],
        'TURN_CNT': [0],
        'PRE_EMOTION_PROB': [None],
    }
)

# 감정대화는 엔티티 대신 emotion, topic을 요구함