
"""
Анализатор упоминаний для мониторинга репутации брендов
Использует открытые NLP модели для автоматизации
"""

import re
import pandas as pd
from typing import List, Dict, Tuple, Optional
import numpy as np

# Для тональности - установить: pip install transformers torch sentencepiece
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Внимание: библиотека transformers не установлена. Используйте: pip install transformers torch")

# Для базовой токенизации и стемминга (альтернатива)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
    # Скачиваем необходимые ресурсы
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except:
        print("Скачиваем ресурсы NLTK...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    print("Внимание: NLTK не установлен. Используйте: pip install nltk")

class MentionAnalyzer:
    """
    Автоматический анализатор упоминаний для мониторинга репутации
    """
    
    # Предопределенные теги (универсальные для любого ОМ)
    UNIVERSAL_TAGS = {
        'quality_issue': 'Проблемы с качеством',
        'positive_feedback': 'Положительный отзыв',
        'negative_feedback': 'Негативный отзыв',
        'comparison': 'Сравнение с конкурентами',
        'question': 'Вопрос/справка',
        'complaint': 'Жалоба',
        'recommendation': 'Рекомендация/совет',
        'social_responsibility': 'Социальная ответственность',
        'corporate_info': 'Корпоративная информация',
        'health_issue': 'Проблемы со здоровьем',
        'price_issue': 'Вопросы цены',
        'service_issue': 'Проблемы с сервисом',
        'product_info': 'Информация о продукте',
        'advertisement': 'Реклама/акция'
    }
    
    def __init__(self, 
                 object_name: str,
                 keywords: List[str],
                 risk_words: Optional[List[str]] = None,
                 positive_words: Optional[List[str]] = None,
                 exclude_phrases: Optional[List[str]] = None,
                 use_advanced_nlp: bool = True):
        """
        Инициализация анализатора
        
        Args:
            object_name: Название объекта мониторинга
            keywords: Ключевые слова для определения релевантности
            risk_words: Слова, указывающие на риск (для оценки опасности)
            positive_words: Слова, указывающие на позитив
            exclude_phrases: Фразы для исключения ложных срабатываний
            use_advanced_nlp: Использовать продвинутые NLP модели
        """
        self.object_name = object_name
        self.keywords = [kw.lower() for kw in keywords]
        self.risk_words = [rw.lower() for rw in (risk_words or [])]
        self.positive_words = [pw.lower() for pw in (positive_words or [])]
        self.exclude_phrases = [ep.lower() for ep in (exclude_phrases or [])]
        
        # Инициализация NLP моделей
        self.sentiment_analyzer = None
        self.tokenizer = None
        self.use_advanced_nlp = use_advanced_nlp
        
        if use_advanced_nlp and TRANSFORMERS_AVAILABLE:
            self._init_nlp_models()
        
        # Словарь для сопоставления тегов
        self.tag_mapping = self._create_tag_mapping()
    
    def _init_nlp_models(self):
        """Инициализация NLP моделей"""
        try:
            # Модель для определения тональности на русском языке
            model_name = "blanchefort/rubert-base-cased-sentiment"
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                framework="pt"
            )
            print(f"Модель тональности {model_name} загружена")
        except Exception as e:
            print(f"Не удалось загрузить модель тональности: {e}")
            self.use_advanced_nlp = False
    
    def _create_tag_mapping(self) -> Dict[str, List[str]]:
        """Создание маппинга ключевых слов для тегов"""
        # Базовые паттерны для универсальных тегов
        mapping = {
            'quality_issue': [
                'качеств', 'некачествен', 'плох', 'ужасн', 'кошмар', 'брак',
                'испорч', 'грязн', 'вредн', 'опасн'
            ],
            'positive_feedback': [
                'нравится', 'люблю', 'обожаю', 'отличн', 'прекрасн', 'замечательн',
                'хорош', 'супер', 'класс', 'лучш'
            ],
            'negative_feedback': [
                'не нравится', 'ненавижу', 'ужасн', 'плох', 'разочарован',
                'отвратительн', 'кошмар'
            ],
            'comparison': [
                'лучше чем', 'хуже чем', 'сравнен', 'в отличие от',
                'по сравнению с', 'чем'
            ],
            'question': [
                'как', 'что', 'где', 'когда', 'почему', 'зачем', 'сколько',
                '?', 'подскажит', 'посоветуй', 'расскажит'
            ],
            'complaint': [
                'жалоб', 'недовол', 'возмущен', 'протест', 'претенз',
                'требую', 'верните', 'вернуть'
            ],
            'health_issue': [
                'здоровь', 'болезн', 'больн', 'аллерг', 'отравлен',
                'сыпь', 'рвота', 'запор', 'колик', 'диарея'
            ],
            'price_issue': [
                'дорог', 'дешев', 'цен', 'стоимость', 'переплат',
                'наценк', 'скидк', 'акци', 'распродаж'
            ],
            'service_issue': [
                'обслуж', 'сервис', 'персонал', 'консульта', 'поддержк',
                'мастер', 'ремонт', 'гаранти'
            ]
        }
        return mapping
    
    def is_relevant(self, text: str) -> bool:
        """
        Определение релевантности сообщения
        
        Args:
            text: Текст сообщения
            
        Returns:
            bool: True если сообщение релевантно
        """
        text_lower = text.lower()
        
        # Проверка на исключающие фразы (ложные срабатывания)
        for exclude in self.exclude_phrases:
            if exclude in text_lower:
                return False
        
        # Проверка наличия ключевых слов
        for keyword in self.keywords:
            # Поиск точного совпадения или совпадения в составе слова
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower, flags=re.IGNORECASE):
                return True
        
        # Дополнительная проверка для сложных случаев
        words = re.findall(r'\b\w+\b', text_lower)
        for word in words:
            for keyword in self.keywords:
                # Частичное совпадение (для склонений, опечаток)
                if keyword in word or word in keyword:
                    return True
        
        return False
    
    def detect_sentiment_advanced(self, text: str) -> str:
        """Определение тональности с использованием ML модели"""
        if not self.sentiment_analyzer:
            return self.detect_sentiment_basic(text)
        
        try:
            result = self.sentiment_analyzer(text[:512])  # Ограничиваем длину
            label = result[0]['label']
            score = result[0]['score']
            
            # Маппинг результатов модели на наши категории
            sentiment_map = {
                'POSITIVE': 'Позитивная',
                'NEGATIVE': 'Негативная',
                'NEUTRAL': 'Нейтральная'
            }
            
            return sentiment_map.get(label, 'Нейтральная')
        except Exception as e:
            print(f"Ошибка анализа тональности: {e}")
            return self.detect_sentiment_basic(text)
    
    def detect_sentiment_basic(self, text: str) -> str:
        """Базовый анализ тональности по ключевым словам"""
        text_lower = text.lower()
        
        # Подсчет позитивных и негативных маркеров
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.risk_words if word in text_lower)
        
        # Анализ эмоциональных маркеров
        positive_patterns = [
            r'\b(отличн|прекрасн|замечательн|супер|класс|лучш|хорош)\b',
            r'[😀😃😄😁😆😍🤩]',
            r'\b(спасибо|благодар|рекомендую|советую)\b'
        ]
        
        negative_patterns = [
            r'\b(плох|ужасн|кошмар|отвратительн|ужасно|плохо)\b',
            r'[😠😡🤬😢😭😤]',
            r'\b(жалоб|недовол|возмущен|разочарован)\b'
        ]
        
        for pattern in positive_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                positive_count += 1
        
        for pattern in negative_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                negative_count += 1
        
        # Определение тональности
        if positive_count > negative_count and positive_count > 0:
            return 'Позитивная'
        elif negative_count > positive_count and negative_count > 0:
            return 'Негативная'
        elif positive_count == negative_count and positive_count > 0:
            return 'Смешанная'
        else:
            return 'Нейтральная'
    
    def assign_tags_advanced(self, text: str) -> List[str]:
        """Автоматическое присвоение тегов с использованием NLP"""
        tags = []
        text_lower = text.lower()
        
        # Проверка каждого тега по паттернам
        for tag_id, patterns in self.tag_mapping.items():
            for pattern in patterns:
                # Поиск паттерна в тексте
                if re.search(r'\b' + pattern + r'\w*\b', text_lower, re.IGNORECASE):
                    if self.UNIVERSAL_TAGS[tag_id] not in tags:
                        tags.append(self.UNIVERSAL_TAGS[tag_id])
                    break  # Переходим к следующему тегу
        
        # Дополнительная логика для специфичных тегов
        self._assign_specific_tags(text_lower, tags)
        
        # Ограничиваем количество тегов (максимум 5)
        return tags[:5]
    
    def _assign_specific_tags(self, text: str, tags: List[str]):
        """Дополнительная логика для специфичных тегов"""
        # Тег "Вопрос/справка"
        if re.search(r'\?|подскажит|посоветуй|расскажит|как\s+\w+|что\s+\w+', text):
            if 'Вопрос/справка' not in tags:
                tags.append('Вопрос/справка')
        
        # Тег "Сравнение с конкурентами"
        competitors = ['нутрилак', 'нутриция', 'нестле', 'хипп', 'семпер',
                      'лукойл', 'татнефть', 'роснефть', 'shell', 'бп']
        if any(comp in text for comp in competitors):
            if 'Сравнение с конкурентами' not in tags:
                tags.append('Сравнение с конкурентами')
        
        # Тег "Жалоба"
        complaint_patterns = [
            r'жалоб\w+', r'недовол\w+', r'возмущ\w+', r'протест\w+',
            r'претенз\w+', r'требую', r'верните', r'напиш\w+\s+жалобу'
        ]
        if any(re.search(pattern, text) for pattern in complaint_patterns):
            if 'Жалоба' not in tags:
                tags.append('Жалоба')
    
    def is_dangerous_for_reputation(self, text: str, sentiment: str) -> bool:
        """
        Определение опасности для репутации
        
        Args:
            text: Текст сообщения
            sentiment: Определенная тональность
            
        Returns:
            bool: True если сообщение опасно для репутации
        """
        text_lower = text.lower()
        
        # Критерии опасности
        danger_criteria = [
            # 1. Негативная тональность
            sentiment == 'Негативная',
            
            # 2. Наличие риск-слов
            any(risk_word in text_lower for risk_word in self.risk_words),
            
            # 3. Жалобы на здоровье/безопасность
            any(health_word in text_lower for health_word in 
                ['отравление', 'аллергия', 'опасно', 'вредно', 'угроза']),
            
            # 4. Призывы к бойкоту/жалобам в органы
            any(action_word in text_lower for action_word in
                ['роспотребнадзор', 'пожалуюсь', 'заявлен', 'иск', 'суд']),
            
            # 5. Вирусный потенциал (восклицания, капс, множественные знаки)
            bool(re.search(r'!{2,}|[A-ZА-Я]{5,}', text))
        ]
        
        # Сообщение опасно, если выполняется хотя бы 2 критерия
        return sum(danger_criteria) >= 2
    
    def analyze_mention(self, text: str) -> Dict:
        """
        Полный анализ одного упоминания
        
        Returns:
            Dict: Результаты анализа
        """
        relevant = self.is_relevant(text)
        
        if not relevant:
            return {
                'relevant': False,
                'sentiment': 'Нерелевантно',
                'tags': [],
                'dangerous': False
            }
        
        # Определение тональности
        if self.use_advanced_nlp and self.sentiment_analyzer:
            sentiment = self.detect_sentiment_advanced(text)
        else:
            sentiment = self.detect_sentiment_basic(text)
        
        # Присвоение тегов
        tags = self.assign_tags_advanced(text)
        
        # Оценка опасности
        dangerous = self.is_dangerous_for_reputation(text, sentiment)
        
        return {
            'relevant': True,
            'sentiment': sentiment,
            'tags': tags,
            'dangerous': dangerous
        }


class BatchAnalyzer:
    """Пакетный анализатор для обработки таблиц"""
    
    def __init__(self, configs: Dict[str, Dict]):
        """
        Инициализация для нескольких объектов мониторинга
        
        Args:
            configs: Словарь конфигураций {object_name: config}
        """
        self.analyzers = {}
        for obj_name, config in configs.items():
            self.analyzers[obj_name] = MentionAnalyzer(
                object_name=obj_name,
                keywords=config.get('keywords', []),
                risk_words=config.get('risk_words', []),
                positive_words=config.get('positive_words', []),
                exclude_phrases=config.get('exclude_phrases', []),
                use_advanced_nlp=config.get('use_advanced_nlp', True)
            )
    
    def analyze_dataframe(self, 
                         df: pd.DataFrame,
                         text_column: str,
                         object_column: Optional[str] = None,
                         object_name: Optional[str] = None) -> pd.DataFrame:
        """
        Анализ датафрейма с упоминаниями
        
        Args:
            df: DataFrame с данными
            text_column: Название столбца с текстом
            object_column: Столбец с указанием объекта (если несколько)
            object_name: Фиксированное название объекта (если один)
            
        Returns:
            DataFrame с результатами анализа
        """
        results = []
        
        for idx, row in df.iterrows():
            text = row[text_column]
            
            # Определение используемого анализатора
            if object_column:
                obj_name = row[object_column]
                analyzer = self.analyzers.get(obj_name)
                if not analyzer:
                    continue
            elif object_name:
                analyzer = self.analyzers.get(object_name)
                if not analyzer:
                    continue
            else:
                raise ValueError("Укажите object_column или object_name")
            
            # Анализ
            analysis = analyzer.analyze_mention(text)
            
            # Формирование строки результата
            result_row = {
                'Текст': text,
                'Сообщение релевантно?': 'Да' if analysis['relevant'] else 'Нет',
                'Тональность': analysis['sentiment'],
                'Опасно для репутации': 'Да' if analysis['dangerous'] else 'Нет'
            }
            
            # Добавление тегов (до 5)
            for i in range(5):
                tag_key = f'Тег {i+1}'
                if i < len(analysis['tags']):
                    result_row[tag_key] = analysis['tags'][i]
                else:
                    result_row[tag_key] = ''
            
            results.append(result_row)
        
        return pd.DataFrame(results)


# Пример использования
def main():
    """Пример использования анализатора"""
    
    # Конфигурация для бренда "Малютка"
    malyutka_config = {
        'keywords': ['малютка', 'nutricia', 'нутриция', 'детск', 'питани'],
        'risk_words': ['запор', 'колики', 'сыпь', 'рвота', 'аллергия', 
                      'плох', 'ужас', 'кошмар', 'некачествен'],
        'positive_words': ['нравится', 'люблю', 'обожаю', 'отличн', 
                          'хорош', 'рекомендую', 'доверяю'],
        'exclude_phrases': ['дом малютки', 'малютка родилась'],
        'use_advanced_nlp': TRANSFORMERS_AVAILABLE
    }
    
    # Конфигурация для "Газпром нефть"
    gazprom_config = {
        'keywords': ['газпромнефть', 'газпром', 'азс газпром', 
                    'gazprom', 'гпн', 'gdrive'],
        'risk_words': ['бадяж', 'некачествен', 'плох', 'брак', 
                      'обман', 'развод', 'жульнич'],
        'positive_words': ['хорош', 'качествен', 'отличн', 'рекомендую'],
        'exclude_phrases': [],
        'use_advanced_nlp': TRANSFORMERS_AVAILABLE
    }
    
    # Инициализация пакетного анализатора
    configs = {
        'Малютка': malyutka_config,
        'Газпром нефть': gazprom_config
    }
    
    analyzer = BatchAnalyzer(configs)
    
    print("Анализатор инициализирован.")
    print(f"Используются продвинутые NLP модели: {TRANSFORMERS_AVAILABLE}")
    
    # Пример анализа одного сообщения
    test_texts = [
        "Малютка очень вкусные каши, мои дети едят с удовольствием!",
        "От малютки у ребенка началась страшная аллергия и сыпь по всему телу!",
        "Газпром на Щербакова опять бадяжит бензин, машина сломалась!"
    ]
    
    for text in test_texts:
        # Определяем объект по тексту
        if any(kw in text.lower() for kw in malyutka_config['keywords']):
            analyzer_name = 'Малютка'
        else:
            analyzer_name = 'Газпром нефть'
        
        result = analyzer.analyzers[analyzer_name].analyze_mention(text)
        print(f"\nАнализ текста: {text[:50]}...")
        print(f"Релевантность: {result['relevant']}")
        print(f"Тональность: {result['sentiment']}")
        print(f"Теги: {result['tags']}")
        print(f"Опасно: {result['dangerous']}")


if __name__ == "__main__":
    main()