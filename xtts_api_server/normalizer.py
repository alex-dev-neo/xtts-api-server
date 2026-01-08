import sys
import warnings

# === ГЛУШИМ ВСЕ ПРЕДУПРЕЖДЕНИЯ ===
# Navec/Slovnet (компоненты Natasha) используют старые методы numpy, 
# которые вызывают RuntimeWarning на новых процессорах (Apple Silicon).
# Это безопасно игнорировать, так как результат корректен.
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    import os
    os.environ["PYTHONWARNINGS"] = "ignore"
# =================================

import re
import nltk
from num2words import num2words
from g2p_en import G2p
from natasha import (
    Segmenter, MorphVocab, NewsEmbedding, 
    NewsMorphTagger, NewsSyntaxParser, Doc
)

# Авто-загрузка ресурсов NLTK
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict', quiet=True)


class HybridNormalizer:
    def __init__(self):
        print("Загрузка моделей... (Warnings подавлены)")
        
        # 1. G2P
        self.g2p = G2p()
        
        # --- ТЮНИНГ ФОНЕМ (Русский акцент) ---
        self.phoneme_map = {
            # Гласные
            'AA': 'а', 'AE': 'а', 'AH': 'а', 'AO': 'о', 'AW': 'ау', 'AY': 'ай',
            'EH': 'е', 'ER': 'ер', 'EY': 'ей', 'IH': 'и', 'IY': 'и', 'OW': 'о',
            'OY': 'ой', 'UH': 'у', 'UW': 'у', 
            # Согласные
            'B': 'б', 'CH': 'ч', 'D': 'д', 'DH': 'з', 'F': 'ф', 'G': 'г',
            'HH': 'х', 'JH': 'дж', 'K': 'к', 'L': 'л', 'M': 'м', 'N': 'н',
            'NG': 'н', 'P': 'п', 'R': 'р', 'S': 'с', 'SH': 'ш', 'T': 'т',
            'TH': 'с', 'V': 'в', 'W': 'в', 'Y': 'й', 'Z': 'з', 'ZH': 'ж'
        }

        # 2. Natasha
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab() 
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        
        self.case_map = {
            'Nom': 'nominative', 'Gen': 'genitive', 'Dat': 'dative',
            'Acc': 'accusative', 'Ins': 'instrumental', 'Loc': 'prepositional'
        }
        self.gender_map = {'Masc': 'm', 'Fem': 'f', 'Neut': 'n'}

        # 3. СЛОВАРЬ ЗАМЕН (IT & Brands)
        self.hard_replacements = {
            # Технические термины и бренды
            r'\bOpenAI\b': 'Опен ЭйАй',
            r'\bChatGPT\b': 'Чат Джи-Пи-Ти',
            r'\bGPT\b': 'ДжиПиТи',
            r'\bGoogle\b': 'Гугл',
            r'\bNano\b': 'Нано',
            r'\bBanana\b': 'Банана',
            r'\bImages\b': 'Имиджес',
            r'\bWindows\b': 'Виндоус',
            r'\bMicrosoft\b': 'Майкрософт',
            r'\bAndroid\b': 'Андроид',
            r'\bApple\b': 'Эппл', 
            r'\biPhone\b': 'Айфон',
            r'\bOsmAnd\b': 'Османд',
            r'\biOS\b': 'Айос',
            r'\bOpenStreetMap\b': 'ОпенСтримМап',
            r'\bPM2\.5\b': 'пиэм дваипять',
            r'\bПМ2\.5\b': 'пиэм дваипять',
            r'\bрт ст\b': 'ртутного столба',
            r'\bрт\. ст\b': 'ртутного столба',
            r'\bрт\. ст\.\b': 'ртутного столба',

            r'\bDone\b': 'Готово',
            r'\bUnable to get response\b': 'Невозможно получить ответ',

            # Единицы и прочее
            r'\bUTC\b': 'Ю-Ти-Си',
            r'\bGMT\b': 'Джи-Эм-Ти',
            r'\bUSA\b': 'США',
            r'\bkm\b': 'километров', 
            r'\bкм\b': 'километров',
            r'\bmm\b': 'миллиметров', 
            r'\bмм\b': 'миллиметров', 
            r'\bm\b': 'метров', 
            r'\bм\b': 'метров', 
            r'\bkg\b': 'килограммов',
            r'\bкг\b': 'килограммов',
            r'\bг\b': 'граммов',
            r'\bг\.\b': 'года',
            r'\bкм/ч\b': 'километров в час',
            r'\bм/с\b': 'метров в секунду',
            r'гПа': 'гектопаскалей',
            r'°C': ' градусов Цельсия',
            r'°F': ' градусов Фаренгейта',
            # Латинская 'C' (часто приходит из внешних API или AI)
            r'\bCO₂\b': 'цеодв+а',
            r'\bCO2\b': 'цеодв+а',    
            # Кириллическая 'С' (если вы написали текст вручную на русской раскладке)
            r'\bСО₂\b': 'цеодв+а',
            r'\bСО2\b': 'цеодв+а',
            r'\bppm\b': 'пипиэм',
            r'мкг/м³': 'микрограмм на метр кубический',
            r'мкг/м3': 'микрограмм на метр кубический',
        }
        
        self.genitive_prepositions = {'от', 'до', 'из', 'без', 'у', 'для', 'вокруг', 'около'}
        self.distance_units = {'километр', 'метр', 'миля', 'километров', 'метров', 'миль', 'грамм', 'граммов', 'килограмм', 'килограммов'}
        self.months = {
            'январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
            'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь',
            'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
            'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря'
        }

    def _english_to_russian(self, match):
        word = match.group(0)
        phonemes = self.g2p(word)
        rus_word = []
        for p in phonemes:
            p_clean = re.sub(r'\d+', '', p) 
            if p_clean in self.phoneme_map:
                rus_word.append(self.phoneme_map[p_clean])
        return "".join(rus_word)

    def _replace_time(self, match):
        h, m = int(match.group(1)), int(match.group(2))
        h_txt = num2words(h, lang='ru', gender='m')
        
        if h % 10 == 1 and h != 11: h_end = "час"
        elif 2 <= h % 10 <= 4 and (h < 10 or h > 20): h_end = "часа"
        else: h_end = "часов"
        
        m_txt = num2words(m, lang='ru', gender='f')
        if m % 10 == 1 and m != 11: m_end = "минута"
        elif 2 <= m % 10 <= 4 and (m < 10 or m > 20): m_end = "минуты"
        else: m_end = "минут"
        
        if m == 0: return f"{h_txt} {h_end} ровно"
        return f"{h_txt} {h_end} {m_txt} {m_end}"

    def normalize(self, text):
        # 0. Форматирование списков: добавляем точку в конце и двойной перенос
        def fix_list_items(m):
            marker = m.group(1)
            content = m.group(2).strip()
            if not content:
                return m.group(0)
            # Добавляем точку, если нет пунктуации
            if not content.endswith(('.', '!', '?', ';', ':', '—', '-')):
                content += '.'
            return f"\n\n{marker}{content}\n\n"

        # 0.1 Обработка нумерованных списков и списков с тире
        # Ищем в начале строки или после переноса
        pattern = r'^\s*(\d+\.\s+|[-—–]\s+)(.+)$'
        text = re.sub(pattern, fix_list_items, text, flags=re.MULTILINE)
        
        # Очистка: убираем лишние переносы (максимум два подряд)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Убираем пробелы в начале строк (кроме тех, что в списке, хотя их мы уже причесали)
        text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
        text = text.strip()

        # 0. Обработка диапазонов (400-600 -> 400 до 600)
        # Поддерживаем разные типы тире и дефисов
        text = re.sub(r'(\d+)\s*[–—-]\s*(\d+)', r'\1 до \2', text)

        # 0.1 Обработка знаков перед числами (+12 -> плюс 12, -6 -> минус 6)
        # Используем пред-обработку, чтобы Natasha видела "плюс" и "минус" как отдельные слова
        text = re.sub(r'(^|\s)\+(\d+)', r'\1плюс \2', text)
        text = re.sub(r'(^|\s)-(\d+)', r'\1минус \2', text)

        # 1. Regex replacements (OpenAI, CO2 и т.д.)
        for pattern, replacement in self.hard_replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Обработка процентов (находит число и знак %)
        def replace_percent(m):
            num = int(m.group(1))
            # Генерируем текст числа
            num_txt = num2words(num, lang='ru')
            # Склоняем слово "процент" в зависимости от числа
            if num % 10 == 1 and num % 100 != 11:
                p_end = "процент"
            elif 2 <= num % 10 <= 4 and (num % 100 < 10 or num % 100 > 20):
                p_end = "процента"
            else:
                p_end = "процентов"
            return f"{num_txt} {p_end}"

        text = re.sub(r'(\d+)\s?%', replace_percent, text)

        text = re.sub(r'\b(\d{1,2}):(\d{2})\b', self._replace_time, text)
        
        def replace_float(m):
            parts = re.split(r'[.,]', m.group(0))
            w, f = int(parts[0]), int(parts[1][0])
            w_txt = num2words(w, lang='ru', gender='f')
            f_txt = num2words(f, lang='ru', gender='f')
            f_end = "десятая" if f == 1 else "десятых"
            return f"{w_txt} целых {f_txt} {f_end}"
        text = re.sub(r'\b\d+[.,]\d+\b', replace_float, text)
        
        text = re.sub(r'\b[A-Za-z]+\b', self._english_to_russian, text)

        # 2. Natasha Pipeline
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        doc.parse_syntax(self.syntax_parser)

        replacements = []
        
        for i, token in enumerate(doc.tokens):
            if token.text.isdigit():
                target_case = 'nominative'
                target_gender = 'm'
                is_ordinal = False 
                
                # Lookahead
                if i + 1 < len(doc.tokens):
                    next_token = doc.tokens[i+1]
                    lemma = next_token.lemma.lower() if next_token.lemma else next_token.text.lower()
                    
                    if lemma in ['год', 'г.']:
                        is_ordinal = True
                        target_case = self.case_map.get(next_token.feats.get('Case'), 'prepositional')
                    elif lemma in self.months:
                        is_ordinal = True
                        target_case = 'genitive'

                if i + 1 < len(doc.tokens):
                    next_token = doc.tokens[i+1]
                    next_lemma = next_token.lemma.lower() if next_token.lemma else next_token.text.lower()
                    if next_lemma in self.distance_units or next_lemma == 'грамм':
                        is_ordinal = False
                        target_case = 'nominative'

                # Natasha Context
                head = next((t for t in doc.tokens if t.id == token.head_id), None)
                if head and not is_ordinal:
                    natasha_case = self.case_map.get(head.feats.get('Case'), 'nominative')
                    target_gender = self.gender_map.get(head.feats.get('Gender'), 'm')
                    target_case = natasha_case
                    head_lemma = head.lemma.lower() if head.lemma else head.text.lower()

                    if i > 0 and doc.tokens[i-1].text.lower() == 'в' and head_lemma in self.distance_units:
                        target_case = 'prepositional'

                    if natasha_case == 'genitive':
                        is_prep = False
                        # Проверяем на 1 или 2 токена назад (из-за возможного "минус"/"плюс")
                        if i > 0 and doc.tokens[i-1].text.lower() in self.genitive_prepositions:
                            is_prep = True
                        elif i > 1 and doc.tokens[i-2].text.lower() in self.genitive_prepositions and \
                             doc.tokens[i-1].text.lower() in ['плюс', 'минус']:
                            is_prep = True
                            
                        if not is_prep:
                            target_case = 'nominative'

                    # ОСОБЫЙ ХАК ДЛЯ "ДО"
                    # Natasha иногда ошибается с синтаксической связью для "до", 
                    # форсируем родительный падеж, если перед числом (или знаком) стоит "до"
                    if i > 0 and doc.tokens[i-1].text.lower() == 'до':
                        target_case = 'genitive'
                    elif i > 1 and doc.tokens[i-2].text.lower() == 'до' and \
                         doc.tokens[i-1].text.lower() in ['плюс', 'минус']:
                        target_case = 'genitive'

                try:
                    to_type = 'ordinal' if is_ordinal else 'cardinal'
                    word_text = num2words(int(token.text), lang='ru', to=to_type, case=target_case, gender=target_gender)
                    replacements.append((token.start, token.stop, word_text))
                except:
                    pass

        for start, stop, rep in reversed(replacements):
            text = text[:start] + rep + text[stop:]

        return text

# --- ТЕСТ ---
if __name__ == "__main__":
    norm = HybridNormalizer()
    t = "На следующую неделю прогнозируется переменная погода: с морозами до -6 в начале недели и потеплением до +12°C к концу."
    t = "Глава OpenAI жмёт руку главе Nano Banana от Google. Ничего необычного — просто мы тестируем новую модель ChatGPT Images 1.5."

    print(f"IN:  {t}")
    print(f"OUT: {norm.normalize(t)}")