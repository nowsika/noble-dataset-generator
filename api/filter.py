#다른언어 섞인거 찾아내기
import json
import os

# 라이브러리가 없으면 자동으로 설치 시도하는 코드 (디자이너 맞춤 서비스!)
try:
    from langdetect import detect, DetectorFactory
except ImportError:
    print("📦 필요한 도구(langdetect)를 설치합니다...")
    os.system('pip install langdetect')
    from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

input_file = "outputs.jsonl"          # 원본 파일명 (꼭 일치해야 함!)
output_clean = "english_only.jsonl"   # 영어 결과 파일
output_others = "multi_lang.jsonl"    # 기타 언어 파일

print(f"🚀 '{input_file}' 분류를 시작합니다...")

matched = 0
others = 0

try:
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_clean, 'w', encoding='utf-8') as f_en, \
         open(output_others, 'w', encoding='utf-8') as f_other:

        for line in f_in:
            if not line.strip(): continue
            
            data = json.loads(line)
            # 질문(scenario)과 답변(model_response_A) 텍스트 가져오기
            q_text = data.get('scenario', "")
            a_text = data.get('model_response_A', {}).get('raw_essay', "")
            
            try:
                # 언어 감지
                lang_q = detect(q_text)
                lang_a = detect(a_text)
                
                # 둘 다 영어('en')인 경우만 통과
                if lang_q == 'en' and lang_a == 'en':
                    f_en.write(json.dumps(data, ensure_ascii=False) + '\n')
                    matched += 1
                else:
                    f_other.write(json.dumps(data, ensure_ascii=False) + '\n')
                    others += 1
                    print(f"   [제외됨] ID {data.get('id')}: {lang_q} -> {lang_a}")
            except:
                # 에러나면 기타 파일로
                f_other.write(json.dumps(data, ensure_ascii=False) + '\n')
                others += 1

    print("-" * 30)
    print("✅ 작업 끝!")
    print(f"🇺🇸 영어 파일: {matched}개 저장됨 -> {output_clean}")
    print(f"🌍 기타 언어: {others}개 저장됨 -> {output_others}")
    print("-" * 30)

except FileNotFoundError:
    print(f"❌ '{input_file}' 파일을 못 찾겠어요! 파일 이름이 맞는지 확인해주세요.")