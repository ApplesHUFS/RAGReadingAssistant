# ReadRAG

텍스트 파일 기반 RAG(Retrieval-Augmented Generation) 시스템입니다. 텍스트 파일을 청크 단위로 분할하고, 의미 검색을 통해 관련된 내용을 찾아 GPT로 답변을 생성합니다.

## 주요 기능

- 텍스트 파일 처리 및 청크 분할
- 의미 기반 검색 (Semantic Search)
- GPT를 활용한 질의응답
- 텍스트 요약 기능

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/ReadRAG.git
cd ReadRAG
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

3. OpenAI API 키 설정 
```bash
# .env 파일에 API 키 입력
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## 디렉토리 구조

```
ReadRAG/
├── readrag/                # 메인 패키지
│   ├── config/            # 설정
│   ├── core/              # 핵심 기능
│   ├── interfaces/        # 사용자 인터페이스
│   ├── services/          # 주요 서비스
│   └── utils/             # 유틸리티
│
├── data/                  # 데이터 저장소
│   ├── books/            # 원본 텍스트 파일
│   └── processed/        # 처리된 데이터
│
└── main.py               # 실행 파일
```

## 사용 방법

1. 프로그램 실행
```bash
python main.py
```

2. 메인 메뉴 옵션
- 새로운 텍스트 파일 추가
- 기존 파일 선택
- 종료

3. 파일 처리 후 가능한 작업
- 질문하기: 텍스트 내용에 대해 질문
- 요약하기: 전체 내용 요약

## 주요 기술 스택

- Python 3.8+
- sentence-transformers: 텍스트 임베딩
- FAISS: 벡터 유사도 검색
- OpenAI GPT: 답변 생성
- KoBART: 한국어 텍스트 요약

## 필요 사항

- Python 3.8 이상
- OpenAI API 키
- 최소 8GB RAM
- (선택사항) CUDA 지원 GPU

## 참고사항

- 텍스트 파일은 UTF-8 인코딩이어야 합니다
- 파일 크기는 10MB 이하를 권장합니다
- GPU가 있는 경우 자동으로 활용됩니다

## License

MIT License

## 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
