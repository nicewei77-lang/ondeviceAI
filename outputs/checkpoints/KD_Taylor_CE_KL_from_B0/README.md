# CUB Student KD 체크포인트

B-0와 Taylor CE KL로 압축한 B-30/B-50/B-70 각각 seed42/43/44를 학습한 결과다. 각 하위 폴더에 config.json, history.json, metrics.json, artifacts.json이 있다.
가중치는 artifacts.json의 정확한 Drive 파일을 내려받아 해당 폴더의 best.pt로 보관한다. 이 파일은 Native Student checkpoint이며 model_state, model_spec, model_digest, contract를 사용한다. 파일 이름 변경은 직렬화 형식 변환을 뜻하지 않는다.
history.json의 accuracy는0~1, metrics.json의 accuracy_pct는0~100이다. selected epoch는1부터 표시한다. 마지막 재개 checkpoint는 별도 링크로 보존한다.
전체 결과와 다운로드 목록은 docs/results/cub_taylor_ce_kl_from_b0를 참고한다.
