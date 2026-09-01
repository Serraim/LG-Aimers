# LG Aimers 9기 — 투구 제구 성공 확률 예측

LG Aimers 9기 Phase 2 해커톤 프로젝트. RandomForest 베이스라인(LB 549.64)에서
시작해 최종 LB 1095.23까지 끌어올린 과정을 정리했습니다.

- **기간**: 2026-08-06 ~ 2026-09-01
- **참여**: Serraim(이준호) + 팀원(baek-32), AI 페어프로그래밍(Claude/Codex/Gemini) 활용
- **최종 최고 제출**: V13 (Z1 평탄게이트, 자유도 1) — **LB 1095.23**
- **목표 대비**: 1150 목표, 실제 1095.23 (-54.77)

전체 회고는 [POSTMORTEM.md](POSTMORTEM.md)에, 실험 하나하나의 상세 원본 기록은
[EXPERIMENTS_LOG.md](EXPERIMENTS_LOG.md)(약 5,800줄)에 있습니다.

## 점수가 오른 과정 — 8단계

| # | 실험 | 무엇을 했나 | LB | 변화 |
|---|---|---|---|---|
| 1 | [E01](experiments/01_baseline_randomforest) | RandomForest 베이스라인 | 549.64 | — |
| 2 | [E02](experiments/02_lightgbm_switch) | LightGBM 전환 + 결측 native 처리 | 777.62 | **+204.87** |
| 3 | [S04](experiments/03_baseball_features) | 야구 도메인 파생 피처 15개 | 801.68 | +24.06 |
| 4 | [S06](experiments/04_three_model_ensemble) | LGBM+CatBoost+ExtraTrees 3모델 앙상블 | 809.02 | +7.34 |
| 5 | [S10](experiments/06_center_calibration_s10) | 중심오차 보정 상수(r, K) 도출 | 909.78 | **+100.76** |
| 6 | [S26](experiments/07_slot_decomposition_s26) / [E61](experiments/08_joint_outcome_e61) | 슬롯별 라벨 분해 + joint-outcome 모델링 (팀 합류) | — | 팀 트랙 병합 |
| 7 | [S67](experiments/09_first_full_stack_s67) | 게이트된 TabM 잔차 보정 + 전체 스택 첫 제출 | 1080.19 | — |
| 8 | [V8](experiments/10_physics_distillation_v8) | 물리(Trackman) 증류 개선 | 1089.95 | +9.76 |
| 9 | [V13](experiments/11_final_best_v13) | Z1 평탄게이트 (자유도 1) — **최종 최고** | **1095.23** | +5.28 |

**실패 사례도 남겨뒀습니다** (뭘 시도했고 왜 안 됐는지가 성공 사례만큼 중요하다고 생각해서):
- [V14](experiments/12_failed_multigate_v14) — 자유도 3개짜리 다중 게이트 스택. OOF에서 확신했던 조합이 실제 LB에서 역전(1090.89, V13 대비 -4.34). 다중 개입은 아무리 검증이 일치해도 한 번에 스택하지 않는다는 교훈.
- [Y1](experiments/13_epoch_sweep_y1) — TabM 에폭 스윕. "더 학습하면 나아진다"는 가설을 실측으로 기각.
- E08(시즌잔차 보정)은 [05_season_residual_e08](experiments/05_season_residual_e08)에서 S10의 기반이 된 트랙으로 확인할 수 있습니다.

## 무엇이 가장 크게 기여했나

```
가장 컸던 것       모델 교체(RF→LightGBM), 중심오차 보정, 게이트된 잔차 보정
꾸준히 기여한 것    변별력/중심오차 분리 사고, "로컬 vs LB 전달률"에 따른 우선순위
가장 배신한 것      다중 개입 스택(V14), 세밀한 피처 튜닝, 에폭 늘리기(Y1)
```

자세한 원인 분석과 5가지 일반화 가능한 교훈은 [POSTMORTEM.md](POSTMORTEM.md) 4장을
참고하세요.

## 폴더 구조

```
POSTMORTEM.md              전체 회고 — 여기부터 읽으면 됨
EXPERIMENTS_LOG.md          실험 원본 기록 (E01~S67 상세)
docs/                       배경 지식 (용어집, 컬럼 설명, 야구 지표, LB 활용 규칙)
experiments/                점수를 실제로 움직인 13개 전환점 — 코드 + 결과 + 제출 증거
```

## 대회 규칙

리더보드 활용 범위는 대회 운영진에 자진 문의해 확인받았습니다
([docs/RULES_LB_PROBING.md](docs/RULES_LB_PROBING.md) 참고). 행 독립성, 미래정보
금지 등 세부 규칙은 팀 저장소의 `AGENTS.md`에 별도로 정리되어 있습니다.
>>>>>>> origin/main
