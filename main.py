import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Linear Regression Trend Analysis",
    page_icon="📈",
    layout="wide",
)

# 마이너스 기호 깨짐만 방지 (한글 폰트는 사용하지 않음)
rcParams["axes.unicode_minus"] = False

# ──────────────────────────────────────────────
# 헬퍼: 컬럼명을 안전한 ASCII 라벨로 변환
# (그래프 라벨에 한글 컬럼명이 들어가도 깨지지 않게)
# ──────────────────────────────────────────────
def safe_label(name: str, fallback: str) -> str:
    """컬럼명에 ASCII 외 문자가 있으면 fallback 반환"""
    try:
        name.encode("ascii")
        return str(name)
    except UnicodeEncodeError:
        return fallback

# ──────────────────────────────────────────────
# 세션 상태 초기화
# ──────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 1

def goto(step: int):
    st.session_state.step = step

# ──────────────────────────────────────────────
# 사이드바: 진행 상황
# ──────────────────────────────────────────────
st.sidebar.title("📊 진행 단계")
steps = [
    "1. 데이터 수집",
    "2. 결측치 제거",
    "3. 변수 선택",
    "4. 데이터 정규화",
    "5. 데이터 분할",
    "6. 추세선 그리기",
    "7. 손실함수 계산",
    "8. 경사하강법",
    "9. 모델 평가",
    "10. 활용(예측)",
]
for i, s in enumerate(steps, start=1):
    if i < st.session_state.step:
        st.sidebar.markdown(f"✅ {s}")
    elif i == st.session_state.step:
        st.sidebar.markdown(f"👉 **{s}**")
    else:
        st.sidebar.markdown(f"⬜ {s}")

st.sidebar.divider()
if st.sidebar.button("🔄 처음부터 다시 시작"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# ──────────────────────────────────────────────
# 메인 타이틀
# ──────────────────────────────────────────────
st.title("📈 추세선 기반 선형 회귀(예측) 분석")
st.caption("제작 : 윤진석")
st.divider()

# ==============================================
# STEP 1. 데이터 수집
# ==============================================
if st.session_state.step == 1:
    st.header("1단계 · 데이터 수집")
    st.write("CSV / Excel 파일을 업로드하세요. (한글 인코딩 자동 감지)")

    file = st.file_uploader("파일 업로드", type=["csv", "xlsx", "xls"])

    if file is not None:
        df = None
        try:
            if file.name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(file)
            else:
                # 여러 인코딩 자동 시도
                raw = file.read()
                for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]:
                    try:
                        df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                        st.success(f"✅ 인코딩 `{enc}` 으로 로드 완료")
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    st.error("지원되는 인코딩으로 파일을 읽지 못했습니다.")
        except Exception as e:
            st.error(f"파일 로드 오류: {e}")

        if df is not None:
            st.session_state.df = df

            st.subheader("📄 상위 10개 행")
            st.dataframe(df.head(10), use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("행 개수", f"{df.shape[0]:,}")
            c2.metric("열 개수", f"{df.shape[1]:,}")

            st.subheader("🧬 속성명 및 타입")
            info_df = pd.DataFrame({
                "속성명": df.columns,
                "타입": [str(t) for t in df.dtypes],
            })
            st.dataframe(info_df, use_container_width=True)

            st.button("➡️ 2단계로 이동", on_click=goto, args=(2,))

# ==============================================
# STEP 2. 결측치 제거
# ==============================================
elif st.session_state.step == 2:
    st.header("2단계 · 결측치 제거")
    df = st.session_state.get("df")
    if df is None:
        st.warning("1단계에서 먼저 데이터를 업로드해 주세요.")
        st.button("⬅️ 1단계로", on_click=goto, args=(1,))
    else:
        st.subheader("📋 데이터 구성 정보 (df.info)")
        buf = io.StringIO()
        df.info(buf=buf)
        st.code(buf.getvalue())

        st.subheader("🕳️ 컬럼별 결측치 개수")
        st.dataframe(
            df.isna().sum().to_frame("결측치 개수"),
            use_container_width=True,
        )

        if st.button("결측치 제거 실행 (dropna)"):
            df_clean = df.dropna().copy()
            st.session_state.df_clean = df_clean
            st.success(f"제거 전 {df.shape[0]:,}행 → 제거 후 {df_clean.shape[0]:,}행")

        if "df_clean" in st.session_state:
            st.subheader("✅ 결측치 제거 후 정보")
            buf2 = io.StringIO()
            st.session_state.df_clean.info(buf=buf2)
            st.code(buf2.getvalue())
            st.button("➡️ 3단계로 이동", on_click=goto, args=(3,))

# ==============================================
# STEP 3. 독립변수 / 종속변수 선택
# ==============================================
elif st.session_state.step == 3:
    st.header("3단계 · 독립변수와 종속변수 선택")
    df_clean = st.session_state.get("df_clean")
    if df_clean is None:
        st.warning("2단계를 먼저 진행해 주세요.")
        st.button("⬅️ 2단계로", on_click=goto, args=(2,))
    else:
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            st.error("숫자형 컬럼이 2개 이상 필요합니다.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                Xname = st.selectbox("독립변수 (X)", num_cols, index=0)
            with c2:
                y_options = [c for c in num_cols if c != Xname]
                Yname = st.selectbox("종속변수 (Y)", y_options, index=0)

            st.info(f"Xname : **{Xname}**  /  Yname : **{Yname}**")
            st.dataframe(df_clean[[Xname, Yname]], use_container_width=True)

            if st.button("✅ 변수 확정 후 4단계로"):
                st.session_state.Xname = Xname
                st.session_state.Yname = Yname
                goto(4)
                st.rerun()

# ==============================================
# STEP 4. 데이터 정규화 (Min-Max)
# ==============================================
elif st.session_state.step == 4:
    st.header("4단계 · 데이터 정규화 (Min-Max)")
    df_clean = st.session_state.get("df_clean")
    Xname = st.session_state.get("Xname")
    Yname = st.session_state.get("Yname")
    if not (Xname and Yname):
        st.warning("3단계를 먼저 진행해 주세요.")
        st.button("⬅️ 3단계로", on_click=goto, args=(3,))
    else:
        Xmax, Xmin = df_clean[Xname].max(), df_clean[Xname].min()
        Ymax, Ymin = df_clean[Yname].max(), df_clean[Yname].min()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Xmax", f"{Xmax:.4f}")
        c2.metric("Xmin", f"{Xmin:.4f}")
        c3.metric("Ymax", f"{Ymax:.4f}")
        c4.metric("Ymin", f"{Ymin:.4f}")

        df_norm = pd.DataFrame()
        df_norm[Xname] = (df_clean[Xname] - Xmin) / (Xmax - Xmin)
        df_norm[Yname] = (df_clean[Yname] - Ymin) / (Ymax - Ymin)

        st.subheader("🧮 정규화 결과 (0 ~ 1)")
        st.dataframe(df_norm, use_container_width=True)

        st.session_state.update(
            Xmax=Xmax, Xmin=Xmin, Ymax=Ymax, Ymin=Ymin, df_norm=df_norm
        )
        st.button("➡️ 5단계로 이동", on_click=goto, args=(5,))

# ==============================================
# STEP 5. 데이터 분할
# ==============================================
elif st.session_state.step == 5:
    st.header("5단계 · 학습 / 테스트 데이터 분할")
    df_norm = st.session_state.get("df_norm")
    Xname = st.session_state.get("Xname")
    Yname = st.session_state.get("Yname")
    if df_norm is None:
        st.warning("4단계를 먼저 진행해 주세요.")
        st.button("⬅️ 4단계로", on_click=goto, args=(4,))
    else:
        test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.3, 0.05)
        random_state = st.number_input("random_state", value=42, step=1)

        if st.button("데이터 분할 실행"):
            x_train, x_test, y_train, y_test = train_test_split(
                df_norm[Xname], df_norm[Yname],
                test_size=test_size, random_state=int(random_state),
            )
            st.session_state.update(
                x_train=x_train, x_test=x_test,
                y_train=y_train, y_test=y_test,
            )
            st.success("분할 완료")

        if "x_train" in st.session_state:
            x_train = st.session_state.x_train
            x_test = st.session_state.x_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("x_train", f"{x_train.shape[0]:,}")
            c2.metric("x_test", f"{x_test.shape[0]:,}")
            c3.metric("y_train", f"{y_train.shape[0]:,}")
            c4.metric("y_test", f"{y_test.shape[0]:,}")
            st.button("➡️ 6단계로 이동", on_click=goto, args=(6,))

# ==============================================
# STEP 6. 추세선 그리기 (초기값)
# ==============================================
elif st.session_state.step == 6:
    st.header("6단계 · 초기 추세선 그리기")
    if "x_train" not in st.session_state:
        st.warning("5단계를 먼저 진행해 주세요.")
        st.button("⬅️ 5단계로", on_click=goto, args=(5,))
    else:
        c1, c2 = st.columns(2)
        a = c1.number_input("기울기 a (초깃값)", value=0.0, step=0.1)
        b = c2.number_input("y절편 b (초깃값)", value=0.0, step=0.1)

        st.latex(f"y = {a} \\cdot x + {b}")

        x = st.session_state.x_train
        y = st.session_state.y_train
        y_hat = a * x + b

        Xname = st.session_state.Xname
        Yname = st.session_state.Yname
        x_lbl = safe_label(Xname, "X")
        y_lbl = safe_label(Yname, "Y")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, alpha=0.5, label="Actual")
        ax.plot(x, y_hat, color="r", label="Trend line")
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)
        ax.set_title("Initial Trend Line")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.session_state.a_init = a
        st.session_state.b_init = b
        st.button("➡️ 7단계로 이동", on_click=goto, args=(7,))

# ==============================================
# STEP 7. 손실함수 계산 (MSE)
# ==============================================
elif st.session_state.step == 7:
    st.header("7단계 · 손실함수(MSE) 계산")
    if "x_train" not in st.session_state:
        st.warning("이전 단계를 먼저 진행해 주세요.")
    else:
        a = st.session_state.get("a_init", 0.0)
        b = st.session_state.get("b_init", 0.0)
        x = st.session_state.x_train
        y = st.session_state.y_train

        st.latex(r"L(y, \hat{y}) = \frac{1}{n}\sum (y - \hat{y})^2")

        y_hat = a * x + b
        mse = float(np.mean((y - y_hat) ** 2))
        st.metric("초기 추세선의 MSE", f"{mse:.6f}")
        st.caption(f"(a = {a}, b = {b} 기준)")

        st.button("➡️ 8단계로 이동", on_click=goto, args=(8,))

# ==============================================
# STEP 8. 경사하강법
# ==============================================
elif st.session_state.step == 8:
    st.header("8단계 · 경사하강법으로 최적 추세선 찾기")
    if "x_train" not in st.session_state:
        st.warning("이전 단계를 먼저 진행해 주세요.")
    else:
        c1, c2, c3 = st.columns(3)
        eta = c1.number_input("학습률 eta", value=0.02, step=0.01, format="%.4f")
        epochs = c2.number_input("반복 횟수", value=2001, step=100, min_value=1)
        log_every = c3.number_input("로그 간격", value=100, step=50, min_value=1)

        if st.button("🚀 학습 시작"):
            x = st.session_state.x_train.values
            y = st.session_state.y_train.values
            x_test_v = st.session_state.x_test.values
            y_test_v = st.session_state.y_test.values

            a, b = 0.0, 0.0
            mse_train_list, mse_test_list = [], []
            logs = []

            Xname = st.session_state.Xname
            Yname = st.session_state.Yname
            x_lbl = safe_label(Xname, "X")
            y_lbl = safe_label(Yname, "Y")

            progress = st.progress(0)
            log_box = st.container()

            for i in range(int(epochs)):
                y_hat = a * x + b
                grad_a = -2 * np.mean((y - y_hat) * x)
                grad_b = -2 * np.mean(y - y_hat)
                a = a - eta * grad_a
                b = b - eta * grad_b

                mse = float(np.mean((y - (a * x + b)) ** 2))
                mse_t = float(np.mean((y_test_v - (a * x_test_v + b)) ** 2))
                mse_train_list.append(mse)
                mse_test_list.append(mse_t)

                if i % int(log_every) == 0:
                    logs.append((i, mse, a, b))

                if i % max(1, int(epochs) // 100) == 0:
                    progress.progress(min(1.0, (i + 1) / int(epochs)))

            progress.progress(1.0)

            with log_box:
                st.subheader("📜 학습 로그")
                log_df = pd.DataFrame(logs, columns=["i", "MSE", "a", "b"])
                st.dataframe(log_df, use_container_width=True)

                st.subheader("📈 최종 추세선")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(x, y, alpha=0.5, label="Train data")
                xs = np.linspace(x.min(), x.max(), 100)
                ax.plot(xs, a * xs + b, color="r", label=f"y = {a:.4f}x + {b:.4f}")
                ax.set_xlabel(x_lbl)
                ax.set_ylabel(y_lbl)
                ax.set_title("Final Trend Line")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

            st.session_state.update(
                a_final=a, b_final=b,
                mse_train_list=mse_train_list,
                mse_test_list=mse_test_list,
            )
            st.success(f"학습 완료 → a = {a:.6f}, b = {b:.6f}")

        if "a_final" in st.session_state:
            st.button("➡️ 9단계로 이동", on_click=goto, args=(9,))

# ==============================================
# STEP 9. 모델 평가
# ==============================================
elif st.session_state.step == 9:
    st.header("9단계 · 모델 평가")
    if "a_final" not in st.session_state:
        st.warning("8단계를 먼저 진행해 주세요.")
        st.button("⬅️ 8단계로", on_click=goto, args=(8,))
    else:
        a = st.session_state.a_final
        b = st.session_state.b_final
        x_test = st.session_state.x_test
        y_test = st.session_state.y_test
        mse_train_list = st.session_state.mse_train_list
        mse_test_list = st.session_state.mse_test_list

        # 손실함수 곡선
        st.subheader("📉 Loss Curve")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(mse_train_list, label="Train")
        ax1.plot(mse_test_list, label="Test")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (MSE)")
        ax1.set_title("Training vs Test Loss")
        ax1.legend()
        st.pyplot(fig1)
        plt.close(fig1)

        # 결정계수
        y_pred = a * x_test + b
        mean_y = np.mean(y_test)
        ssr = np.sum((y_test - y_pred) ** 2)
        sst = np.sum((y_test - mean_y) ** 2)
        r2 = 1 - ssr / sst

        st.metric("결정계수 R²", f"{r2:.6f}")

        # 예측 vs 실제
        st.subheader("🎯 Predicted vs Actual")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.scatter(y_test, y_pred, color="blue", alpha=0.6, label="Predicted")
        lo = float(min(y_test.min(), y_pred.min()))
        hi = float(max(y_test.max(), y_pred.max()))
        ax2.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect prediction")
        ax2.set_xlabel("Actual")
        ax2.set_ylabel("Predicted")
        ax2.set_title("Predicted vs Actual")
        ax2.legend()
        st.pyplot(fig2)
        plt.close(fig2)

        st.button("➡️ 10단계로 이동", on_click=goto, args=(10,))

# ==============================================
# STEP 10. 활용 (예측)
# ==============================================
elif st.session_state.step == 10:
    st.header("10단계 · 새로운 데이터로 예측")
    if "a_final" not in st.session_state:
        st.warning("8단계를 먼저 진행해 주세요.")
        st.button("⬅️ 8단계로", on_click=goto, args=(8,))
    else:
        a = st.session_state.a_final
        b = st.session_state.b_final
        Xmax = st.session_state.Xmax
        Xmin = st.session_state.Xmin
        Ymax = st.session_state.Ymax
        Ymin = st.session_state.Ymin
        Xname = st.session_state.Xname
        Yname = st.session_state.Yname

        st.write(f"학습된 모델 (정규화 공간): y = **{a:.6f}** · x + **{b:.6f}**")
        st.write(f"X 범위: [{Xmin:.4f}, {Xmax:.4f}] / Y 범위: [{Ymin:.4f}, {Ymax:.4f}]")

        Xnew = st.number_input(
            f"새로운 {Xname} 값을 입력하세요",
            value=float((Xmin + Xmax) / 2),
            format="%.6f",
        )

        if st.button("🔮 예측 실행"):
            Xnew_scal = (Xnew - Xmin) / (Xmax - Xmin)
            Ypred_norm = a * Xnew_scal + b
            Ynew = (Ymax - Ymin) * Ypred_norm + Ymin
            st.success(
                f"**{Xname}** 이(가) **{Xnew}** 일 때, "
                f"**{Yname}** 은(는) **{Ynew:.6f}** (으)로 예측됩니다."
            )
