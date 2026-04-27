# -*- coding: utf-8 -*-
"""
추세선 기반 선형 회귀(예측) 분석 - Streamlit 웹앱
제작 : 윤진석
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib

# 한글 폰트 지원 (Streamlit Cloud에서도 동작)
try:
    import koreanize_matplotlib  # noqa: F401
except Exception:
    # 한글 폰트가 없을 경우 기본값으로 진행 (마이너스 깨짐 방지)
    matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# 페이지 기본 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="추세선 기반 선형 회귀(예측) 분석",
    page_icon="📈",
    layout="wide",
)

st.title("📈 추세선 기반 선형 회귀(예측) 분석")
st.caption("제작 : 윤진석")

# ─────────────────────────────────────────────────────────────
# 세션 상태 초기화
# ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "step": 1,           # 현재 진행 단계
        "df": None,          # 원본 데이터
        "df_clean": None,    # 결측치 제거 데이터
        "Xname": None,
        "Yname": None,
        "Xmax": None, "Xmin": None,
        "Ymax": None, "Ymin": None,
        "df_norm": None,
        "x_train": None, "x_test": None,
        "y_train": None, "y_test": None,
        "a": 0.0, "b": 0.0,
        "eta": 0.02,
        "iters": 2001,
        "MSE_train_list": [],
        "MSE_test_list": [],
        "trained": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────
# 사이드바 : 진행 상황 표시 + 초기화
# ─────────────────────────────────────────────────────────────
STEPS = [
    "1단계 : 데이터 수집",
    "2단계 : 결측치 제거",
    "3단계 : 독립/종속 변수 선택",
    "4단계 : 데이터 정규화",
    "5단계 : 데이터 분할",
    "6단계 : 추세선 그리기",
    "7단계 : 손실함수(오차) 계산",
    "8단계 : 경사하강법 학습",
    "9단계 : 모델 평가",
    "10단계 : 활용(예측)",
]

with st.sidebar:
    st.header("진행 단계")
    for i, name in enumerate(STEPS, start=1):
        if i < st.session_state.step:
            st.markdown(f"✅ {name}")
        elif i == st.session_state.step:
            st.markdown(f"👉 **{name}**")
        else:
            st.markdown(f"⬜ {name}")
    st.divider()
    if st.button("🔄 처음부터 다시 시작"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ─────────────────────────────────────────────────────────────
# 공통 시각화 함수
# ─────────────────────────────────────────────────────────────
def plot_scatter_with_line(x, y, y_hat, xname, yname, title=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.6, label="실제 데이터")
    # 직선이 깔끔하게 보이도록 x 정렬
    order = np.argsort(np.array(x))
    ax.plot(np.array(x)[order], np.array(y_hat)[order], color="red", label="추세선")
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    if title:
        ax.set_title(title)
    ax.legend()
    return fig


# ─────────────────────────────────────────────────────────────
# 1단계 : 데이터 수집
# ─────────────────────────────────────────────────────────────
if st.session_state.step >= 1:
    st.header("1단계 : 데이터 수집")
    st.write("CSV 또는 Excel(.xlsx) 파일을 업로드하세요. (한글 인코딩 자동 처리)")

    uploaded = st.file_uploader(
        "파일 업로드 (CSV / XLSX)",
        type=["csv", "xlsx", "xls"],
        key="uploader",
    )

    if uploaded is not None:
        try:
            file_name = uploaded.name.lower()
            if file_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded)
            else:
                # CSV: 한글 인코딩 자동 시도
                raw = uploaded.read()
                df = None
                for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]:
                    try:
                        df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                        st.success(f"인코딩 '{enc}'(으)로 정상 읽기 성공")
                        break
                    except Exception:
                        continue
                if df is None:
                    st.error("CSV를 읽을 수 없습니다. 파일 인코딩을 확인하세요.")
                    st.stop()

            st.session_state.df = df

            st.subheader("📄 데이터 상위 10개")
            st.dataframe(df.head(10))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("행(row) 개수", df.shape[0])
                st.metric("열(column) 개수", df.shape[1])
            with col2:
                st.subheader("속성명 및 타입")
                dtype_df = pd.DataFrame({
                    "속성명": df.columns,
                    "타입": df.dtypes.astype(str).values,
                })
                st.dataframe(dtype_df, use_container_width=True)

            if st.button("➡️ 2단계로 이동", key="to2"):
                st.session_state.step = max(st.session_state.step, 2)
                st.rerun()

        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")


# ─────────────────────────────────────────────────────────────
# 2단계 : 결측치 제거
# ─────────────────────────────────────────────────────────────
if st.session_state.step >= 2 and st.session_state.df is not None:
    st.header("2단계 : 결측치 제거")
    df = st.session_state.df

    st.subheader("데이터 구성 정보 (df.info)")
    buf = io.StringIO()
    df.info(buf=buf)
    st.code(buf.getvalue())

    st.subheader("결측치 개수")
    st.dataframe(df.isna().sum().rename("결측치 개수").to_frame())

    if st.button("🧹 결측치 제거 실행", key="dropna"):
        df_clean = df.dropna().copy()
        st.session_state.df_clean = df_clean

    if st.session_state.df_clean is not None:
        st.success(f"결측치 제거 완료 : {st.session_state.df.shape[0]} → {st.session_state.df_clean.shape[0]} 행")
        buf2 = io.StringIO()
        st.session_state.df_clean.info(buf=buf2)
        st.code(buf2.getvalue())
        st.dataframe(st.session_state.df_clean.head(10))

        if st.button("➡️ 3단계로 이동", key="to3"):
            st.session_state.step = max(st.session_state.step, 3)
            st.rerun()


# ─────────────────────────────────────────────────────────────
# 3단계 : 독립변수와 종속변수 선택 (숫자형만)
# ─────────────────────────────────────────────────────────────
if st.session_state.step >= 3 and st.session_state.df_clean is not None:
    st.header("3단계 : 독립변수(X) / 종속변수(Y) 선택")

    df_clean = st.session_state.df_clean
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("숫자형 컬럼이 2개 이상 필요합니다.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            Xname = st.selectbox("독립변수 X 선택 (숫자형)", numeric_cols, key="x_sel")
        with col2:
            y_options = [c for c in numeric_cols if c != Xname]
            Yname = st.selectbox("종속변수 Y 선택 (숫자형)", y_options, key="y_sel")

        st.info(f"선택 결과 → Xname : **{Xname}**  /  Yname : **{Yname}**")
        st.dataframe(df_clean[[Xname, Yname]].head(20))

        if st.button("➡️ 4단계로 이동", key="to4"):
            st.session_state.Xname = Xname
            st.session_state.Yname = Yname
            st.session_state.step = max(st.session_state.step, 4)
            st.rerun()


# ─────────────────────────────────────────────────────────────
# 4단계 : 미니맥스 정규화
# ─────────────────────────────────────────────────────────────
if st.session_state.step >= 4 and st.session_state.Xname is not None:
    st.header("4단계 : 데이터 정규화 (Min-Max Scaling)")

    df_clean = st.session_state.df_clean
    Xname = st.session_state.Xname
    Yname = st.session_state.Yname

    if st.button("🧮 정규화 실행", key="do_norm"):
        Xmax = df_clean[Xname].max()
        Xmin = df_clean[Xname].min()
        Ymax = df_clean[Yname].max()
        Ymin = df_clean[Yname].min()

        df_norm = pd.DataFrame()
        df_norm[Xname] = (df_clean[Xname] - Xmin) / (Xmax - Xmin)
        df_norm[Yname] = (df_clean[Yname] - Ymin) / (Ymax - Ymin)

        st.session_state.Xmax, st.session_state.Xmin = Xmax, Xmin
        st.session_state.Ymax, st.session_state.Ymin = Ymax, Ymin
        st.session_state.df_norm = df_norm

    if st.session_state.df_norm is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{Xname} max", f"{st.session_state.Xmax:.4f}")
        c2.metric(f"{Xname} min", f"{st.session_state.Xmin:.4f}")
        c3.metric(f"{Yname} max", f"{st.session_state.Ymax:.4f}")
        c4.metric(f"{Yname} min", f"{st.session_state.Ymin:.4f}")

        st.subheader("정규화된 데이터")
        st.dataframe(st.session_state.df_norm.head(20))

        if st.button("➡️ 5단계로 이동", key="to5"):
            st.session_state.step = max(st.session_state.step, 5)
            st.rerun()


# ─────────────────────────────────────────────────────────────
# 5단계 : 데이터 분할
# ─────────────────────────────────────────────────────────────
if st.session_state.step >= 5 and st.session_state.df_norm is not None:
    st.header("5단계 : 데이터 분할 (Train / Test)")

    test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.3, 0.05)
    random_state = st.number_input("random_state", value=42, step=1)

    if st.button("✂️ 데이터 분할 실행", key="do_split"):
        Xname = st.session_state.Xname
        Yname = st.session_state.Yname
        df_norm = st.session_state.df_norm
        x_train, x_test, y_train, y_test = train_test_split(
            df_norm[Xname], df_norm[Yname],
            test_size=test_size, random_state=int(random_state),
        )
        st.session_state.x_train = x_train
        st.session_state.x_test = x_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

    if st.session_state.x_train is not None:
        st.success(
            f"x_train: {st.session_state.x_train.shape}, "
            f"x_test: {st.session_state.x_test.shape}, "
            f"y_train: {st.session_state.y_train.shape}, "
            f"y_test: {st.session_state.y_test.shape}"
        )
        if st.button("➡️ 6단계로 이동", key="to6"):
            st.session_state.step = max(st.session_state.step, 6)
            st.rerun()


# ─────────────────────────────────────────────────────────────
# 6단계 : 추세선 그리기 (초깃값)
# ─────────────────────────────────────────────────────────────
if st.session_state.step >= 6 and st.session_state.x_train is not None:
    st.header("6단계 : 추세선 그리기 (초기값)")

    col1, col2 = st.columns(2)
    with col1:
        a0 = st.number_input("기울기 a 초깃값", value=0.0, step=0.1, format="%.4f")
    with col2:
        b0 = st.number_input("y절편 b 초깃값", value=0.0, step=0.1, format="%.4f")

    st.latex(f"y = {a0} \\cdot x + {b0}")

    if st.button("📊 추세선 시각화", key="show_init_line"):
        st.session_state.a = float(a0)
        st.session_state.b = float(b0)
        x = st.session_state.x_train
        y = st.session_state.y_train
        y_hat = st.session_state.a * x + st.session_state.b
        fig = plot_scatter_with_line(
            x, y, y_hat,
            st.session_state.Xname, st.session_state.Yname,
            title="초기 추세선",
        )
        st.pyplot(fig)

        if st.button("➡️ 7단계로 이동", key="to7"):
            st.session_state.step = max(st.session_state.step, 7)
            st.rerun()

    # 이미 시각화 후에도 다음 단계 진행 버튼을 남겨둠
    if st.session_state.step == 6:
        if st.button("➡️ 7단계로 이동 (스킵)", key="to7_skip"):
            st.session_state.step = 7
            st.rerun()


# ─────────────────────────────────────────────────────────────
# 7단계 : 손실함수(MSE)로 오차 계산
# ─────────────────────────────────────────────────────────────
def mse_loss(y, y_hat):
    return float(np.mean((y - y_hat) ** 2))

if st.session_state.step >= 7 and st.session_state.x_train is not None:
    st.header("7단계 : 손실함수(MSE)를 이용한 오차 계산")
    st.latex(r"L(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2")

    if st.button("📐 현재 추세선의 MSE 계산", key="calc_mse"):
        x = st.session_state.x_train
        y = st.session_state.y_train
        y_hat = st.session_state.a * x + st.session_state.b
        mse = mse_loss(y, y_hat)
        st.success(f"MSE = {mse:.6f}  (a={st.session_state.a}, b={st.session_state.b})")

    if st.button("➡️ 8단계로 이동", key="to8"):
        st.session_state.step = max(st.session_state.step, 8)
        st.rerun()


# ─────────────────────────────────────────────────────────────
# 8단계 : 경사하강법으로 최적 추세선 구하기
# ─────────────────────────────────────────────────────────────
def gd_step(x, y, a, b, eta):
    y_hat = a * x + b
    grad_a = -2 * np.mean((y - y_hat) * x)
    grad_b = -2 * np.mean(y - y_hat)
    a_new = a - eta * grad_a
    b_new = b - eta * grad_b
    return a_new, b_new

if st.session_state.step >= 8 and st.session_state.x_train is not None:
    st.header("8단계 : 경사하강법으로 최적 추세선 학습")

    col1, col2, col3 = st.columns(3)
    with col1:
        eta = st.number_input("학습률 eta", value=0.02, step=0.01, format="%.4f")
    with col2:
        iters = st.number_input("반복 횟수", value=2001, step=100, min_value=1)
    with col3:
        snapshot_every = st.number_input("그래프 표시 간격", value=100, step=50, min_value=1)

    if st.button("🚀 학습 시작", key="train"):
        st.session_state.eta = float(eta)
        st.session_state.iters = int(iters)

        x = st.session_state.x_train.values if hasattr(st.session_state.x_train, "values") else st.session_state.x_train
        y = st.session_state.y_train.values if hasattr(st.session_state.y_train, "values") else st.session_state.y_train
        x_test = st.session_state.x_test.values if hasattr(st.session_state.x_test, "values") else st.session_state.x_test
        y_test = st.session_state.y_test.values if hasattr(st.session_state.y_test, "values") else st.session_state.y_test

        a, b = 0.0, 0.0
        train_losses, test_losses = [], []

        progress = st.progress(0)
        log_box = st.empty()
        plot_box = st.empty()
        logs = []

        total = int(iters)
        for i in range(total):
            a, b = gd_step(x, y, a, b, st.session_state.eta)
            tr_mse = mse_loss(y, a * x + b)
            te_mse = mse_loss(y_test, a * x_test + b)
            train_losses.append(tr_mse)
            test_losses.append(te_mse)

            if i % int(snapshot_every) == 0 or i == total - 1:
                logs.append(f"i={i}, MSE={tr_mse:.6f}, a={a:.6f}, b={b:.6f}")
                log_box.code("\n".join(logs[-15:]))  # 마지막 15줄
                fig = plot_scatter_with_line(
                    x, y, a * x + b,
                    st.session_state.Xname, st.session_state.Yname,
                    title=f"i={i}, a={a:.4f}, b={b:.4f}",
                )
                plot_box.pyplot(fig)
                plt.close(fig)

            progress.progress((i + 1) / total)

        st.session_state.a = float(a)
        st.session_state.b = float(b)
        st.session_state.MSE_train_list = train_losses
        st.session_state.MSE_test_list = test_losses
        st.session_state.trained = True
        st.success(f"학습 완료! 최종 a={a:.6f}, b={b:.6f}")

    if st.session_state.trained:
        if st.button("➡️ 9단계로 이동", key="to9"):
            st.session_state.step = max(st.session_state.step, 9)
            st.rerun()


# ─────────────────────────────────────────────────────────────
# 9단계 : 모델 평가
# ─────────────────────────────────────────────────────────────
def r2_score_manual(y, pred):
    y = np.array(y); pred = np.array(pred)
    mean = np.mean(y)
    ssr = np.sum((y - pred) ** 2)
    sst = np.sum((y - mean) ** 2)
    return 1 - ssr / sst

if st.session_state.step >= 9 and st.session_state.trained:
    st.header("9단계 : 모델 평가")

    # (1) 손실함수 그래프
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(st.session_state.MSE_train_list, label="학습데이터")
    ax1.plot(st.session_state.MSE_test_list, label="테스트데이터")
    ax1.set_xlabel("학습횟수")
    ax1.set_ylabel("손실함수값")
    ax1.set_title("학습/테스트 손실함수 변화")
    ax1.legend()
    st.pyplot(fig1)

    # (2) 결정계수 R^2
    a = st.session_state.a
    b = st.session_state.b
    x_test = st.session_state.x_test
    y_test = st.session_state.y_test
    y_pred = a * x_test + b

    r2 = r2_score_manual(y_test, y_pred)
    st.metric("결정계수 R²", f"{r2:.6f}")

    # (3) 예측값 vs 실제값
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(y_test, y_pred, color="blue", alpha=0.6, label="예측값")
    lo = float(min(np.min(y_test), np.min(y_pred)))
    hi = float(max(np.max(y_test), np.max(y_pred)))
    ax2.plot([lo, hi], [lo, hi], "r--", lw=2, label="완벽한 예측선")
    ax2.set_xlabel("실제값")
    ax2.set_ylabel("예측값")
    ax2.set_title("예측값 vs 실제값")
    ax2.legend()
    st.pyplot(fig2)

    if st.button("➡️ 10단계로 이동", key="to10"):
        st.session_state.step = max(st.session_state.step, 10)
        st.rerun()


# ─────────────────────────────────────────────────────────────
# 10단계 : 활용 (새로운 데이터로 예측)
# ─────────────────────────────────────────────────────────────
if st.session_state.step >= 10 and st.session_state.trained:
    st.header("10단계 : 활용 (새로운 데이터로 예측)")

    Xname = st.session_state.Xname
    Yname = st.session_state.Yname
    Xmax, Xmin = st.session_state.Xmax, st.session_state.Xmin
    Ymax, Ymin = st.session_state.Ymax, st.session_state.Ymin
    a, b = st.session_state.a, st.session_state.b

    st.write(f"학습된 모델: **y = {a:.6f} · x + {b:.6f}**  (정규화 공간 기준)")

    Xnew = st.number_input(
        f"새로운 {Xname} 값을 입력하세요",
        value=float((Xmax + Xmin) / 2),
        format="%.6f",
    )

    if st.button("🔮 예측하기", key="predict"):
        Xnew_scal = (Xnew - Xmin) / (Xmax - Xmin)
        Ypred_scal = a * Xnew_scal + b
        Ynew = (Ymax - Ymin) * Ypred_scal + Ymin
        st.success(f"{Xname} 이(가) **{Xnew}** 일 때, "
                   f"{Yname} 은(는) **{Ynew:.6f}** 으로 예측됩니다.")

    st.balloons()
