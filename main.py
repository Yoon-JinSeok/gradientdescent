import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from PIL import Image

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Linear Regression Trend Analysis",
    page_icon="📈",
    layout="wide",
)

rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# ──────────────────────────────────────────────
# 헬퍼: 컬럼명이 ASCII 가 아니면 fallback 사용 (그래프 라벨용)
# ──────────────────────────────────────────────
def safe_label(name: str, fallback: str) -> str:
    try:
        str(name).encode("ascii")
        return str(name)
    except UnicodeEncodeError:
        return fallback

# ──────────────────────────────────────────────
# 세션 상태 초기화
# ──────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 1

def goto(step: int):
    st.session_state.step = max(st.session_state.step, step)

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

current = st.session_state.step  # 현재 진행 중인 단계 번호

# ==============================================
# STEP 1. 데이터 수집
# ==============================================
if current >= 1:
    with st.container(border=True):
        st.header("1단계 · 데이터 수집")
        st.write("CSV / Excel 파일을 업로드하세요. (한글 인코딩 자동 감지)")

        file = st.file_uploader("파일 업로드", type=["csv", "xlsx", "xls"], key="uploader")

        if file is not None and "df" not in st.session_state:
            df = None
            try:
                if file.name.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(file)
                else:
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

        if "df" in st.session_state:
            df = st.session_state.df

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

            if current == 1:
                st.button("➡️ 2단계로 이동", on_click=goto, args=(2,), key="to2")

# ==============================================
# STEP 2. 결측치 제거
# ==============================================
if current >= 2:
    with st.container(border=True):
        st.header("2단계 · 결측치 제거")
        df = st.session_state.get("df")

        st.subheader("📋 데이터 구성 정보 (df.info)")
        buf = io.StringIO()
        df.info(buf=buf)
        st.code(buf.getvalue())

        st.subheader("🕳️ 컬럼별 결측치 개수")
        st.dataframe(
            df.isna().sum().to_frame("결측치 개수"),
            use_container_width=True,
        )

        if "df_clean" not in st.session_state:
            if st.button("결측치 제거 실행 (dropna)", key="do_dropna"):
                st.session_state.df_clean = df.dropna().copy()
                st.rerun()

        if "df_clean" in st.session_state:
            df_clean = st.session_state.df_clean
            st.success(f"제거 전 {df.shape[0]:,}행 → 제거 후 {df_clean.shape[0]:,}행")

            st.subheader("✅ 결측치 제거 후 정보")
            buf2 = io.StringIO()
            df_clean.info(buf=buf2)
            st.code(buf2.getvalue())

            if current == 2:
                st.button("➡️ 3단계로 이동", on_click=goto, args=(3,), key="to3")

# ==============================================
# STEP 3. 변수 선택
# ==============================================
if current >= 3:
    with st.container(border=True):
        st.header("3단계 · 독립변수와 종속변수 선택")
        df_clean = st.session_state.get("df_clean")
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

        if len(num_cols) < 2:
            st.error("숫자형 컬럼이 2개 이상 필요합니다.")
        else:
            # 이미 선택했다면 그 값을 기본값으로
            default_x = st.session_state.get("Xname", num_cols[0])
            if default_x not in num_cols:
                default_x = num_cols[0]

            c1, c2 = st.columns(2)
            with c1:
                Xname = st.selectbox(
                    "독립변수 (X)", num_cols,
                    index=num_cols.index(default_x), key="X_select"
                )
            y_options = [c for c in num_cols if c != Xname]
            default_y = st.session_state.get("Yname", y_options[0])
            if default_y not in y_options:
                default_y = y_options[0]
            with c2:
                Yname = st.selectbox(
                    "종속변수 (Y)", y_options,
                    index=y_options.index(default_y), key="Y_select"
                )

            st.info(f"Xname : **{Xname}**  /  Yname : **{Yname}**")
            st.dataframe(df_clean[[Xname, Yname]], use_container_width=True)

            # 변수 확정
            if "Xname" not in st.session_state or "Yname" not in st.session_state:
                if st.button("✅ 변수 확정", key="confirm_var"):
                    st.session_state.Xname = Xname
                    st.session_state.Yname = Yname
                    st.rerun()
            else:
                # 변경 가능
                if (Xname != st.session_state.Xname) or (Yname != st.session_state.Yname):
                    if st.button("🔁 선택 변경 적용 (이후 단계 초기화)", key="change_var"):
                        # 이후 단계 결과 초기화
                        for k in ["df_norm", "Xmax", "Xmin", "Ymax", "Ymin",
                                  "x_train", "x_test", "y_train", "y_test",
                                  "a_init", "b_init", "a_final", "b_final",
                                  "mse_train_list", "mse_test_list"]:
                            st.session_state.pop(k, None)
                        st.session_state.Xname = Xname
                        st.session_state.Yname = Yname
                        st.session_state.step = 3
                        st.rerun()

                if current == 3:
                    st.button("➡️ 4단계로 이동", on_click=goto, args=(4,), key="to4")

# ==============================================
# STEP 4. 데이터 정규화
# ==============================================
if current >= 4:
    with st.container(border=True):
        st.header("4단계 · 데이터 정규화 (Min-Max)")
        df_clean = st.session_state.df_clean
        Xname = st.session_state.Xname
        Yname = st.session_state.Yname

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

        if current == 4:
            st.button("➡️ 5단계로 이동", on_click=goto, args=(5,), key="to5")

# ==============================================
# STEP 5. 데이터 분할
# ==============================================
if current >= 5:
    with st.container(border=True):
        st.header("5단계 · 학습 / 테스트 데이터 분할")
        df_norm = st.session_state.df_norm
        Xname = st.session_state.Xname
        Yname = st.session_state.Yname

        c1, c2 = st.columns(2)
        test_size = c1.slider("테스트 데이터 비율", 0.1, 0.5, 0.3, 0.05, key="test_size")
        random_state = c2.number_input("random_state", value=42, step=1, key="rs")

        if "x_train" not in st.session_state:
            if st.button("데이터 분할 실행", key="do_split"):
                x_train, x_test, y_train, y_test = train_test_split(
                    df_norm[Xname], df_norm[Yname],
                    test_size=test_size, random_state=int(random_state),
                )
                st.session_state.update(
                    x_train=x_train, x_test=x_test,
                    y_train=y_train, y_test=y_test,
                )
                st.rerun()

        if "x_train" in st.session_state:
            x_train = st.session_state.x_train
            x_test = st.session_state.x_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test

            st.success("분할 완료")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("x_train", f"{x_train.shape[0]:,}")
            c2.metric("x_test", f"{x_test.shape[0]:,}")
            c3.metric("y_train", f"{y_train.shape[0]:,}")
            c4.metric("y_test", f"{y_test.shape[0]:,}")

            if current == 5:
                st.button("➡️ 6단계로 이동", on_click=goto, args=(6,), key="to6")

# ==============================================
# STEP 6. 초기 추세선
# ==============================================
if current >= 6:
    with st.container(border=True):
        st.header("6단계 · 초기 추세선 그리기")

        c1, c2 = st.columns(2)
        a = c1.number_input("기울기 a (초깃값)", value=0.0, step=0.1, key="a_init_input")
        b = c2.number_input("y절편 b (초깃값)", value=0.0, step=0.1, key="b_init_input")

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

        if current == 6:
            st.button("➡️ 7단계로 이동", on_click=goto, args=(7,), key="to7")

# ==============================================
# STEP 7. 손실함수 계산
# ==============================================
if current >= 7:
    with st.container(border=True):
        st.header("7단계 · 손실함수(MSE) 계산")

        a = st.session_state.get("a_init", 0.0)
        b = st.session_state.get("b_init", 0.0)
        x = st.session_state.x_train
        y = st.session_state.y_train

        st.latex(r"L(y, \hat{y}) = \frac{1}{n}\sum (y - \hat{y})^2")

        y_hat = a * x + b
        mse = float(np.mean((y - y_hat) ** 2))
        st.metric("초기 추세선의 MSE", f"{mse:.6f}")
        st.caption(f"(a = {a}, b = {b} 기준)")

        if current == 7:
            st.button("➡️ 8단계로 이동", on_click=goto, args=(8,), key="to8")

# ==============================================
# STEP 8. 경사하강법
# ==============================================
if current >= 8:
    with st.container(border=True):
        st.header("8단계 · 경사하강법으로 최적 추세선 찾기")

        c1, c2, c3, c4 = st.columns(4)
        eta = c1.number_input("학습률 eta", value=0.02, step=0.01,
                              format="%.4f", key="eta")
        epochs = c2.number_input("반복 횟수", value=2001, step=100,
                                 min_value=1, key="epochs")
        log_every = c3.number_input("로그 간격", value=100, step=50,
                                    min_value=1, key="log_every")
        frame_every = c4.number_input("GIF 프레임 간격", value=50, step=10,
                                      min_value=1, key="frame_every",
                                      help="작을수록 부드럽지만 GIF 용량이 커집니다")

        gif_duration = st.slider(
            "프레임 1장당 표시 시간 (ms)", 30, 500, 120, 10, key="gif_dur"
        )

        if "a_final" not in st.session_state:
            if st.button("🚀 학습 시작 (GIF 생성 포함)", key="do_train"):
                x = st.session_state.x_train.values
                y = st.session_state.y_train.values
                x_test_v = st.session_state.x_test.values
                y_test_v = st.session_state.y_test.values

                Xname = st.session_state.Xname
                Yname = st.session_state.Yname
                x_lbl = safe_label(Xname, "X")
                y_lbl = safe_label(Yname, "Y")

                a_, b_ = 0.0, 0.0
                mse_train_list, mse_test_list = [], []
                logs = []
                frames = []  # GIF 프레임 누적

                # 그래프 축 범위 고정 (애니메이션 떨림 방지)
                x_min, x_max = float(x.min()), float(x.max())
                y_min, y_max = float(y.min()), float(y.max())
                x_pad = (x_max - x_min) * 0.05
                y_pad = (y_max - y_min) * 0.1
                xs = np.linspace(x_min, x_max, 100)

                progress = st.progress(0, text="학습 + GIF 프레임 캡처 중...")

                total = int(epochs)
                for i in range(total):
                    y_hat = a_ * x + b_
                    grad_a = -2 * np.mean((y - y_hat) * x)
                    grad_b = -2 * np.mean(y - y_hat)
                    a_ = a_ - eta * grad_a
                    b_ = b_ - eta * grad_b

                    mse = float(np.mean((y - (a_ * x + b_)) ** 2))
                    mse_t = float(np.mean((y_test_v - (a_ * x_test_v + b_)) ** 2))
                    mse_train_list.append(mse)
                    mse_test_list.append(mse_t)

                    if i % int(log_every) == 0:
                        logs.append((i, mse, a_, b_))

                    # GIF 프레임 캡처
                    if (i % int(frame_every) == 0) or (i == total - 1):
                        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
                        ax.scatter(x, y, alpha=0.4, s=20, label="Train data")
                        ax.plot(xs, a_ * xs + b_, color="red", lw=2,
                                label=f"y = {a_:.4f}x + {b_:.4f}")
                        ax.set_xlim(x_min - x_pad, x_max + x_pad)
                        ax.set_ylim(y_min - y_pad, y_max + y_pad)
                        ax.set_xlabel(x_lbl)
                        ax.set_ylabel(y_lbl)
                        ax.set_title(f"Epoch {i}  |  MSE = {mse:.6f}")
                        ax.legend(loc="upper left")
                        ax.grid(alpha=0.3)
                        fig.tight_layout()

                        buf = io.BytesIO()
                        fig.savefig(buf, format="png")
                        plt.close(fig)
                        buf.seek(0)
                        # PIL Image 로 변환 후 메모리에 보관
                        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

                    if i % max(1, total // 100) == 0:
                        progress.progress(
                            min(1.0, (i + 1) / total),
                            text=f"학습 진행 {i+1}/{total} (프레임 {len(frames)}장)",
                        )

                progress.progress(1.0, text="GIF 합성 중...")

                # GIF 합성
                gif_buf = io.BytesIO()
                if len(frames) >= 2:
                    frames[0].save(
                        gif_buf,
                        format="GIF",
                        save_all=True,
                        append_images=frames[1:],
                        duration=int(gif_duration),
                        loop=0,
                        optimize=True,
                        disposal=2,
                    )
                else:
                    # 프레임이 1장뿐이면 단일 이미지로 저장
                    frames[0].save(gif_buf, format="GIF")

                gif_bytes = gif_buf.getvalue()

                st.session_state.update(
                    a_final=a_, b_final=b_,
                    mse_train_list=mse_train_list,
                    mse_test_list=mse_test_list,
                    train_logs=logs,
                    train_gif=gif_bytes,
                )
                progress.empty()
                st.rerun()

        if "a_final" in st.session_state:
            a_ = st.session_state.a_final
            b_ = st.session_state.b_final
            logs = st.session_state.get("train_logs", [])

            st.success(f"학습 완료 → a = {a_:.6f}, b = {b_:.6f}")

            # 🎬 학습 과정 GIF
            if "train_gif" in st.session_state:
                st.subheader("🎬 학습 과정 애니메이션 (GIF)")
                gif_bytes = st.session_state.train_gif
                st.image(gif_bytes, caption="경사하강법으로 추세선이 갱신되는 과정",
                         use_container_width=True)
                st.download_button(
                    label="⬇️ GIF 다운로드",
                    data=gif_bytes,
                    file_name="gradient_descent.gif",
                    mime="image/gif",
                )
                st.caption(f"GIF 용량: {len(gif_bytes) / 1024:.1f} KB")

            # 학습 로그
            st.subheader("📜 학습 로그")
            log_df = pd.DataFrame(logs, columns=["i", "MSE", "a", "b"])
            st.dataframe(log_df, use_container_width=True)

            # 최종 추세선 (정적)
            x = st.session_state.x_train.values
            y = st.session_state.y_train.values
            Xname = st.session_state.Xname
            Yname = st.session_state.Yname
            x_lbl = safe_label(Xname, "X")
            y_lbl = safe_label(Yname, "Y")

            st.subheader("📈 최종 추세선")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x, y, alpha=0.5, label="Train data")
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, a_ * xs + b_, color="r",
                    label=f"y = {a_:.4f}x + {b_:.4f}")
            ax.set_xlabel(x_lbl)
            ax.set_ylabel(y_lbl)
            ax.set_title("Final Trend Line")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            # 재학습 옵션
            if st.button("🔁 다시 학습 (파라미터 변경 후)", key="retrain"):
                for k in ["a_final", "b_final", "mse_train_list",
                          "mse_test_list", "train_logs", "train_gif"]:
                    st.session_state.pop(k, None)
                st.rerun()

            if current == 8:
                st.button("➡️ 9단계로 이동", on_click=goto, args=(9,), key="to9")


# ==============================================
# STEP 9. 모델 평가
# ==============================================
if current >= 9:
    with st.container(border=True):
        st.header("9단계 · 모델 평가")

        a_ = st.session_state.a_final
        b_ = st.session_state.b_final
        x_test = st.session_state.x_test
        y_test = st.session_state.y_test
        mse_train_list = st.session_state.mse_train_list
        mse_test_list = st.session_state.mse_test_list

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

        y_pred = a_ * x_test + b_
        mean_y = np.mean(y_test)
        ssr = np.sum((y_test - y_pred) ** 2)
        sst = np.sum((y_test - mean_y) ** 2)
        r2 = 1 - ssr / sst

        st.metric("결정계수 R²", f"{r2:.6f}")

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

        if current == 9:
            st.button("➡️ 10단계로 이동", on_click=goto, args=(10,), key="to10")

# ==============================================
# STEP 10. 활용 (예측)
# ==============================================
if current >= 10:
    with st.container(border=True):
        st.header("10단계 · 새로운 데이터로 예측")

        a_ = st.session_state.a_final
        b_ = st.session_state.b_final
        Xmax = st.session_state.Xmax
        Xmin = st.session_state.Xmin
        Ymax = st.session_state.Ymax
        Ymin = st.session_state.Ymin
        Xname = st.session_state.Xname
        Yname = st.session_state.Yname

        st.write(f"학습된 모델 (정규화 공간): y = **{a_:.6f}** · x + **{b_:.6f}**")
        st.write(f"X 범위: [{Xmin:.4f}, {Xmax:.4f}] / Y 범위: [{Ymin:.4f}, {Ymax:.4f}]")

        Xnew = st.number_input(
            f"새로운 {Xname} 값을 입력하세요",
            value=float((Xmin + Xmax) / 2),
            format="%.6f",
            key="Xnew",
        )

        Xnew_scal = (Xnew - Xmin) / (Xmax - Xmin)
        Ypred_norm = a_ * Xnew_scal + b_
        Ynew = (Ymax - Ymin) * Ypred_norm + Ymin

        st.success(
            f"**{Xname}** 이(가) **{Xnew}** 일 때, "
            f"**{Yname}** 은(는) **{Ynew:.6f}** (으)로 예측됩니다."
        )
