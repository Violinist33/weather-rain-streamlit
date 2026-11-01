from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from prediction import predict_weather

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Чи буде дощ завтра?", page_icon="🌦️", layout="centered")


@st.cache_resource
def load_bundle(file_version):
    # Версія файла оновлює кеш після заміни моделі.
    return joblib.load(ROOT / "models" / "aussie_rain.joblib")


# Назва, мінімум, максимум, крок. Усі одиниці відповідають weatherAUS.
FIELDS = {
    "MinTemp": ("Мінімальна температура, °C", -30.0, 60.0, 0.1),
    "MaxTemp": ("Максимальна температура, °C", -30.0, 60.0, 0.1),
    "Temp9am": ("Температура о 09:00, °C", -30.0, 60.0, 0.1),
    "Temp3pm": ("Температура о 15:00, °C", -30.0, 60.0, 0.1),
    "Rainfall": ("Опади за добу, мм", 0.0, 500.0, 0.1),
    "Evaporation": ("Випаровування, мм", 0.0, 150.0, 0.1),
    "Sunshine": ("Сонячне сяйво, год", 0.0, 24.0, 0.1),
    "Humidity9am": ("Вологість о 09:00, %", 0.0, 100.0, 1.0),
    "Humidity3pm": ("Вологість о 15:00, %", 0.0, 100.0, 1.0),
    "Pressure9am": ("Тиск о 09:00, гПа", 850.0, 1100.0, 0.1),
    "Pressure3pm": ("Тиск о 15:00, гПа", 850.0, 1100.0, 0.1),
    "Cloud9am": ("Хмарність о 09:00, восьмі частки неба", 0.0, 8.0, 1.0),
    "Cloud3pm": ("Хмарність о 15:00, восьмі частки неба", 0.0, 8.0, 1.0),
    "WindGustSpeed": ("Найсильніший порив, км/год", 0.0, 250.0, 1.0),
    "WindSpeed9am": ("Швидкість вітру о 09:00, км/год", 0.0, 250.0, 1.0),
    "WindSpeed3pm": ("Швидкість вітру о 15:00, км/год", 0.0, 250.0, 1.0),
}

st.markdown("""
<style>
.stApp { background: #f5f8fc; }
.block-container { max-width: 960px; padding-top: 2.5rem; }
h1 { letter-spacing: -0.04em; }
[data-testid="stForm"] { background: white; border-radius: 18px; padding: 24px; }
[data-testid="stMetric"] { background: white; border: 1px solid #dce5ed;
    border-radius: 14px; padding: 18px; }
</style>
""", unsafe_allow_html=True)
st.caption("WEATHER LAB  /  АВСТРАЛІЯ")
st.title("Чи буде дощ завтра? 🌦️")
st.write("Введіть сьогоднішні спостереження за погодою та отримайте оцінку ймовірності дощу на наступний день.")

try:
    bundle = load_bundle((ROOT / "models" / "aussie_rain.joblib").stat().st_mtime_ns)
except FileNotFoundError:
    st.error("Файл моделі відсутній. Додайте models/aussie_rain.joblib до проєкту.")
    st.stop()

categories = dict(zip(bundle["categorical_cols"], bundle["encoder"].categories_))
example_path = ROOT / "example.json"
example = json.loads(example_path.read_text(encoding="utf-8")) if example_path.exists() else {}


def fill_example():
    for key in bundle["input_cols"]:
        value = example.get(key)
        st.session_state[key] = value if value is not None else (None if key in FIELDS else "Не знаю")
    st.session_state.pop("result", None)


def clear_form():
    for key in bundle["input_cols"]:
        st.session_state[key] = None if key in FIELDS else "Не знаю"
    st.session_state.pop("result", None)


left, right = st.columns(2)
left.button("Заповнити приклад", on_click=fill_example, use_container_width=True, disabled=not example)
right.button("Очистити поля", on_click=clear_form, use_container_width=True)
st.caption("Невідомі значення залиште порожніми або оберіть «Не знаю». Приклад — історичне спостереження, а не поточна погода.")

data = {}


def number(key):
    label, low, high, step = FIELDS[key]
    data[key] = st.number_input(label, low, high, value=None, step=step,
                                key=key, placeholder="Не знаю")


def category(key, label):
    options = ["Не знаю"] + [str(c) for c in categories[key] if pd.notna(c)]
    value = st.selectbox(label, options, key=key)
    data[key] = None if value == "Не знаю" else value


with st.form("weather_form"):
    category("Location", "Метеостанція в Австралії")
    tabs = st.tabs(["Температура й опади", "Вологість і небо", "Вітер"])
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            for key in ["MinTemp", "Temp9am", "Rainfall", "Evaporation"]:
                number(key)
        with col2:
            for key in ["MaxTemp", "Temp3pm", "Sunshine"]:
                number(key)
            category("RainToday", "Чи був дощ сьогодні? (Yes — так, No — ні)")
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            for key in ["Humidity9am", "Pressure9am", "Cloud9am"]:
                number(key)
        with col2:
            for key in ["Humidity3pm", "Pressure3pm", "Cloud3pm"]:
                number(key)
        st.caption("Хмарність: 0 — чисте небо, 8 — повністю вкрите хмарами.")
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            for key in ["WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"]:
                number(key)
        with col2:
            category("WindGustDir", "Напрям найсильнішого пориву")
            category("WindDir9am", "Напрям вітру о 09:00")
            category("WindDir3pm", "Напрям вітру о 15:00")
        st.caption("N — північ, S — південь, E — схід, W — захід; NE — північний схід.")
    submitted = st.form_submit_button("Отримати прогноз", type="primary", use_container_width=True)

if submitted:
    st.session_state.pop("result", None)
    if data["Location"] is None:
        st.warning("Оберіть метеостанцію.")
    elif not any(data.get(c) is not None for c in bundle["numeric_cols"]):
        st.warning("Вкажіть хоча б один числовий показник погоди.")
    elif (data["MinTemp"] is not None and data["MaxTemp"] is not None
          and data["MinTemp"] > data["MaxTemp"]):
        st.error("Мінімальна температура не може бути більшою за максимальну.")
    else:
        result = predict_weather(data, bundle)
        result["location"] = data["Location"]
        st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]
    st.subheader(f"Прогноз · {result['location']}")
    st.caption("Результат для останніх відправлених даних. Після змін натисніть «Отримати прогноз» ще раз.")
    if result["prediction"] == "Yes":
        st.info("🌧️ Прогнозується дощ завтра — Yes")
    else:
        st.success("🌤️ Дощ завтра не прогнозується — No")
    col1, col2 = st.columns(2)
    col1.metric("Ймовірність дощу (Yes)", f"{result['rain_probability']:.1%}")
    col2.metric(f"Ймовірність відповіді {result['prediction']}", f"{result['prediction_probability']:.1%}")
    st.progress(result["rain_probability"])
    if result["missing_count"]:
        st.caption(f"Невідомих показників: {result['missing_count']}. Числові пропуски заповнено середніми з навчання; невідомі категорії оброблено збереженим кодувальником.")

with st.expander("Як формується прогноз"):
    st.write("Введені дані → заповнення числових пропусків → масштабування → кодування категорій → модель → відповідь та ймовірність.")
    st.write("Обробники вже навчені. Під час прогнозування застосовується transform, повторного навчання на введених даних немає.")
    st.caption(f"Модель: {type(bundle['model']).__name__}. Дані: weatherAUS. Це навчальний прогноз на основі історичних спостережень; погодний сервіс у реальному часі не підключений.")

st.divider()
st.caption("Домашнє завдання · Деплоймент моделі зі Streamlit · Anton Lupandin")
