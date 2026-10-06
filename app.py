import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
import plotly.express as px

st.set_page_config(page_title="ORION - XGBoost Trading", layout="wide")
st.title("⚡ ORION — XGBoost Quant Trading System")

# ==========================================
# 1. FUNÇÃO DE DOWNLOAD E ENGENHARIA DE FEATURES
# ==========================================
@st.cache_data
def load_and_build_features(ticker: str, start_date: str, end_date: str):
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        return df
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    data = df.copy()

    # --- TENDÊNCIA (EMA 8 e EMA 21) ---
    data['ema_8'] = data['Close'].ewm(span=8, adjust=False).mean()
    data['ema_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['dist_ema_8'] = (data['Close'] - data['ema_8']) / data['Close']
    data['dist_ema_21'] = (data['Close'] - data['ema_21']) / data['Close']
    data['ema_spread'] = (data['ema_8'] - data['ema_21']) / data['ema_21']

    # --- MOMENTUM (RSI 14) ---
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data['rsi_14'] = 100 - (100 / (1 + rs))

    # --- VOLATILIDADE (ATR 14) ---
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    data['atr_14'] = atr
    data['atr_pct'] = atr / data['Close']
    data['vol_ratio'] = atr / (atr.rolling(50).mean() + 1e-9)

    # --- TARGET (Alta nos próximos 3 períodos acima de 0.5%) ---
    FORWARD_PERIODS = 3
    THRESHOLD = 0.005
    future_return = data['Close'].shift(-FORWARD_PERIODS) / data['Close'] - 1
    data['target'] = (future_return > THRESHOLD).astype(int)

    return data.dropna()

# ==========================================
# 2. BARRA LATERAL (PARÂMETROS)
# ==========================================
st.sidebar.header("Configuração do Ativo")
ticker = st.sidebar.text_input("Ticker (Ex: PETR4.SA, VALE3.SA, BTC-USD)", value="PETR4.SA")
start_date = st.sidebar.date_input("Data Inicial", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Data Final", value=pd.to_datetime("2026-10-01"))

st.sidebar.header("Parâmetros do Modelo & Risco")
proba_threshold = st.sidebar.slider("Probabilidade Mínima de Entrada", 0.50, 0.85, 0.60, 0.05)
atr_multiplier = st.sidebar.slider("Stop Loss (Multiplicador ATR)", 1.0, 3.0, 1.5, 0.1)
risk_reward = st.sidebar.slider("Relação Risco / Retorno", 1.0, 3.0, 2.0, 0.5)

# ==========================================
# 3. EXECUÇÃO
# ==========================================
if st.sidebar.button("Rodar ORION System"):
    with st.spinner("Processando dados e treinando o XGBoost..."):
        df_data = load_and_build_features(ticker, str(start_date), str(end_date))

        if df_data.empty:
            st.error("Não foram encontrados dados para este Ticker/Intervalo.")
        else:
            feature_cols = ['dist_ema_8', 'dist_ema_21', 'ema_spread', 'rsi_14', 'atr_pct', 'vol_ratio']
            X = df_data[feature_cols]
            y = df_data['target']

            # Divisão Out-of-Time (80% Treino / 20% Teste)
            split_idx = int(len(df_data) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)

            # Predição no conjunto de teste
            df_test = df_data.iloc[split_idx:].copy()
            df_test['proba_alta'] = model.predict_proba(X_test)[:, 1]

            # --- BACKTEST ---
            trades = []
            in_pos = False
            entry, stop, take = 0, 0, 0

            for i in range(len(df_test)):
                row = df_test.iloc[i]
                if in_pos:
                    if row['Low'] <= stop:
                        pnl = (stop - entry) / entry
                        trades.append({'Exit Date': row.name, 'Return': pnl, 'Outcome': 'Stop Loss'})
                        in_pos = False
                    elif row['High'] >= take:
                        pnl = (take - entry) / entry
                        trades.append({'Exit Date': row.name, 'Return': pnl, 'Outcome': 'Take Profit'})
                        in_pos = False

                if not in_pos and row['proba_alta'] >= proba_threshold:
                    in_pos = True
                    entry = row['Close']
                    atr = row['atr_14']
                    stop = entry - (atr * atr_multiplier)
                    take = entry + (atr * atr_multiplier * risk_reward)

            df_trades = pd.DataFrame(trades)

            # --- EXIBIÇÃO DE RESULTADOS ---
            last_proba = df_test['proba_alta'].iloc[-1]
            st.subheader("Sinal Atual do Mercado")
            col1, col2, col3 = st.columns(3)
            col1.metric("Último Preço Fechamento", f"R$ {df_test['Close'].iloc[-1]:.2f}")
            col2.metric("Probabilidade de Alta (XGBoost)", f"{last_proba*100:.1f}%")
            
            if last_proba >= proba_threshold:
                col3.success("🟢 SINAL DE COMPRA ATIVO")
            else:
                col3.info("⚪ AGUARDAR FORA DO MERCADO")

            st.divider()

            if not df_trades.empty:
                win_rate = (df_trades['Outcome'] == 'Take Profit').mean() * 100
                cum_return = (1 + df_trades['Return']).prod() - 1
                
                st.subheader("Resultados do Backtest (Fora da Amostra / Teste)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Total de Trades Disparados", len(df_trades))
                m2.metric("Taxa de Acerto (Win Rate)", f"{win_rate:.1f}%")
                m3.metric("Retorno Acumulado", f"{cum_return*100:.2f}%")

                df_trades['Equity'] = (1 + df_trades['Return']).cumprod()
                fig_eq = px.line(df_trades, x='Exit Date', y='Equity', title="Evolução do Patrimônio")
                st.plotly_chart(fig_eq, use_container_width=True)
            else:
                st.warning("Nenhum trade disparado no período de teste com o limiar selecionado.")

            st.subheader("Importância das Variáveis no Modelo")
            imp = pd.Series(model.feature_importances_, index=feature_cols).reset_index()
            imp.columns = ['Indicador', 'Importância']
            fig_imp = px.bar(imp, x='Importância', y='Indicador', orientation='h')
            st.plotly_chart(fig_imp, use_container_width=True)
