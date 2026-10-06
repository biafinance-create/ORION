import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
import plotly.express as px

st.set_page_config(page_title="ORION - XGBoost Trading System", layout="wide")
st.title("⚡ ORION — XGBoost Quant Trading System")

# ==========================================
# LISTA DE AÇÕES DA CARTEIRA IBOVESPA
# ==========================================
IBOV_TICKERS = {
    "PETR4.SA": "PETR4.SA - PETROBRAS PN",
    "PETR3.SA": "PETR3.SA - PETROBRAS ON",
    "VALE3.SA": "VALE3.SA - VALE ON",
    "ITUB4.SA": "ITUB4.SA - ITAÚ UNIBANCO PN",
    "BBDC4.SA": "BBDC4.SA - BRADESCO PN",
    "BBDC3.SA": "BBDC3.SA - BRADESCO ON",
    "BBAS3.SA": "BBAS3.SA - BANCO DO BRASIL ON",
    "B3SA3.SA": "B3SA3.SA - B3 ON",
    "ABEV3.SA": "ABEV3.SA - AMBEV S/A ON",
    "WEGE3.SA": "WEGE3.SA - WEG ON",
    "RENT3.SA": "RENT3.SA - LOCALIZA ON",
    "PRIO3.SA": "PRIO3.SA - PRIO ON",
    "SUZB3.SA": "SUZB3.SA - SUZANO ON",
    "EQTL3.SA": "EQTL3.SA - EQUATORIAL ON",
    "GGBR4.SA": "GGBR4.SA - GERDAU PN",
    "ITSA4.SA": "ITSA4.SA - ITAÚSA PN",
    "SBSP3.SA": "SBSP3.SA - SABESP ON",
    "BPAC11.SA": "BPAC11.SA - BTG PACTUAL UNT",
    "EMBJ3.SA": "EMBJ3.SA - EMBRAER ON",
    "ENEV3.SA": "ENEV3.SA - ENEVA ON",
    "RDOR3.SA": "RDOR3.SA - REDE D'OR ON",
    "LREN3.SA": "LREN3.SA - LOJAS RENNER ON",
    "RADL3.SA": "RADL3.SA - RAIADROGASIL ON",
    "BBSE3.SA": "BBSE3.SA - BB SEGURIDADE ON",
    "UGPA3.SA": "UGPA3.SA - ULTRAPAR ON",
    "HAPV3.SA": "HAPV3.SA - HAPVIDA ON",
    "VBBR3.SA": "VBBR3.SA - VIBRA ON",
    "RAIL3.SA": "RAIL3.SA - RUMO ON",
    "CPLE3.SA": "CPLE3.SA - COPEL ON",
    "CMIG4.SA": "CMIG4.SA - CEMIG PN",
    "CSAN3.SA": "CSAN3.SA - COSAN ON",
    "ELET3.SA": "AXIA3.SA - AXIA ENERGIA ON",
    "TOTS3.SA": "TOTS3.SA - TOTVS ON",
    "VIVT3.SA": "VIVT3.SA - TELEFÔNICA BRASIL ON",
    "TIMS3.SA": "TIMS3.SA - TIM ON",
    "SANB11.SA": "SANB11.SA - SANTANDER UNT",
    "CCRO3.SA": "MOTV3.SA - MOTIVA ON",
    "TAEE11.SA": "TAEE11.SA - TAESA UNT",
    "CSNA3.SA": "CSNA3.SA - SIDENÚRGICA NACIONAL ON",
    "BRAP4.SA": "BRAP4.SA - BRADESPAR PN",
    "ALOS3.SA": "ALOS3.SA - ALLOS ON",
    "ASAI3.SA": "ASAI3.SA - ASSAÍ ON",
    "AURE3.SA": "AURE3.SA - AUREN ON",
    "AZZA3.SA": "AZZA3.SA - AZZAS 2154 ON",
    "BRAV3.SA": "BRAV3.SA - BRAVA ON",
    "CXSE3.SA": "CXSE3.SA - CAIXA SEGURIDADE ON",
    "CEAB3.SA": "CEAB3.SA - CEA MODAS ON",
    "COGN3.SA": "COGN3.SA - COGNA ON",
    "CSMG3.SA": "CSMG3.SA - COPASA ON",
    "CPFE3.SA": "CPFE3.SA - CPFL ENERGIA ON",
    "CMIN3.SA": "CMIN3.SA - CSN MINERAÇÃO ON",
    "CURY3.SA": "CURY3.SA - CURY ON",
    "CYRE3.SA": "CYRE3.SA - CYRELA ON",
    "DIRR3.SA": "DIRR3.SA - DIRECIONAL ON",
    "ENGI11.SA": "ENGI11.SA - ENERGISA UNT",
    "EGIE3.SA": "EGIE3.SA - ENGIE BRASIL ON",
    "FLRY3.SA": "FLRY3.SA - FLEURY ON",
    "GOAU4.SA": "GOAU4.SA - GERDAU METALLURGICA PN",
    "HYPE3.SA": "HYPE3.SA - HYPERA ON",
    "IGTI11.SA": "IGTI11.SA - IGUATEMI UNT",
    "ISAE4.SA": "ISAE4.SA - ISA ENERGIA PN",
    "KLBN11.SA": "KLBN11.SA - KLABIN UNT",
    "MGLU3.SA": "MGLU3.SA - MAGAZINE LUIZA ON",
    "POMO4.SA": "POMO4.SA - MARCOPOLO PN",
    "MBRF3.SA": "MBRF3.SA - MARFRIG ON",
    "BEEF3.SA": "BEEF3.SA - MINERVA ON",
    "MRVE3.SA": "MRVE3.SA - MRV ON",
    "MULT3.SA": "MULT3.SA - MULTIPLAN ON",
    "NATU3.SA": "NATU3.SA - NATURA ON",
    "PSSA3.SA": "PSSA3.SA - PORTO SEGURO ON",
    "SMFT3.SA": "SMFT3.SA - SMART FIT ON",
    "TEND3.SA": "TEND3.SA - TENDA ON",
    "USIM5.SA": "USIM5.SA - USIMINAS PNA",
    "VAMO3.SA": "VAMO3.SA - VAMOS ON",
    "VIVA3.SA": "VIVA3.SA - VIVARA ON",
    "YDUQ3.SA": "YDUQ3.SA - YDUQS ON"
}

# ==========================================
# ENGENHARIA DE FEATURES
# ==========================================
@st.cache_data
def load_and_build_features(ticker: str):
    start_date = "2020-01-01"
    end_date = "2026-10-01"
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        return df
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    data = df.copy()

    # 1. TENDÊNCIA (EMA 8 e EMA 21)
    data['ema_8'] = data['Close'].ewm(span=8, adjust=False).mean()
    data['ema_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['dist_ema_8'] = (data['Close'] - data['ema_8']) / data['Close']
    data['dist_ema_21'] = (data['Close'] - data['ema_21']) / data['Close']
    data['ema_spread'] = (data['ema_8'] - data['ema_21']) / data['ema_21']

    # 2. MOMENTUM (RSI 14)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data['rsi_14'] = 100 - (100 / (1 + rs))

    # 3. VOLATILIDADE (ATR 14)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    data['atr_14'] = atr
    data['atr_pct'] = atr / data['Close']
    data['vol_ratio'] = atr / (atr.rolling(50).mean() + 1e-9)

    # TARGET (Alta nos próximos 3 períodos acima de 0.5%)
    FORWARD_PERIODS = 3
    THRESHOLD = 0.005
    future_return = data['Close'].shift(-FORWARD_PERIODS) / data['Close'] - 1
    data['target'] = (future_return > THRESHOLD).astype(int)

    return data.dropna()

# ==========================================
# BARRA LATERAL (APENAS SELEÇÃO DA AÇÃO)
# ==========================================
st.sidebar.header("Seleção do Ativo")
selected_label = st.sidebar.selectbox(
    "Escolha a ação para análise:",
    options=list(IBOV_TICKERS.values()),
    index=0
)

# Recupera a chave do ticker (ex: "PETR4.SA")
selected_ticker = [k for k, v in IBOV_TICKERS.items() if v == selected_label][0]

# Parâmetros padrão otimizados
proba_threshold = 0.60
atr_multiplier = 1.5
risk_reward = 2.0

# ==========================================
# EXECUÇÃO AUTOMÁTICA
# ==========================================
with st.spinner(f"Analisando {selected_label} com XGBoost..."):
    df_data = load_and_build_features(selected_ticker)

    if df_data.empty:
        st.error("Não foram encontrados dados para este ativo.")
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
        st.subheader(f"Análise Quantitativa: {selected_label}")
        
        last_proba = df_test['proba_alta'].iloc[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Último Fechamento", f"R$ {df_test['Close'].iloc[-1]:.2f}")
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
            m1.metric("Total de Trades", len(df_trades))
            m2.metric("Taxa de Acerto (Win Rate)", f"{win_rate:.1f}%")
            m3.metric("Retorno Acumulado", f"{cum_return*100:.2f}%")

            df_trades['Equity'] = (1 + df_trades['Return']).cumprod()
            fig_eq = px.line(df_trades, x='Exit Date', y='Equity', title="Curva de Patrimônio do Backtest")
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.warning("Nenhum trade disparado no período de teste para este ativo.")

        st.subheader("Importância das Variáveis no Modelo")
        imp = pd.Series(model.feature_importances_, index=feature_cols).reset_index()
        imp.columns = ['Indicador', 'Importância']
        fig_imp = px.bar(imp, x='Importância', y='Indicador', orientation='h')
        st.plotly_chart(fig_imp, use_container_width=True)
