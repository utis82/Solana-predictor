import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from datetime import datetime
import os

st.set_page_config(
    page_title="Paper Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Style mobile-friendly
st.markdown("""
<style>
    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 10px;
        border-radius: 5px;
    }
    .metric-positive {
        color: #00ff00;
    }
    .metric-negative {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)

def load_state():
    """Charge l'état du bot"""
    if not os.path.exists('data/bot_state.json'):
        return None
    
    with open('data/bot_state.json', 'r') as f:
        return json.load(f)

def main():
    st.title("📈 Paper Trading Dashboard")
    
    # Bouton de refresh
    if st.button("🔄 Rafraîchir", use_container_width=True):
        st.rerun()
    
    # Chargement de l'état
    state = load_state()
    
    if state is None:
        st.error("❌ Aucune donnée disponible. Le bot n'a pas encore démarré.")
        st.info("Lance le bot avec : `python railway_bot.py`")
        return
    
    # Dernière mise à jour
    last_update = datetime.fromisoformat(state['last_update'])
    st.caption(f"Dernière MAJ : {last_update.strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Métriques principales
    initial_capital = 100000
    current_capital = state['capital']
    position = state['position']
    
    # Calcul du portfolio value actuel
    if state['portfolio_history']:
        latest = state['portfolio_history'][-1]
        current_price = latest['price']
        portfolio_value = latest['portfolio_value']
        roi = latest['roi']
    else:
        current_price = 0
        portfolio_value = current_capital
        roi = 0
    
    st.markdown("---")
    
    # Métriques en colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "💰 Portfolio Total",
            f"${portfolio_value:,.0f}",
            f"{roi:+.2f}%"
        )
        st.metric(
            "💵 Cash",
            f"${current_capital:,.0f}"
        )
    
    with col2:
        st.metric(
            "📊 Position SOL",
            f"{position:.4f}",
            f"${position * current_price:,.0f}" if current_price > 0 else "$0"
        )
        st.metric(
            "💸 Profit/Perte",
            f"${portfolio_value - initial_capital:+,.0f}"
        )
    
    st.markdown("---")
    
    # Graphique du portfolio
    st.subheader("📈 Évolution du Portfolio")
    
    if state['portfolio_history']:
        df_portfolio = pd.DataFrame(state['portfolio_history'])
        df_portfolio['timestamp'] = pd.to_datetime(df_portfolio['timestamp'])
        
        fig = go.Figure()
        
        # Ligne du portfolio
        fig.add_trace(go.Scatter(
            x=df_portfolio['timestamp'],
            y=df_portfolio['portfolio_value'],
            mode='lines',
            name='Portfolio',
            line=dict(color='#00D9FF', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 217, 255, 0.1)'
        ))
        
        # Ligne de capital initial
        fig.add_hline(
            y=initial_capital,
            line_dash="dash",
            line_color="gray",
            annotation_text="Capital initial"
        )
        
        # Trades sur le graphique
        if state['trades']:
            trades_df = pd.DataFrame(state['trades'])
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            
            buys = trades_df[trades_df['action'] == 'BUY']
            sells = trades_df[trades_df['action'].isin(['SELL', 'STOP_LOSS', 'TAKE_PROFIT'])]
            
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys['timestamp'],
                    y=[portfolio_value] * len(buys),  # Approximation
                    mode='markers',
                    name='Achat',
                    marker=dict(size=12, color='green', symbol='triangle-up')
                ))
            
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells['timestamp'],
                    y=[portfolio_value] * len(sells),
                    mode='markers',
                    name='Vente',
                    marker=dict(size=12, color='red', symbol='triangle-down')
                ))
        
        fig.update_layout(
            height=400,
            template='plotly_dark',
            hovermode='x unified',
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas encore d'historique disponible")
    
    st.markdown("---")
    
    # Historique des trades
    st.subheader("📋 Historique des Trades")
    
    if state['trades']:
        trades_df = pd.DataFrame(state['trades'])
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Affichage inversé (plus récent en premier)
        trades_display = trades_df.iloc[::-1].head(20).copy()
        
        for idx, trade in trades_display.iterrows():
            action = trade['action']
            timestamp = trade['timestamp'].strftime('%d/%m %H:%M')
            price = trade['price']
            predicted = trade['predicted_change']
            
            # Couleur selon action
            if action == 'BUY':
                emoji = "🟢"
                color = "green"
            elif action == 'SELL':
                emoji = "🔴"
                color = "red"
            elif action == 'STOP_LOSS':
                emoji = "🛑"
                color = "darkred"
            elif action == 'TAKE_PROFIT':
                emoji = "💚"
                color = "darkgreen"
            else:
                emoji = "⚪"
                color = "gray"
            
            # Raison
            if action == 'BUY':
                reason = f"Prédiction bullish {predicted:+.2f}%"
            elif action == 'SELL':
                reason = f"Prédiction bearish {predicted:+.2f}%"
            elif action == 'STOP_LOSS':
                pnl = trade.get('pnl_pct', 0)
                reason = f"Stop Loss déclenché ({pnl:+.2f}%)"
            elif action == 'TAKE_PROFIT':
                pnl = trade.get('pnl_pct', 0)
                reason = f"Take Profit déclenché ({pnl:+.2f}%)"
            else:
                reason = "Signal"
            
            # Affichage
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 2])
                
                with col1:
                    st.markdown(f"### {emoji}")
                
                with col2:
                    st.markdown(f"**{action}**")
                    st.caption(timestamp)
                
                with col3:
                    st.write(f"${price:.2f}")
                    st.caption(reason)
                
                if 'pnl' in trade and trade['pnl'] != 0:
                    pnl_color = "green" if trade['pnl'] > 0 else "red"
                    st.markdown(
                        f"<span style='color:{pnl_color}'>P/L: ${trade['pnl']:+,.2f} ({trade['pnl_pct']:+.2f}%)</span>",
                        unsafe_allow_html=True
                    )
                
                st.markdown("---")
        
        # Stats des trades
        st.subheader("📊 Statistiques")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_trades = len(trades_df)
            st.metric("Trades totaux", total_trades)
        
        with col2:
            buys = len(trades_df[trades_df['action'] == 'BUY'])
            st.metric("Achats", buys)
        
        with col3:
            sells = len(trades_df[trades_df['action'].isin(['SELL', 'STOP_LOSS', 'TAKE_PROFIT'])])
            st.metric("Ventes", sells)
        
        # Win rate
        if 'pnl' in trades_df.columns:
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            total_completed = winning_trades + losing_trades
            
            if total_completed > 0:
                win_rate = (winning_trades / total_completed) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                with col2:
                    st.metric("Gagnants", winning_trades, delta_color="normal")
                
                with col3:
                    st.metric("Perdants", losing_trades, delta_color="inverse")
                
                # Total P/L
                total_pnl = trades_df['pnl'].sum()
                st.metric("P/L Total Réalisé", f"${total_pnl:+,.2f}")
    else:
        st.info("Aucun trade exécuté pour le moment")
    
    st.markdown("---")
    
    # Footer
    st.caption("Dashboard actualisé automatiquement par le bot")
    st.caption("Le bot vérifie le marché toutes les heures")

if __name__ == "__main__":
    main()