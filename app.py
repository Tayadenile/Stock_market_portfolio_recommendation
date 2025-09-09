import streamlit as st
import pandas as pd
import joblib
import numpy as np
import scipy.optimize as sco
import plotly.express as px
import mysql.connector
import bcrypt
from email_validator import validate_email, EmailNotValidError
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AlphaVue",
    page_icon="🤖",
    layout="wide"
)

# --- DATABASE CONNECTION ---
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=st.secrets["connections"]["mysql"]["host"],
            user=st.secrets["connections"]["mysql"]["user"],
            password=st.secrets["connections"]["mysql"]["password"],
            database=st.secrets["connections"]["mysql"]["database"]
        )
        if connection.is_connected():
            return connection
    except mysql.connector.Error as err:
        st.error(f"Database Error: {err}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}. Please ensure your secrets.toml file is correctly configured.")
        return None

# --- DATABASE FUNCTIONS ---
def create_user(username, email, password):
    conn = get_db_connection()
    if not conn: return False
    cursor = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        cursor.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)", (username, email, hashed_password))
        conn.commit()
        return True
    except mysql.connector.Error:
        return False
    finally:
        cursor.close()
        conn.close()

def check_user(username, password):
    conn = get_db_connection()
    if not conn: return None
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
        return user
    return None

def save_portfolio(user_id, inputs, risk_profile, stock_df, mf_df):
    conn = get_db_connection()
    if not conn: return False
    cursor = conn.cursor()
    try:
        portfolio_sql = """
        INSERT INTO portfolios (user_id, age, experience, primary_goal, market_reaction, 
                                horizon, stock_investment_amount, mf_investment_amount, mf_investment_mode, risk_appetite)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        portfolio_data = (
            user_id, inputs['age'], inputs['experience'], inputs['primary_goal'], inputs['market_reaction'],
            inputs['investment_horizon'], inputs['stock_investment_amount'], inputs['mf_investment_amount'],
            inputs['mf_investment_style'], risk_profile
        )
        cursor.execute(portfolio_sql, portfolio_data)
        portfolio_id = cursor.lastrowid

        if not stock_df.empty:
            stock_sql = "INSERT INTO portfolio_stocks (portfolio_id, ticker, invested_amount, expected_return_amount, weight) VALUES (%s, %s, %s, %s, %s)"
            for _, row in stock_df.iterrows():
                stock_data = (portfolio_id, row['Stock Symbol'], row['Investment Amount (₹)'], row[f"Projected Return (₹)"], row['Allocation (%)'])
                cursor.execute(stock_sql, stock_data)

        if not mf_df.empty:
            mf_sql = "INSERT INTO portfolio_mutual_funds (portfolio_id, fund_name, invested_amount, expected_return_amount, total_investment_sip, weight) VALUES (%s, %s, %s, %s, %s, %s)"
            for _, row in mf_df.iterrows():
                total_sip = row.get('Total Contribution (₹)', None)
                mf_data = (portfolio_id, row['Fund Name'], row['Lumpsum Investment (₹)'] if 'Lumpsum Investment (₹)' in row else row['Monthly Investment (₹)'], row['Projected Return (₹)'], total_sip, row['Allocation (%)'])
                cursor.execute(mf_sql, mf_data)
        conn.commit()
        return True
    except mysql.connector.Error as err:
        st.error(f"Database save error: {err}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

def get_user_portfolio(user_id):
    conn = get_db_connection()
    if not conn: return None, None, None
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM portfolios WHERE user_id = %s ORDER BY saved_at DESC LIMIT 1", (user_id,))
        portfolio = cursor.fetchone()
        if not portfolio:
            return None, None, None
        
        cursor.execute("SELECT * FROM portfolio_stocks WHERE portfolio_id = %s", (portfolio['portfolio_id'],))
        stocks = pd.DataFrame(cursor.fetchall())
        
        cursor.execute("SELECT * FROM portfolio_mutual_funds WHERE portfolio_id = %s", (portfolio['portfolio_id'],))
        mfs = pd.DataFrame(cursor.fetchall())
        
        return portfolio, stocks, mfs
    finally:
        cursor.close()
        conn.close()

def delete_portfolio(portfolio_id):
    conn = get_db_connection()
    if not conn: return False
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM portfolios WHERE portfolio_id = %s", (portfolio_id,))
        conn.commit()
        return True
    except mysql.connector.Error as err:
        st.error(f"Delete Error: {err}")
        return False
    finally:
        cursor.close()
        conn.close()

def update_portfolio(portfolio_id, user_id, inputs, risk_profile, stock_df, mf_df):
    conn = get_db_connection()
    if not conn: return False
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM portfolio_stocks WHERE portfolio_id = %s", (portfolio_id,))
        cursor.execute("DELETE FROM portfolio_mutual_funds WHERE portfolio_id = %s", (portfolio_id,))

        update_sql = """
        UPDATE portfolios SET age=%s, experience=%s, primary_goal=%s, market_reaction=%s, horizon=%s,
        stock_investment_amount=%s, mf_investment_amount=%s, mf_investment_mode=%s, risk_appetite=%s
        WHERE portfolio_id = %s AND user_id = %s
        """
        update_data = (
            inputs['age'], inputs['experience'], inputs['primary_goal'], inputs['market_reaction'],
            inputs['investment_horizon'], inputs['stock_investment_amount'], inputs['mf_investment_amount'],
            inputs['mf_investment_style'], risk_profile, portfolio_id, user_id
        )
        cursor.execute(update_sql, update_data)

        if not stock_df.empty:
            stock_sql = "INSERT INTO portfolio_stocks (portfolio_id, ticker, invested_amount, expected_return_amount, weight) VALUES (%s, %s, %s, %s, %s)"
            for _, row in stock_df.iterrows():
                stock_data = (portfolio_id, row['Stock Symbol'], row['Investment Amount (₹)'], row[f"Projected Return (₹)"], row['Allocation (%)'])
                cursor.execute(stock_sql, stock_data)

        if not mf_df.empty:
            mf_sql = "INSERT INTO portfolio_mutual_funds (portfolio_id, fund_name, invested_amount, expected_return_amount, total_investment_sip, weight) VALUES (%s, %s, %s, %s, %s, %s)"
            for _, row in mf_df.iterrows():
                total_sip = row.get('Total Contribution (₹)', None)
                mf_data = (portfolio_id, row['Fund Name'], row['Lumpsum Investment (₹)'] if 'Lumpsum Investment (₹)' in row else row['Monthly Investment (₹)'], row['Projected Return (₹)'], total_sip, row['Allocation (%)'])
                cursor.execute(mf_sql, mf_data)

        conn.commit()
        return True
    except mysql.connector.Error as err:
        st.error(f"Update Error: {err}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

# --- HELPER FUNCTIONS ---
def calculate_sip_future_value(monthly_investment, annual_rate, investment_years):
    if annual_rate == 0: return monthly_investment * investment_years * 12, monthly_investment * investment_years * 12
    monthly_rate = annual_rate / 12 / 100
    months = investment_years * 12
    future_value = monthly_investment * ((((1 + monthly_rate)**months) - 1) / monthly_rate) * (1 + monthly_rate)
    total_investment = monthly_investment * months
    return future_value, total_investment

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_std, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    if p_std == 0: return float('inf')
    return -(p_ret - risk_free_rate) / p_std

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]
    result = sco.minimize(neg_sharpe_ratio, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# --- DATA LOADING ---
@st.cache_resource
def load_models_and_data():
    models = {"risk_profiler": joblib.load('data/risk_profiler_model.joblib'), "allocator": joblib.load('data/allocator_model.joblib'), "scaler": joblib.load('data/allocator_scaler.joblib'), "risk_columns": joblib.load('data/risk_model_columns.joblib'), "alloc_map": joblib.load('data/allocation_map.joblib')}
    return models

@st.cache_data
def load_csv_files():
    data_files = {"stocks_fc": pd.read_csv('data/stock_forecast_results.csv'), "stocks_risk": pd.read_csv('data/Stock_Risk_Categories.csv'), "mf_fc": pd.read_csv('data/mutual_fund_forecast_metrics_filtered.csv'), "mf_meta": pd.read_csv('data/Mutual_Fund_Metadata (1).csv').dropna(subset=['Scheme_Name']), "all_prices": pd.read_csv('data/all_stocks_close_prices.csv', index_col='Date', parse_dates=True), "mf_navs": pd.read_csv('data/combined_mutual_fund_navs.csv', index_col='date', parse_dates=True)}
    return data_files

models = load_models_and_data()
data_files = load_csv_files()
risk_profiler_model = models["risk_profiler"]
risk_model_columns = models["risk_columns"]
allocator_model = models["allocator"]
allocator_scaler = models["scaler"]
allocation_map = models["alloc_map"]
stocks_fc = data_files["stocks_fc"]
stocks_risk = data_files["stocks_risk"]
mf_fc = data_files["mf_fc"]
mf_meta = data_files["mf_meta"]
all_prices = data_files["all_prices"]
mf_navs = data_files["mf_navs"]

# --- PAGE DEFINITIONS ---

def login_signup_page():
    st.title("Welcome to AlphaVue!!")
    choice = st.radio("Choose an option:", ["Login", "Sign Up"], horizontal=True)
    if choice == "Login":
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                user = check_user(username, password)
                if user:
                    st.session_state['logged_in'] = True
                    st.session_state['user_id'] = user['user_id']
                    st.session_state['username'] = user['username']
                    st.rerun()
                else:
                    st.error("Incorrect username or password")
    elif choice == "Sign Up":
        with st.form("signup_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                if password != confirm_password:
                    st.error("Passwords do not match!")
                elif not re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$", password):
                    st.error("Password must be at least 8 characters long and include uppercase, lowercase, a number, and a special character.")
                else:
                    try:
                        validate_email(email)
                        if create_user(username, email, password):
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error("Username or Email already exists.")
                    except EmailNotValidError as e:
                        st.error(str(e))

def home_page():
    st.title("Your Dashboard")
    portfolio, stocks, mfs = get_user_portfolio(st.session_state['user_id'])
    
    if portfolio:
        st.header("Your Saved Recommendations")
        st.subheader("📈 Stock Allocation")
        # Rename columns for display
        stocks_display = stocks.drop(columns=['portfolio_id', 'stock_record_id']).rename(columns={
            'ticker': 'Stock Symbol',
            'invested_amount': 'Invested Amount (₹)',
            'expected_return_amount': 'Projected Return (₹)',
            'weight': 'Allocation (%)'
        })
        st.dataframe(stocks_display)

        st.subheader("💰 Mutual Fund Allocation")
        # Rename columns for display
        mfs_display = mfs.drop(columns=['portfolio_id', 'mf_record_id']).rename(columns={
            'fund_name': 'Fund Name',
            'invested_amount': 'Invested Amount (₹)',
            'expected_return_amount': 'Projected Return (₹)',
            'total_investment_sip': 'Total SIP Investment (₹)',
            'weight': 'Allocation (%)'
        })
        st.dataframe(mfs_display)
        
        st.subheader("✨ Final Summary")
        time_periods = sorted(list(set([1, 3, 5, portfolio['horizon']])))
        summary_data = []
        for t in time_periods:
            stock_lumpsum = float(portfolio['stock_investment_amount'])
            projected_stock_value = 0
            if not stocks.empty:
                saved_tickers = stocks['ticker'].tolist()
                saved_prices = all_prices[saved_tickers].copy()
                saved_prices.ffill(inplace=True)
                saved_prices.dropna(inplace=True)
                if not saved_prices.empty:
                    saved_returns = saved_prices.pct_change().mean() * 252
                    saved_weights = stocks['weight'].astype(float).values
                    portfolio_return = np.sum(saved_returns * saved_weights)
                    projected_stock_value = stock_lumpsum * (1 + portfolio_return) ** t

            projected_mf_value, total_mf_investment = 0, 0
            if not mfs.empty:
                mfs_with_cagr = pd.merge(mfs, mf_fc, left_on='fund_name', right_on='Fund Name')
                if portfolio['mf_investment_mode'] == 'Lumpsum (One-Time)':
                    total_mf_investment = float(portfolio['mf_investment_amount'])
                    for _, row in mfs_with_cagr.iterrows():
                        investment_per_fund = total_mf_investment * float(row['weight'])
                        cagr = row['Historical_5Y_CAGR (%)'] / 100
                        projected_mf_value += investment_per_fund * (1 + cagr) ** t
                else: # SIP
                    for _, row in mfs_with_cagr.iterrows():
                        sip_per_fund = float(portfolio['mf_investment_amount']) * float(row['weight'])
                        cagr = row['Historical_5Y_CAGR (%)']
                        fv, total_inv_per_fund = calculate_sip_future_value(sip_per_fund, cagr, t)
                        projected_mf_value += fv
                        total_mf_investment += total_inv_per_fund
            
            total_invested = stock_lumpsum + total_mf_investment
            total_projected_value = projected_stock_value + projected_mf_value
            total_profit = total_projected_value - total_invested
            percent_gain = (total_profit / total_invested) * 100 if total_invested > 0 else 0
            summary_data.append((f"{t} Years", f"₹{total_invested:,.2f}", f"₹{total_projected_value:,.2f}", f"₹{total_profit:,.2f}", f"{percent_gain:.2f}%"))
        
        summary_df = pd.DataFrame(summary_data, columns=["Investment Period", "Total Investment", "Estimated Value", "Profit Earned", "Return Rate (%)"])
        st.dataframe(summary_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Edit Recommendations"):
                st.session_state['edit_mode'] = True
                st.session_state['portfolio_to_edit'] = portfolio
                st.session_state['page'] = "Get Recommendations"
                st.rerun()
        with col2:
            if st.button("Delete Recommendations"):
                if delete_portfolio(portfolio['portfolio_id']):
                    st.success("Recommendations deleted.")
                    st.rerun()
                else:
                    st.error("Could not delete Recommendations.")
    else:
        st.info("You haven't saved any recommendations yet. Use the sidebar to get recommendations.")

def recommendation_page():
    edit_data = st.session_state.get('portfolio_to_edit', {}) if st.session_state.get('edit_mode') else {}
    st.header("Step 1: Enter Your Details")
    with st.form(key='user_profile_form'):
        st.subheader("About You")
        col1, col2 = st.columns(2)
        with col1: age = st.number_input("What is your age?", 18, 70, int(edit_data.get('age', 35)), 1)
        with col2: experience = st.selectbox("What is your investment experience?", ('Beginner', 'Intermediate', 'Advanced'), index=['Beginner', 'Intermediate', 'Advanced'].index(edit_data.get('experience', 'Beginner')))
        st.subheader("Your Investment Goals")
        col3, col4 = st.columns(2)
        with col3: primary_goal = st.selectbox("What is your primary investment goal?", ('Steady Growth', 'Capital Protection', 'Aggressive Wealth Creation'), index=['Steady Growth', 'Capital Protection', 'Aggressive Wealth Creation'].index(edit_data.get('primary_goal', 'Steady Growth')))
        with col4: market_reaction = st.selectbox("How would you react to a 20% market drop?", ('Do nothing', 'Buy more', 'Sell some', 'Sell all'), index=['Do nothing', 'Buy more', 'Sell some', 'Sell all'].index(edit_data.get('market_reaction', 'Do nothing')))
        st.subheader("Your Investment Plan")
        col5, col6 = st.columns(2)
        with col5:
            investment_horizon = st.number_input("How many years do you plan to invest for?", 1, 40, int(edit_data.get('horizon', 10)), 1)
            stock_investment_amount = st.number_input("How much do you want to invest in stocks (₹)?", 10000, value=int(edit_data.get('stock_investment_amount', 50000)), step=5000)
        with col6:
            mf_investment_style = st.radio("How do you want to invest in mutual funds?", ('Lumpsum (One-Time)', 'Monthly SIP'), index=['Lumpsum (One-Time)', 'Monthly SIP'].index(edit_data.get('mf_investment_mode', 'Lumpsum (One-Time)')), horizontal=True)
            mf_investment_amount = st.number_input("How much do you want to invest in mutual funds (₹)?", 500, value=int(edit_data.get('mf_investment_amount', 10000)), step=500)
        submit_label = "Update Profile" if st.session_state.get('edit_mode') else "Generate My Profile"
        submit_button = st.form_submit_button(label=submit_label)
        if submit_button:
            with st.spinner('Analyzing your profile...'):
                user_data = pd.DataFrame({'Age': [age], 'Primary_Goal': [primary_goal], 'Market_Drop_Reaction': [market_reaction], 'Investment_Experience': [experience]})
                user_data_encoded = pd.get_dummies(user_data)
                user_data_aligned = user_data_encoded.reindex(columns=risk_model_columns, fill_value=0)
                predicted_risk = risk_profiler_model.predict(user_data_aligned)[0]
                st.session_state['generated_recommendations'] = True
                st.session_state['risk_profile'] = predicted_risk
                st.session_state['user_inputs'] = {'age': age, 'experience': experience, 'primary_goal': primary_goal, 'market_reaction': market_reaction, 'investment_horizon': investment_horizon, 'stock_investment_amount': stock_investment_amount, 'mf_investment_amount': mf_investment_amount, 'mf_investment_style': mf_investment_style}

    if st.session_state.get('generated_recommendations', False):
        st.success(f"Profile Generated! Your predicted risk profile is: **{st.session_state['risk_profile']}**")
        st.markdown("---")
        
        # --- STEP 2: ASSET ALLOCATION ---
        risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
        user_profile_data = pd.DataFrame({'Age': [st.session_state['user_inputs']['age']],'Investment_Horizon_Yrs': [st.session_state['user_inputs']['investment_horizon']],'RiskProfile_Encoded': [risk_map[st.session_state['risk_profile']]]})
        scaled_data = allocator_scaler.transform(user_profile_data)
        cluster = allocator_model.predict(scaled_data)[0]
        allocation = allocation_map[cluster]
        st.header("Step 2: Your Recommended Asset Allocation")
        st.write("Based on your profile, we recommend allocating your investment capital as follows:")
        asset_icons = {"Equity": "📈", "Debt": "📄", "Gold": "🪙"}
        for asset_class, percentage in allocation.items():
            st.subheader(f"{asset_icons.get(asset_class, '💰')} {asset_class}: {percentage}%")
            st.progress(percentage)
        st.markdown("---")

        # --- STEP 3: SPECIFIC ASSET RECOMMENDATIONS ---
        st.header("Step 3: Your Personalized Portfolio Recommendations")
        
        # --- Stock Portfolio (Modern Portfolio Theory) ---
        st.subheader("📈 Optimized Stock Portfolio")
        st.info(f"**Disclaimer:** Projections are based on historical data and are for illustrative purposes only. Past performance is not indicative of future results.", icon="⚠️")

        with st.expander("Configure Your Stock Portfolio Optimization"):
            num_stocks_to_consider = st.slider("How many top historical performers should we analyze?", min_value=1, max_value=20, value=10, step=1, help="We will select the best stocks from this pool to build an optimized portfolio.")

        stock_risk_map = {'Low': ['Low Risk', 'Medium Risk'],'Medium': ['Medium Risk'],'High': ['Medium Risk', 'High Risk']}
        eligible_stock_categories = stock_risk_map[st.session_state['risk_profile']]
        eligible_stocks = stocks_risk[stocks_risk['Risk Category'].isin(eligible_stock_categories)]
        recommended_stocks = pd.merge(eligible_stocks, stocks_fc, on='Ticker')
        candidate_stocks = recommended_stocks.sort_values(by='Historical_5Y_CAGR', ascending=False).head(num_stocks_to_consider)
        candidate_tickers = candidate_stocks['Ticker'].tolist()
        prices_df = all_prices[candidate_tickers].copy()
        prices_df.ffill(inplace=True)
        prices_df.dropna(inplace=True) 

        if num_stocks_to_consider < 2:
            st.info("Portfolio optimization requires more than one stock. Here is your selected top stock.")
            if not candidate_stocks.empty:
                single_stock = candidate_stocks.iloc[0]
                st.subheader("📊 Recommended Allocation")
                st.metric(label=f"Stock Symbol: {single_stock['Ticker']}", value="100% Allocation")
                st.subheader("📋 Performance")
                st.metric("Historical 5Y CAGR", f"{single_stock['Historical_5Y_CAGR']:.2f}%")
            else:
                st.warning("No stocks found in this category.")
        elif prices_df.empty or len(prices_df.columns) < 2:
            st.error("Insufficient historical data for the selected stocks to build a portfolio. Please adjust filters or risk profile.")
        else:
            with st.spinner("Optimizing portfolio... this may take a moment."):
                returns = prices_df.pct_change()
                mean_returns = returns.mean()
                cov_matrix = returns.cov()
                risk_free_rate = 0.02
                optimal_portfolio = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
                optimal_weights = optimal_portfolio['x']
                std, ret = portfolio_annualised_performance(optimal_weights, mean_returns, cov_matrix)
                sharpe = (ret - risk_free_rate) / std
                st.success("✅ Portfolio Optimized!")
                col1, col2 = st.columns([2,1])
                with col1:
                    st.subheader("📊 Recommended Allocation")
                    investment_amounts = [w * st.session_state['user_inputs']['stock_investment_amount'] for w in optimal_weights]
                    annual_returns = mean_returns * 252
                    expected_returns_amt = [(amt * (1 + r)**st.session_state['user_inputs']['investment_horizon']) for amt, r in zip(investment_amounts, annual_returns)]
                    allocation_df = pd.DataFrame({'Stock Symbol': prices_df.columns, 'Investment Amount (₹)': investment_amounts, f"Projected Return (₹)": expected_returns_amt, 'Allocation (%)': optimal_weights})
                    st.session_state['stock_df_to_save'] = allocation_df.copy()
                    display_df = allocation_df.drop(columns=['Allocation (%)'])
                    display_df['Investment Amount (₹)'] = display_df['Investment Amount (₹)'].map('{:,.2f}'.format)
                    display_df[f"Projected Return (₹)"] = display_df[f"Projected Return (₹)"].map('{:,.2f}'.format)
                    st.dataframe(display_df, use_container_width=True)
                with col2:
                    st.subheader("📋 Overall Performance")
                    st.metric("Expected Annual Return", f"{ret*100:.2f}%")
                    st.metric("Annual Volatility (Risk)", f"{std*100:.2f}%")
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

        st.markdown("---")
        st.subheader("💰 Recommended Mutual Funds")
        num_mf = st.number_input("How many mutual funds should we recommend?", 1, 10, 4, 1)
        risk_meter_map = {'Low': ['Low to Moderate Risk'],'Medium': ['Moderate Risk', 'Moderately High Risk'],'High': ['High Risk', 'Very High Risk']}
        eligible_mf_categories = risk_meter_map[st.session_state['risk_profile']]
        eligible_mf_meta = mf_meta[mf_meta['Riskometer'].isin(eligible_mf_categories)]
        recommended_mf = pd.merge(eligible_mf_meta, mf_fc, left_on='Scheme_Name', right_on='Fund Name')
        top_mf = recommended_mf.sort_values(by='Historical_5Y_CAGR (%)', ascending=False).head(num_mf)
        st.write(f"Top {num_mf} recommended funds for you based on their historical performance:")
        st.info(f"**Disclaimer:** Past performance is not indicative of future results. Projections are based on historical data and are for illustrative purposes only.", icon="⚠️")
        if not top_mf.empty:
            investment_style = st.session_state['user_inputs']['mf_investment_style']
            total_investment_amount = st.session_state['user_inputs']['mf_investment_amount']
            investment_horizon = st.session_state['user_inputs']['investment_horizon']
            cagr_sum = top_mf['Historical_5Y_CAGR (%)'].sum()
            top_mf['weight'] = top_mf['Historical_5Y_CAGR (%)'] / cagr_sum if cagr_sum > 0 else 1 / len(top_mf)
            fund_names, investment_amounts, expected_returns, total_investments_sip = [], [], [], []
            for index, row in top_mf.iterrows():
                fund_names.append(row['Fund Name'])
                investment_per_fund = total_investment_amount * row['weight']
                if investment_style == 'Lumpsum (One-Time)':
                    investment_amounts.append(investment_per_fund)
                    future_value = investment_per_fund * (1 + (row['Historical_5Y_CAGR (%)'] / 100)) ** investment_horizon
                    expected_returns.append(future_value)
                else:
                    investment_amounts.append(investment_per_fund)
                    fv, total_inv = calculate_sip_future_value(investment_per_fund, row['Historical_5Y_CAGR (%)'], investment_horizon)
                    expected_returns.append(fv)
                    total_investments_sip.append(total_inv)
            mf_df_data = {'Fund Name': fund_names, 'Lumpsum Investment (₹)' if investment_style == 'Lumpsum (One-Time)' else 'Monthly Investment (₹)': investment_amounts}
            if investment_style == 'Monthly SIP': mf_df_data['Total Contribution (₹)'] = total_investments_sip
            mf_df_data['Projected Return (₹)'] = expected_returns
            mf_allocation_df = pd.DataFrame(mf_df_data)
            # Rename 'weight' to 'Allocation (%)' before concatenation
            top_mf = top_mf.rename(columns={'weight': 'Allocation (%)'})
            st.session_state['mf_df_to_save'] = pd.concat([mf_allocation_df, top_mf['Allocation (%)'].reset_index(drop=True)], axis=1)
            display_mf_df = mf_allocation_df.copy()
            display_mf_df.iloc[:, 1] = display_mf_df.iloc[:, 1].map('{:,.2f}'.format)
            if investment_style == 'Monthly SIP':
                display_mf_df.iloc[:, 2] = display_mf_df.iloc[:, 2].map('{:,.2f}'.format)
                display_mf_df.iloc[:, 3] = display_mf_df.iloc[:, 3].map('{:,.2f}'.format)
            else:
                display_mf_df.iloc[:, 2] = display_mf_df.iloc[:, 2].map('{:,.2f}'.format)
            st.dataframe(display_mf_df, use_container_width=True)
        else:
            st.warning("No mutual funds found in this category.")
        
        st.markdown("---")
        st.header("✨ Your Final Portfolio Summary")
        time_periods = sorted(list(set([1, 3, 5, st.session_state['user_inputs']['investment_horizon']])))
        summary_data = []
        for t in time_periods:
            stock_lumpsum = st.session_state['user_inputs']['stock_investment_amount']
            projected_stock_value = 0
            if 'optimal_weights' in locals() or 'optimal_weights' in globals():
                stock_portfolio_return = np.sum((returns.mean() * 252) * optimal_weights)
                projected_stock_value = stock_lumpsum * (1 + stock_portfolio_return) ** t
            elif not candidate_stocks.empty:
                single_stock_cagr = candidate_stocks.iloc[0]['Historical_5Y_CAGR'] / 100
                projected_stock_value = stock_lumpsum * (1 + single_stock_cagr) ** t
            projected_mf_value, total_mf_investment = 0, 0
            if not top_mf.empty:
                if st.session_state['user_inputs']['mf_investment_style'] == 'Lumpsum (One-Time)':
                    total_mf_investment = st.session_state['user_inputs']['mf_investment_amount']
                    for _, row in top_mf.iterrows():
                        investment_per_fund = st.session_state['user_inputs']['mf_investment_amount'] * row['Allocation (%)']
                        cagr = row['Historical_5Y_CAGR (%)'] / 100
                        projected_mf_value += investment_per_fund * (1 + cagr) ** t
                else:
                    for _, row in top_mf.iterrows():
                        sip_per_fund = st.session_state['user_inputs']['mf_investment_amount'] * row['Allocation (%)']
                        cagr = row['Historical_5Y_CAGR (%)']
                        fv, total_inv_per_fund = calculate_sip_future_value(sip_per_fund, cagr, t)
                        projected_mf_value += fv
                        total_mf_investment += total_inv_per_fund
            total_invested = stock_lumpsum + total_mf_investment
            total_projected_value = projected_stock_value + projected_mf_value
            total_profit = total_projected_value - total_invested
            percent_gain = (total_profit / total_invested) * 100 if total_invested > 0 else 0
            summary_data.append((f"{t} Years", f"₹{total_invested:,.2f}", f"₹{total_projected_value:,.2f}", f"₹{total_profit:,.2f}", f"{percent_gain:.2f}%"))
        summary_df = pd.DataFrame(summary_data, columns=["Investment Period", "Total Investment", "Estimated Value", "Profit Earned", "Return Rate (%)"])
        st.dataframe(summary_df, use_container_width=True)

        save_button_label = "Update Recommendations" if st.session_state.get('edit_mode') else "Save Recommendations"
        if st.button(save_button_label):
            stock_df = st.session_state.get('stock_df_to_save', pd.DataFrame())
            mf_df = st.session_state.get('mf_df_to_save', pd.DataFrame())
            if st.session_state.get('edit_mode'):
                if update_portfolio(edit_data['portfolio_id'], st.session_state['user_id'], st.session_state['user_inputs'], st.session_state['risk_profile'], stock_df, mf_df):
                    st.success("Portfolio updated successfully!")
                    st.session_state['edit_mode'] = False
                    st.session_state['page'] = "Home"
                    st.rerun()
            else:
                if save_portfolio(st.session_state['user_id'], st.session_state['user_inputs'], st.session_state['risk_profile'], stock_df, mf_df):
                    st.success("Portfolio saved successfully!")
                    st.session_state['generated_recommendations'] = False
                    st.session_state['page'] = "Home"
                    st.rerun()

def dashboard_page(dashboard_type):
    st.header(f"{dashboard_type} Dashboard")
    # IMPORTANT: Replace these with your actual Power BI embed URLs
    urls = {
        "Stock": r"https://app.powerbi.com/view?r=eyJrIjoiOTU2MjY5ZjgtNmVmZS00NDQ3LTk1OGUtNTY0OGRjN2UyODA0IiwidCI6IjE0ZjljNmYzLTIyMGUtNDA4Ni1iYzc5LTFlNjUxZTQwZDZhYiJ9",
        "MF": r"https://app.powerbi.com/view?r=eyJrIjoiNmE2ZWFlZGUtMjc2NC00M2E0LTgzMzAtODE5OTMxYjlkNDEzIiwidCI6IjE0ZjljNmYzLTIyMGUtNDA4Ni1iYzc5LTFlNjUxZTQwZDZhYiJ9"
    }
    st.markdown(f'<iframe title="{dashboard_type} Dashboard" width="100%" height="600" src="{urls[dashboard_type]}" frameborder="0" allowFullScreen="true"></iframe>', unsafe_allow_html=True)

# --- MAIN APP ROUTER ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login_signup_page()
else:
    with st.sidebar:
        st.header(f"Welcome, {st.session_state['username']}")
        
        if st.session_state.get('edit_mode', False):
            st.session_state['page'] = "Get Recommendations"
        
        page_options = ["Home", "Get Recommendations", "Stock Dashboard", "MF Dashboard"]
        try:
            current_page_index = page_options.index(st.session_state.get('page', 'Home'))
        except ValueError:
            current_page_index = 0

        page = st.radio("Navigation", page_options, index=current_page_index)

        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    if st.session_state.get('page') != page and not st.session_state.get('edit_mode'):
        st.session_state['page'] = page
        st.session_state['generated_recommendations'] = False
        st.rerun()

    if st.session_state.get('page') == "Home":
        home_page()
    elif st.session_state.get('page') == "Get Recommendations":
        recommendation_page()
    elif st.session_state.get('page') == "Stock Dashboard":
        dashboard_page("Stock")
    elif st.session_state.get('page') == "MF Dashboard":
        dashboard_page("MF")