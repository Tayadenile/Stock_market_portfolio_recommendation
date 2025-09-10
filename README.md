# AlphaVue_portfolio_recommendation_System

# 🌟 AlphaVue – Intelligent Investment Assistant

## Key Features
- **User Authentication**: Secure signup/login with bcrypt password hashing  
- **Personalized Risk Profiling**: AI-powered model predicts user's risk appetite *(Low, Medium, High)*  
- **Optimal Asset Allocation**: Balanced allocation across Equity, Debt, and Gold  
- **Dynamic Portfolio Recommendations**:  
  - Optimized stock portfolios *(Modern Portfolio Theory)*  
  - Top-performing mutual funds based on risk profile  
- **Portfolio Management**: Save, update, or delete user portfolios  
- **Admin Panel**: Manage users, monitor app statistics  
- **Live Market Data**: Snapshot of indices and trending stocks via `yfinance`  

---

## 🛠️ Technologies Used

| Component      | Technology |
|----------------|------------|
| **Frontend**   | Streamlit |
| **Backend & ML** | Python, Pandas, NumPy, Scikit-learn, Prophet, SciPy, joblib |
| **Database**   | MySQL |
| **Security**   | bcrypt, email-validator, regex-based password validation |
| **Visualization** | Streamlit UI, Plotly |
| **Data Sources** | JSON files, CSVs, yfinance API |
| **Hosting**    | Streamlit Community Cloud, AWS Elastic Beanstalk, AWS EC2 |

---

## 🧩 Core Functionalities

### 1. User Authentication
- Login/Signup with validation  
- Password strength enforcement  
- Email format validation  
- Password hashing using **bcrypt**  

### 2. Risk Profiling
- Collects user data: *Age, Investment Experience, Primary Goal, Market Reaction, Investment Horizon*  
- Predicts risk profile *(Low/Medium/High)* using **Random Forest Classifier**  
- Stores risk profile in `st.session_state`  

### 3. Asset Allocation
- Uses **K-Means clustering** to map users to investor personas  
- Retrieves recommended Equity/Debt/Gold percentages from allocation map  

### 4. Optimized Stock Portfolio
- Filters stocks based on risk profile  
- Uses **Modern Portfolio Theory (MPT)** to optimize weights for *Maximum Sharpe Ratio*  
- Calculates expected annual return, volatility, and projected investment returns  

### 5. Mutual Fund Recommendations
- Filters funds based on risk profile  
- Allocates lumpsum or SIP proportionally based on historical CAGR  
- Calculates projected returns over investment horizon  

### 6. Portfolio Management
- Save, update, retrieve, or delete portfolios in **MySQL**  
- Presents summary of stock & mutual fund allocations and performance metrics  

---

## 🔗 Database Schema

### 1. `users`
| Column Name   | Data Type | Description |
|---------------|-----------|-------------|
| user_id       | INT | Primary Key, unique user identifier |
| username      | VARCHAR | User's chosen username |
| email         | VARCHAR | User's email address |
| password_hash | VARCHAR | Hashed password using bcrypt |
| role          | VARCHAR | Role of the user (user or admin) |
| created_at    | TIMESTAMP | Timestamp of user creation |

### 2. `portfolios`
| Column Name   | Data Type | Description |
|---------------|-----------|-------------|
| portfolio_id  | INT | Primary Key |
| user_id       | INT | Foreign Key to users.user_id |
| age           | INT | User age |
| experience    | VARCHAR | Investment experience |
| primary_goal  | VARCHAR | Investment goal |
| risk_appetite | VARCHAR | Predicted risk profile (Low/Medium/High) |
| created_at    | TIMESTAMP | Timestamp of portfolio creation |

### 3. `portfolio_stocks`
| Column Name   | Data Type | Description |
|---------------|-----------|-------------|
| stock_record_id | INT | Primary Key |
| portfolio_id    | INT | Foreign Key to portfolios.portfolio_id |
| ticker          | VARCHAR | Stock symbol |
| invested_amount | DECIMAL | Amount invested in the stock |
| expected_return | DECIMAL | Projected return on investment |
| weight          | DECIMAL | Portfolio allocation weight |

### 4. `portfolio_mutual_funds`
| Column Name   | Data Type | Description |
|---------------|-----------|-------------|
| mf_record_id     | INT | Primary Key |
| portfolio_id     | INT | Foreign Key to portfolios.portfolio_id |
| fund_name        | VARCHAR | Mutual fund name |
| invested_amount  | DECIMAL | Amount invested |
| expected_return  | DECIMAL | Projected return |
| total_investment_sip | DECIMAL | Total SIP contributions (nullable) |
| weight           | DECIMAL | Portfolio allocation weight |

### 5. `portfolio_summary`
| Column Name   | Data Type | Description |
|---------------|-----------|-------------|
| summary_id    | INT | Primary Key |
| portfolio_id  | INT | Foreign Key to portfolios.portfolio_id |
| investment_period | VARCHAR | Investment horizon (years) |
| total_investment  | DECIMAL | Total invested amount |
| estimated_value   | DECIMAL | Projected portfolio value |
| profit_earned     | DECIMAL | Expected profit |
| return_rate       | DECIMAL | Return percentage |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+  
- Git  
- MySQL database  
- Internet access (for **yfinance API**)  

---

## 🚀 Future Enhancements
Planned improvements for **AlphaVue** include:  
- Real-time market updates and alerts  
- Integration with ETFs, bonds, and cryptocurrencies  
- Improved ML models for better risk profiling and asset allocation  
- Interactive portfolio growth visualization  
- Mobile-responsive UI enhancements  

---

## 📚 References
- [Python Official Documentation](https://docs.python.org/3/)  
- [Streamlit Documentation](https://docs.streamlit.io/)  
- [Scikit-learn Documentation](https://scikit-learn.org/)  
- [yfinance Documentation](https://pypi.org/project/yfinance/)  

---

## 📬 Contact
For questions, feedback, or collaboration:  

**Name**: Nilesh Kumar Tayade
**Email**: [nileshtayade2@gmail.com](mailto:nileshtayade2@gmail.co  
**GitHub**: [https://github.com/Tayadenile](https://github.com/Tayadenile)  

---

## 📜 License
This project is licensed under the **MIT License**. See the LICENSE file for details.

