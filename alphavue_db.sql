-- Pehle apne database se connect karein, maana uska naam 'portfolio_project1' hai
CREATE DATABASE IF NOT EXISTS alphavue;
USE alphavue;

-- Purane tables ko drop kar dein taaki naya schema theek se ban sake (ISSE AAPKA PURANA DATA DELETE HO JAYEGA!)
DROP TABLE IF EXISTS portfolio_stocks;
DROP TABLE IF EXISTS portfolio_mutual_funds;
DROP TABLE IF EXISTS portfolios;
DROP TABLE IF EXISTS users;

-- Users table banayein
CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL, -- Hashed password store karne ke liye
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Portfolios table banayein jismein user_id foreign key hoga
CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL, -- Users table se link karne ke liye
    stock_investment_amount DECIMAL(15, 2) NOT NULL,
    mf_investment_amount DECIMAL(15, 2) NOT NULL,
    horizon INT NOT NULL,
    risk_appetite VARCHAR(50) NOT NULL,
    mf_investment_mode VARCHAR(50) NOT NULL,
    saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE -- Users table se link
);

ALTER TABLE portfolios
ADD COLUMN age INT NOT NULL AFTER user_id,
ADD COLUMN experience VARCHAR(50) NOT NULL AFTER age,
ADD COLUMN primary_goal VARCHAR(100) NOT NULL AFTER experience,
ADD COLUMN market_reaction VARCHAR(100) NOT NULL AFTER primary_goal;

-- Portfolio_stocks table banayein
CREATE TABLE IF NOT EXISTS portfolio_stocks (
    stock_record_id INT AUTO_INCREMENT PRIMARY KEY,
    portfolio_id INT NOT NULL,
    ticker VARCHAR(50) NOT NULL,
    invested_amount DECIMAL(15, 2),
    expected_return_amount DECIMAL(15, 2),
    weight DECIMAL(5, 4),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id) ON DELETE CASCADE
);

-- Portfolio_mutual_funds table banayein
CREATE TABLE IF NOT EXISTS portfolio_mutual_funds (
    mf_record_id INT AUTO_INCREMENT PRIMARY KEY,
    portfolio_id INT NOT NULL,
    fund_name VARCHAR(255) NOT NULL,
    invested_amount DECIMAL(15, 2),
    expected_return_amount DECIMAL(15, 2),
    total_investment_sip DECIMAL(15, 2), -- Sirf SIP ke liye
    weight DECIMAL(5, 4),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id) ON DELETE CASCADE
);


select * from users;
select * from portfolios;
select * from portfolio_stocks;
select * from portfolio_mutual_funds;
