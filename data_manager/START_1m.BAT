COLOR C
py -3.10 -m data_sources.influx_to_sql --mode both --symbol BTCUSDT --exchange binance --timeframe 1m --start '-120d' --bucket prices --sqlite ".\exports\1m_influx.db"