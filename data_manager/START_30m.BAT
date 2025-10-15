COLOR A
py -3.10 -m data_sources.influx_to_sql --mode both --symbol BTCUSDT --exchange binance --timeframe 30m --start '-120d' --bucket prices --sqlite ".\exports\30m_influx.db"