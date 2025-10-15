COLOR A
py -3.10 -m data_sources.influx_to_sql --mode both --symbol BTCUSDT --exchange binance --timeframe 1h --start '-120d' --bucket prices --sqlite ".\exports\Binance_BTCUSDT_1h_fix.db"