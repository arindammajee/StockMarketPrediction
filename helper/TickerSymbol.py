import pandas as pd
import yahooquery as yq
import os
df = df = yq.get_exchanges()
currentDir = os.getcwd()
df.to_csv(os.path.join(currentDir, "ExchangeList.csv"))

class TickerSymbol:
    def __init__(self) -> None:
        pass

    def get_symbol(query, preferred_exchange='AMS'):
        try:
            data = yq.search(query)
        except ValueError: # Will catch JSONDecodeError
            print(query)
        else:
            quotes = data['quotes']
            if len(quotes) == 0:
                return 'No Symbol Found'

            symbol = quotes[0]['symbol']
            for quote in quotes:
                if quote['exchange'] == preferred_exchange:
                    symbol = quote['symbol']
                    break
            return symbol

companies = ['Amazon', 'TCS', 'Adanipower', 'Tatapower', 'Accenture', 'Tata Consultancy Service']
df = pd.DataFrame({'Company name': companies})
df['Company symbol'] = df.apply(lambda x: TickerSymbol.get_symbol(x['Company name']), axis=1)
df.to_csv(os.path.join(currentDir, "TickerList.csv"))