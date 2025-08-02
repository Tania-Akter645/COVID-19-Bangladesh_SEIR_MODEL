#Author=Tania Akter Rahima
import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print(df.columns)

    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df = df.sort_values('date')
    df.ffill(inplace=True)

    df['daily_confirmed'] = df['total_confirmed'].diff().fillna(0)
    df['daily_recovered'] = df['total_recovered'].diff().fillna(0)
    df['daily_deaths'] = df['total_deaths'].diff().fillna(0)
    df['daily_new_confirmed'] = df['new_confirmed'].diff().fillna(0)
    df['daily_new_deaths'] = df['new_deaths'].diff().fillna(0)
    df['daily_active'] = df['active'].diff().fillna(0)
    df['daily_sample'] = df['daily_collected_sample'].diff().fillna(0)
    df['daily_infection_rate'] = df['infectionRate(%)'].diff().fillna(0)
    df['daily_recovery_rate'] = df['recoveryRate(%)'].diff().fillna(0)
    df['daily_mortality_rate'] = df['mortalityRate(%)'].diff().fillna(0)
    df['daily_quarantine'] = df['total_quarantine'].diff().fillna(0)
    df['daily_now_quarantine'] = df['now_in_quarantine'].diff().fillna(0)
    df['daily_released_quarantine'] = df['released_from_quarantine'].diff().fillna(0)

    return df

if __name__ == "__main__":
    df = load_and_clean_data('data/COVID-19-Bangladesh.csv')
    print(df.head())