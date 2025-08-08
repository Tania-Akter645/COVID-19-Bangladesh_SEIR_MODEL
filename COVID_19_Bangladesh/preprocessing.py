#Author=Tania Akter Rahima
import pandas as pd
# here i load the dataset
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath) #filepath = COVID_19_Bangladesh
    print(df.columns) # for checking data looks

    df['date'] = pd.to_datetime(df['date'], dayfirst=True) #day comes first
    df = df.sort_values('date')
    #. Handle Missing Values
    df.ffill(inplace=True) # forward fill to fill any missing (NaN) values

# .diff() calculates the difference between the current and previous row.
# .fillna(0) handles the first row where no previous value exists.

    # Confirmed, Recovered, Deaths
    df['daily_confirmed'] = df['total_confirmed'].diff().fillna(0)
    df['daily_recovered'] = df['total_recovered'].diff().fillna(0)
    df['daily_deaths'] = df['total_deaths'].diff().fillna(0)

    # Newly Reported Confirmed & Deaths
    df['daily_new_confirmed'] = df['new_confirmed'].diff().fillna(0)
    df['daily_new_deaths'] = df['new_deaths'].diff().fillna(0)

    #  Active Cases
    df['daily_active'] = df['active'].diff().fillna(0)

    #Sample Collection
    df['daily_sample'] = df['daily_collected_sample'].diff().fillna(0)

    # Rates (% change daily)
    df['daily_infection_rate'] = df['infectionRate(%)'].diff().fillna(0)
    df['daily_recovery_rate'] = df['recoveryRate(%)'].diff().fillna(0)
    df['daily_mortality_rate'] = df['mortalityRate(%)'].diff().fillna(0)

    # Quarantine Data
    df['daily_quarantine'] = df['total_quarantine'].diff().fillna(0)
    df['daily_now_quarantine'] = df['now_in_quarantine'].diff().fillna(0)
    df['daily_released_quarantine'] = df['released_from_quarantine'].diff().fillna(0)

    return df  # for clean dataframe, i use return df

# this statement helps to Run this block only when this file is run directly, not when it's imported
if __name__ == "__main__":
    df = load_and_clean_data('data/COVID-19-Bangladesh.csv') # I collected this data from kaggle

    # use print for getting output
    print(df.head())
