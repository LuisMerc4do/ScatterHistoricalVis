import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import matplotlib.dates as mdates
from scipy import stats # For sd

# References. https://datavizcatalogue.com/index.html
# Matplotlib dates https://matplotlib.org/stable/api/dates_api.html
# Scipy stats for sd https://www.nature.com/articles/s41592-019-0686-2

def load_and_preprocess_data():
    # Load and preprocess the weather data
    # Define column names based on the description
    column_names = [
        'Date', 'Avg_Temp', 'Min_Temp', 'Max_Temp',
        'Historical_Avg_Min', 'Historical_Avg_Max',
        'Historical_Lowest', 'Historical_Highest',
        'Rainfall', 'Historical_Avg_Rain', 'Historical_Highest_Rain'
    ]
    
    # Load the CSV file
    df = pd.read_csv('DMVA3T2.csv', header=None, names=column_names)
    # Parse date column
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
    
    # Extract date components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['MonthName'] = df['Date'].dt.strftime('%b')
    
    # Calculate temperature anomalies
    df['Avg_Temp_Deviation'] = df['Avg_Temp'] - (df['Historical_Avg_Min'] + df['Historical_Avg_Max'])/2
    df['Max_Temp_Deviation'] = df['Max_Temp'] - df['Historical_Avg_Max']
    df['Min_Temp_Deviation'] = df['Min_Temp'] - df['Historical_Avg_Min']
    
    # Calculate rainfall anomalies
    df['Rainfall_Deviation'] = df['Rainfall'] - df['Historical_Avg_Rain']
    
    # Create moving averages
    df['7d_Avg_Temp'] = df['Avg_Temp'].rolling(window=7, center=True).mean()
    df['7d_Rainfall'] = df['Rainfall'].rolling(window=7).sum()
    
    # Create cumulative rainfall
    df['Cumulative_Rainfall'] = df['Rainfall'].cumsum()
    df['Cumulative_Avg_Rain'] = df['Historical_Avg_Rain'].cumsum()
    
    # Extreme days, if standard deviation > 2 than the average
    df['Extreme_Temp_Day'] = (abs(stats.zscore(df['Avg_Temp'])) > 2)
    df['Extreme_Rain_Day'] = (df['Rainfall'] > df['Historical_Highest_Rain'] * 0.7)
    
    return df

# Decided to create separate functions for reusability and divisibility, might be unnecessary but using best coding practices
def plot_temperature_ovw(df):
    plt.figure(figsize=(12, 6))
    
    # Plot temperatures
    plt.plot(df['Date'], df['Max_Temp'], 'r-', alpha=0.7, label='Daily Maximum')
    plt.plot(df['Date'], df['Min_Temp'], 'b-', alpha=0.7, label='Daily Minimum')
    plt.plot(df['Date'], df['Avg_Temp'], 'g-', label='Daily Average')
    plt.plot(df['Date'], df['7d_Avg_Temp'], 'k-', linewidth=2, label='7-Day Moving Average')
    
    # Plot historical averages
    plt.plot(df['Date'], df['Historical_Avg_Max'], 'r--', alpha=0.5, label='Historical Avg Max')
    plt.plot(df['Date'], df['Historical_Avg_Min'], 'b--', alpha=0.5, label='Historical Avg Min')
    
    # Highlight extreme temperature days
    extreme_days = df[df['Extreme_Temp_Day']]
    if not extreme_days.empty:
        plt.scatter(extreme_days['Date'], extreme_days['Avg_Temp'], 
                   color='orange', s=50, label='Temperature Extremes')
    
    # Formatting
    plt.title('Annual Temperature Patterns with Historical Metrics')
    plt.ylabel('Temperature (°C)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Format x-axis to show months
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.tight_layout()
    
    plt.savefig('rslt_temperature_overview.png', dpi=300)
    return plt


def plot_temperature_anomalies(df):
    # Create a pivot table for monthly temperature anomalies
    pivot_temp = df.pivot_table(
        index='Day', columns='Month', values='Avg_Temp_Deviation', aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with diverging colors (blue=cold, red=warm)
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    sns.heatmap(pivot_temp, cmap='RdBu_r', norm=norm,
                cbar_kws={'label': 'Temperature Anomaly (°C)'})
    
    # Formatting
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.title('Temperature Anomalies Throughout Year')
    plt.xlabel('Month')
    plt.ylabel('Day of Month')
    plt.xticks(np.arange(12)+0.5, month_labels)
    
    plt.tight_layout()
    plt.savefig('rslt_temperature_anomalies.png', dpi=300)
    return plt


def plot_rainfall_analysis(df):
    # Create rainfall analysis plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Daily rainfall with historical context
    ax1.bar(df['Date'], df['Rainfall'], color='cornflowerblue', alpha=0.4, label='Daily Rainfall')
    ax1.plot(df['Date'], df['Historical_Avg_Rain'], color='steelblue', linestyle='--', label='Historical Average')

    # Highlight extreme rainfall days
    extreme_rain_days = df[df['Extreme_Rain_Day']]
    if not extreme_rain_days.empty:
        ax1.bar(extreme_rain_days['Date'], extreme_rain_days['Rainfall'], color='crimson', label='Heavy Rainfall')
    
    ax1.set_title('Daily Rainfall with Historical Context')
    ax1.set_ylabel('Rainfall (mm)')
    ax1.set_xlabel('Date')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Cumulative rainfall plot
    ax2.plot(df['Date'], df['Cumulative_Rainfall'], color='teal', linewidth=2, label='Actual Cumulative')
    ax2.plot(df['Date'], df['Cumulative_Avg_Rain'], color='darkslategray', linestyle='--', linewidth=2, label='Expected (Historical)')

    # Calculate and show the difference
    end_date = df['Date'].iloc[-1]
    actual_total = df['Cumulative_Rainfall'].iloc[-1]
    expected_total = df['Cumulative_Avg_Rain'].iloc[-1]
    percent_diff = ((actual_total - expected_total) / expected_total) * 100

    # Add annotation about the difference
    ax2.annotate(f'Total: {actual_total:.1f}mm ({percent_diff:.1f}% vs historical)',
                 xy=(end_date, actual_total), xytext=(-150, 20),
                 textcoords='offset points', ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.6))
    
    ax2.set_title('Cumulative Rainfall Comparison')
    ax2.set_ylabel('Total Rainfall (mm)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Format x-axis to show months in the second subplot
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('rslt_rainfall_analysis.png', dpi=300)
    return fig

def plot_climate_extremes(df):
    # Find days exceeding historical extremes
    record_high_days = df[df['Max_Temp'] > df['Historical_Highest']]
    record_low_days = df[df['Min_Temp'] < df['Historical_Lowest']]
    record_rain_days = df[df['Rainfall'] > df['Historical_Highest_Rain']]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Temperature records
    ax1.plot(df['Date'], df['Max_Temp'], 'r-', alpha=0.5, label='Daily Highest')
    ax1.plot(df['Date'], df['Min_Temp'], 'b-', alpha=0.5, label='Daily lowest')
    ax1.plot(df['Date'], df['Historical_Highest'], 'r--', alpha=0.7, label='Historical Highest')
    ax1.plot(df['Date'], df['Historical_Lowest'], 'b--', alpha=0.7, label='Historical Lowest')
    
    # Highlight new records
    if not record_high_days.empty:
        ax1.scatter(record_high_days['Date'], record_high_days['Max_Temp'], 
                  color='red', s=80, marker='^', label='New High Records')
    
    if not record_low_days.empty:
        ax1.scatter(record_low_days['Date'], record_low_days['Min_Temp'], 
                  color='blue', s=80, marker='v', label='New Low Records')
    
    ax1.set_title('Temperature Records')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Format x-axis to show months (set on ax1 since they sharex)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # Rainfall records
    ax2.bar(df['Date'], df['Rainfall'], color='green', alpha=0.6)
    ax2.plot(df['Date'], df['Historical_Highest_Rain'], 'r--', 
           alpha=0.7, label='Historical Highest')
    
    # Highlight new rainfall records
    if not record_rain_days.empty:
        ax2.bar(record_rain_days['Date'], record_rain_days['Rainfall'], 
              color='red', label='New Rain Records')
    
    ax2.set_title('Rainfall Records')
    ax2.set_ylabel('Rainfall (mm)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rslt_climate_extremes.png', dpi=300)
    return fig


def main():
    # Main function to run all analyses
    print("Starting weather data analysis...")
    
    # Load and preprocess the data
    df = load_and_preprocess_data()
    print(f"365 days of weather data loaded")
    # Generate all plots
    plot_temperature_ovw(df)
    plot_temperature_anomalies(df)
    plot_rainfall_analysis(df)
    plot_climate_extremes(df)
    
    print("Analysis complete. All plots saved.")


if __name__ == "__main__":
    main()