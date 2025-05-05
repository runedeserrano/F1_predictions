import fastf1
import pandas as pd

def get_full_qualifying_results(year, gp_name):
    fastf1.Cache.enable_cache("f1_cache")
    
    # Load session
    session = fastf1.get_session(year, gp_name, 'Q')
    session.load()

    # Get all laps
    laps = session.laps
    laps = laps.sort_values('LapStartTime').reset_index(drop=True)

    # Calculate time difference between laps
    laps['TimeDiff'] = laps['LapStartTime'].diff().dt.total_seconds()

    # Detect big gaps
    big_gaps = laps[laps['TimeDiff'] > 300]
    gap_indices = big_gaps.index.tolist()
    
    if len(gap_indices) < 2:
        raise ValueError("Not enough big gaps detected! (maybe red flag?)")
    
    # Split into Q1, Q2, Q3
    q1_laps = laps.loc[:gap_indices[0]].copy()
    q2_laps = laps.loc[gap_indices[0]+1 : gap_indices[1]].copy()
    q3_laps = laps.loc[gap_indices[1]+1 :].copy()

    # Helper to get best lap for drivers
    def get_best_laps(laps_df, stint_name):
        laps_df = laps_df.dropna(subset=["LapTime"])
        best_laps = laps_df.sort_values(["Driver", "LapTime"]).drop_duplicates(subset="Driver", keep="first").copy()
        best_laps["Stint"] = stint_name
        return best_laps[["Driver", "LapTime", "Stint"]]

    # Best laps for each part
    best_q1 = get_best_laps(q1_laps, "Q1")
    best_q2 = get_best_laps(q2_laps, "Q2")
    best_q3 = get_best_laps(q3_laps, "Q3")

    # Combine all
    full_results = pd.concat([best_q1, best_q2, best_q3])

    # Make Stint a categorical so Q3 > Q2 > Q1 automatically when sorting
    full_results["Stint"] = pd.Categorical(full_results["Stint"], categories=["Q1", "Q2", "Q3"], ordered=True)

    # Keep the best stint (Q3 better than Q2 better than Q1) and best lap
    full_results = full_results.sort_values(["Driver", "Stint", "LapTime"], ascending=[True, False, True])
    full_results = full_results.drop_duplicates(subset="Driver", keep="first")

    # Convert LapTime to seconds and round
    full_results = full_results.assign(LapTime=full_results["LapTime"].dt.total_seconds().round(3))

    # Final sort by best LapTime
    full_results = full_results.sort_values("LapTime").reset_index(drop=True)

    # Add a position column
    full_results.index += 1
    full_results.index.name = 'Position'

    return full_results

# Usage example
full_results = get_full_qualifying_results(2025, 'Miami')
print(full_results)
